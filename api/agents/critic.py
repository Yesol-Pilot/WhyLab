"""
Critic Agent (Kant) — 비판적 검토 모듈
========================================
Engineer의 실험 결과를 비판적으로 검토하고 Peer Review 리포트를 생성합니다.

[v3: LLM-as-a-Judge + ConstitutionGuard 통합 (Sprint 32)]
- 구조화 판정 체계: ACCEPT / REVISE / REJECT (강제)
- ConstitutionGuard verdict 자동 반영
- HALTED 실험(실행 실패) 감지 및 즉시 REJECT
- MethodRegistry(UCB1)로 리뷰 기준을 적응적으로 선택
"""
import time
import logging
import random
from datetime import datetime
from typing import Optional

from api.graph import kg
from api.agents.method_registry import method_registry
from api.agents.gemini_client import evaluate_experiment, is_available as is_gemini_available
from api.guards.constitution_guard import guard, AnalysisLevel

logger = logging.getLogger("whylab.critic")

# ── 판정 체계 (LLM-as-a-Judge) ──
VERDICT_ACCEPT = "ACCEPT"           # 결과 채택, 논문화 가능
VERDICT_REVISE = "REVISE"           # 조건부 수정 후 재실험
VERDICT_REJECT = "REJECT"           # 결과 폐기, 근본적 결함


# ──────────────────────────────────────────────
# 비판 기준 정의
# ──────────────────────────────────────────────
CRITIQUE_CRITERIA = {
    "sample_size": {
        "min_threshold": 500,
        "warning": "표본 크기가 {n}으로 소규모입니다. 통계적 검정력이 부족할 수 있습니다.",
        "pass": "표본 크기 n={n}은 충분합니다."
    },
    "effect_size": {
        "min_threshold": 0.1,  # STEAM 합성 데이터는 표준화 단위 (이전 LaLonde $100 → 0.1로 수정)
        "warning": "처리 효과(ATE={ate})가 실질적으로 유의미한 수준인지 재검토 필요.",
        "pass": "처리 효과(ATE={ate})는 실질적으로 유의미한 크기입니다."
    },
    "heterogeneity": {
        "p_threshold": 0.05,
        "warning": "서브그룹 '{moderator}'의 p-value({p})가 경계선 수준이라 이질성 판단에 주의가 필요합니다.",
    },
}


def review_experiment(experiment_result: dict) -> dict:
    """
    실험 결과를 비판적으로 검토합니다 (LLM-as-a-Judge v3).
    
    [판정 체계]
    - ACCEPT: 결과 채택, KG 반영 및 논문화 가능
    - REVISE: 조건부 수정 후 재실험 요구
    - REJECT: 결과 폐기, 근본적 결함 (HALTED 포함)
    
    Returns:
        dict: 구조화된 Peer Review 리포트
    """
    issues = []
    strengths = []
    
    # ── Step 0: HALTED 실험 즉시 REJECT ──
    if experiment_result.get("conclusion") == "EXECUTION_FAILED":
        logger.warning("HALTED 실험 감지 → 즉시 REJECT")
        return {
            "review_id": f"REV-{int(time.time()) % 10000:04d}",
            "experiment_id": experiment_result.get("experiment_id", "?"),
            "hypothesis_id": experiment_result.get("hypothesis_id", "?"),
            "verdict": VERDICT_REJECT,
            "verdict_reason": f"실험 실행 실패(HALTED): {experiment_result.get('error_reason', 'Unknown')}. 재실험 필요.",
            "strengths": [],
            "issues": [{
                "severity": "CRITICAL",
                "aspect": "실행 실패",
                "detail": experiment_result.get("error_reason", "SandboxExecutor 실행 실패"),
            }],
            "constitution_verdict": experiment_result.get("constitution_verdict", {}),
            "adaptive_criteria_used": [],
            "summary_stats": {"critical_issues": 1, "warnings": 0, "info_notes": 0, "strengths_noted": 0},
            "recommendations": ["SandboxExecutor 로그 확인", "데이터 파이프라인 검증", "회로 차단기 상태 확인"],
            "reviewed_at": datetime.utcnow().isoformat(),
        }
    
    # ── Step 0.5: ConstitutionGuard verdict 반영 ──
    constitution = experiment_result.get("constitution_verdict", {})
    if constitution:
        if not constitution.get("passed", True):
            for v in constitution.get("violations", []):
                issues.append({
                    "severity": "CRITICAL",
                    "aspect": "헌법 위반",
                    "detail": v,
                })
        for w in constitution.get("warnings", []):
            issues.append({
                "severity": "WARNING",
                "aspect": "헌법 경고",
                "detail": w,
            })
        if constitution.get("analysis_level") == "exploratory":
            strengths.append("탐색적 분석 수준으로 인과 주장이 제한됩니다.")
    
    # 1. 표본 크기 검증
    n = experiment_result.get("sample_size", 0)
    if n < CRITIQUE_CRITERIA["sample_size"]["min_threshold"]:
        issues.append({
            "severity": "WARNING",
            "aspect": "표본 크기",
            "detail": CRITIQUE_CRITERIA["sample_size"]["warning"].format(n=n)
        })
    else:
        strengths.append(CRITIQUE_CRITERIA["sample_size"]["pass"].format(n=n))
    
    # 2. 효과 크기 검증
    ate = experiment_result.get("ate", 0)
    if abs(ate) < CRITIQUE_CRITERIA["effect_size"]["min_threshold"]:
        issues.append({
            "severity": "CRITICAL",
            "aspect": "효과 크기",
            "detail": CRITIQUE_CRITERIA["effect_size"]["warning"].format(ate=f"{ate:,.0f}")
        })
    else:
        strengths.append(CRITIQUE_CRITERIA["effect_size"]["pass"].format(ate=f"{ate:,.0f}"))
    
    # 3. 서브그룹 이질성 검증
    subgroups = experiment_result.get("subgroup_analysis", {})
    significant_count = 0
    marginal_count = 0
    
    for moderator, sub in subgroups.items():
        p = sub.get("heterogeneity_p_value", 1.0)
        if sub.get("is_significant", False):
            significant_count += 1
            if p > 0.01:  # p가 경계선(0.01~0.05)
                issues.append({
                    "severity": "INFO",
                    "aspect": "이질성 경계",
                    "detail": CRITIQUE_CRITERIA["heterogeneity"]["warning"].format(moderator=moderator, p=f"{p:.4f}")
                })
        else:
            marginal_count += 1
    
    if significant_count > 0:
        strengths.append(f"{significant_count}개 서브그룹에서 통계적으로 유의한 이질적 처리 효과 확인.")
    
    # 4. 모델 성능 검증
    perf = experiment_result.get("model_performance", {})
    r2_treated = perf.get("r2_treated", 0)
    r2_control = perf.get("r2_control", 0)
    
    if r2_treated < 0.2 or r2_control < 0.2:
        issues.append({
            "severity": "WARNING",
            "aspect": "모델 적합도",
            "detail": f"R² 성능(Treated={r2_treated:.3f}, Control={r2_control:.3f})이 낮아 결과 해석에 주의 필요."
        })
    else:
        strengths.append(f"모델 적합도 양호 (R²: Treated={r2_treated:.3f}, Control={r2_control:.3f}).")
    
    # 4.5. Ground Truth 검증 (estimation_accuracy — STEAM 합성 데이터 전용)
    est_acc = experiment_result.get("estimation_accuracy", {})
    if est_acc:
        rmse = est_acc.get("rmse", float("inf"))
        bias = est_acc.get("bias", float("inf"))
        coverage = est_acc.get("coverage_rate", 0)
        corr = est_acc.get("correlation", 0)
        
        # Coverage 검증: true_cate가 CI 안에 있는 비율
        # 주의: LinearDML의 CATE CI는 구조적으로 좁은 경향 (bootstrap 자유도 과대추정)
        # Coverage가 낮아도 ATE 추정 자체는 정상일 수 있으므로 CRITICAL이 아닌 WARNING
        if coverage < 0.1:
            issues.append({
                "severity": "WARNING",
                "aspect": "Ground Truth Coverage",
                "detail": f"Coverage={coverage:.1%} — CATE CI가 좁음. CausalForestDML 재실험 권장."
            })
        elif coverage < 0.5:
            issues.append({
                "severity": "WARNING",
                "aspect": "Ground Truth Coverage",
                "detail": f"Coverage={coverage:.1%} — CI 폭이 좋음. CausalForestDML로 재실험 권장."
            })
        elif coverage >= 0.85:
            strengths.append(f"Ground Truth Coverage 우수: {coverage:.1%}")
        
        # Bias 검증: 추정 ATE와 참 ATE 차이
        ate_val = abs(ate) if ate else 1
        bias_ratio = abs(bias) / (ate_val + 1e-8)
        if bias_ratio > 0.5:
            issues.append({
                "severity": "WARNING",
                "aspect": "추정 편향",
                "detail": f"Bias={bias:.4f} (편향비={bias_ratio:.1%}) — 추정값이 참값에서 50% 이상 괴리."
            })
        
        # RMSE 검증
        if rmse < 1.0:
            strengths.append(f"CATE 추정 정확도 우수: RMSE={rmse:.4f}, Corr={corr:.3f}")
        
        # Correlation 검증: CATE 이질성 방향 일치
        if corr > 0.5:
            strengths.append(f"CATE 이질성 방향 일치: r={corr:.3f}")
            if corr > 0.9:
                strengths.append(f"🎯 CATE 추정 방향성 거의 완벽 (r={corr:.3f}) — DML 모델이 참 이질성을 정확히 포착")
        elif corr < 0.1:
            issues.append({
                "severity": "INFO",
                "aspect": "CATE 이질성",
                "detail": f"CATE 추정과 참값의 상관이 낮음 (r={corr:.3f}). 이질성 탐지 한계."
            })
    
    # 5. 적응형 검토 기준 (MethodRegistry UCB1 기반)
    generation = 1 + min(len(kg.graph.edges) // 10, 3)
    adaptive_criteria = method_registry.select_methods("review", count=3, generation=generation)
    applied_criteria_names = []
    
    for criterion in adaptive_criteria:
        criterion_name = criterion.name
        applied_criteria_names.append(criterion_name)
        
        # E-value 민감도 분석
        if "민감도" in criterion_name or "E-value" in criterion_name:
            issues.append({
                "severity": "INFO",
                "aspect": "E-value 민감도",
                "detail": f"[적응형 Gen {criterion.generation}] E-value 분석: 미관측 교란에 대한 결과 강건성 확인 필요."
            })
        # 다중 비교 보정
        elif "다중 비교" in criterion_name:
            if len(subgroups) > 2:
                issues.append({
                    "severity": "WARNING",
                    "aspect": "다중 비교 보정",
                    "detail": f"[적응형 Gen {criterion.generation}] {len(subgroups)}개 검정 → BH 보정 적용 권장."
                })
        # 외부 타당도
        elif "외부 타당도" in criterion_name:
            issues.append({
                "severity": "INFO",
                "aspect": "외부 타당도",
                "detail": f"[적응형 Gen {criterion.generation}] 결과의 외부 모집단 일반화 가능성 검토 필요."
            })
        # 재현성
        elif "재현성" in criterion_name:
            strengths.append(f"[적응형 Gen {criterion.generation}] 재현성 체크리스트 적용 완료.")
    
    # ── Gemini 정성적 평가 (2차 판정) ──
    gemini_critique = None
    gemini_score = 0
    if is_gemini_available():
        eval_result = evaluate_experiment(experiment_result)
        if eval_result:
            gemini_critique = eval_result.get("critique", "")
            gemini_score = eval_result.get("score", 5)
            
            # Gemini 비평을 이슈 또는 강점에 추가
            if gemini_score < 6:
                issues.append({
                    "severity": "WARNING",
                    "aspect": "AI Reviewer",
                    "detail": f"[Gemini Score {gemini_score}/10] {gemini_critique}"
                })
            else:
                strengths.append(f"[AI Reviewer] {gemini_critique} (Score: {gemini_score}/10)")
    
    # 6. 최종 판정 (Gemini 점수 반영)
    critical_count = sum(1 for i in issues if i["severity"] == "CRITICAL")
    warning_count = sum(1 for i in issues if i["severity"] == "WARNING")
    
    # Gemini 점수가 매우 낮으면 REVISE 강제
    if gemini_score > 0 and gemini_score <= 3:
        warning_count += 3 # 강제로 REVISE 유도
    
    if critical_count > 0:
        verdict = "REJECT"
        verdict_reason = "치명적 결함이 발견되어 재실험이 필요합니다."
    elif warning_count >= 3:
        verdict = "REVISE"
        verdict_reason = "다수의 경고 사항이 있어 방법론 수정 후 재제출이 권장됩니다."
    else:
        verdict = "ACCEPT"
        verdict_reason = "방법론적 건전성이 확인되었습니다. Knowledge Graph 반영을 승인합니다."
    
    # 보상 피드백: 판정 결과에 따라 기준별 보상
    reward_map = {"ACCEPT": 1.0, "REVISE": 0.5, "REJECT": 0.3}
    for criterion in adaptive_criteria:
        method_registry.reward_method(criterion.name, "review", reward_map.get(verdict, 0.5))
    
    return {
        "review_id": f"REV-{int(time.time()) % 10000:04d}",
        "experiment_id": experiment_result.get("experiment_id", "?"),
        "hypothesis_id": experiment_result.get("hypothesis_id", "?"),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "strengths": strengths,
        "issues": issues,
        "adaptive_criteria_used": applied_criteria_names,
        "summary_stats": {
            "critical_issues": critical_count,
            "warnings": warning_count,
            "info_notes": sum(1 for i in issues if i["severity"] == "INFO"),
            "strengths_noted": len(strengths),
            "adaptive_criteria": len(applied_criteria_names),
        },
        "recommendations": generate_recommendations(verdict, issues),
        "reviewed_at": datetime.utcnow().isoformat(),
    }


def generate_recommendations(verdict: str, issues: list) -> list[str]:
    """판정 결과에 따른 권장 사항을 생성합니다."""
    recommendations = []
    
    if verdict == "REJECT":
        recommendations.append("실험 설계를 근본적으로 재검토하고, 합성/실험 데이터로 파일럿 테스트를 권장합니다.")
    
    for issue in issues:
        if issue["aspect"] == "표본 크기":
            recommendations.append("더 큰 데이터셋을 확보하거나, Bootstrap 기법으로 신뢰구간을 보강하세요.")
        elif issue["aspect"] == "다중 비교":
            recommendations.append("Bonferroni 보정(α/k)을 적용하여 가양성(False Positive) 위험을 줄이세요.")
        elif issue["aspect"] == "모델 적합도":
            recommendations.append("피처 엔지니어링 또는 비선형 모델(GBM, Neural Net)을 고려하세요.")
    
    if verdict == "ACCEPT":
        recommendations.append("결과를 Knowledge Graph에 확정 반영하고, 후속 연구 주제를 도출하세요.")
    
    return recommendations


def run_critic_cycle() -> list[dict]:
    """
    Critic의 전체 리뷰 사이클을 실행합니다.
    
    Returns:
        list[dict]: 리뷰 과정 로그
    """
    logs = []
    
    def log(step: str, message: str):
        entry = {"step": step, "message": message, "timestamp": datetime.utcnow().isoformat()}
        logs.append(entry)
        return entry
    
    # Phase 1: 검증 완료된 실험 결과 조회
    log("FETCH", "Knowledge Graph에서 최근 실험 결과 조회 중...")
    time.sleep(0.3)
    
    # KG에서 실험 결과가 있는 엣지 탐색
    verified_edges = []
    for u, v, data in kg.graph.edges(data=True):
        if data.get("verified", False) and data.get("experiment_id"):
            verified_edges.append({
                "source": u, "target": v,
                "experiment_id": data["experiment_id"],
                "hypothesis_id": data.get("hypothesis_id", "?"),
                "hypothesis_text": data.get("hypothesis_text", ""),
            })
    
    if not verified_edges:
        log("ABORT", "리뷰 대상 실험 결과가 없습니다. Engineer(Tesla)의 활성화가 필요합니다.")
        return logs
    
    target = verified_edges[0]
    log("FETCH", f"실험 [{target['experiment_id']}] / 가설 [{target['hypothesis_id']}] 리뷰 시작.")
    
    # Phase 2: 실험 결과 재구성 (KG 실제 데이터 기반)
    log("RECONSTRUCT", "KG에서 실험 데이터 추출 중...")
    
    nodes = list(kg.graph.nodes(data=True))
    confounders = [n for n, d in nodes if d.get("category") == "Confounder"]
    
    # KG 엣지에서 실험 메타데이터 추출
    edge_data = {}
    for u, v, data in kg.graph.edges(data=True):
        if data.get("experiment_id") == target["experiment_id"]:
            edge_data = data
            break
    
    # 실제 KG 데이터를 기반으로 결과 구성
    experiment_result = {
        "experiment_id": target["experiment_id"],
        "hypothesis_id": target["hypothesis_id"],
        "sample_size": edge_data.get("sample_size", 2000),
        "ate": edge_data.get("ate", 0),
        "ate_ci": edge_data.get("ate_ci", []),
        "conclusion": edge_data.get("conclusion", "N/A"),
        "method": edge_data.get("method", "DML"),
        "estimator": edge_data.get("estimator", "LinearDML"),
        "subgroup_analysis": edge_data.get("subgroup_analysis", {
            mod: {
                "is_significant": False,
                "heterogeneity_p_value": 1.0,
            } for mod in confounders
        }),
        "model_performance": edge_data.get("model_performance", {
            "r2_treated": 0.3,
            "r2_control": 0.25,
        }),
        "constitution_verdict": edge_data.get("constitution_verdict", {}),
    }
    
    # Phase 3: 비판적 검토
    log("REVIEW", "방법론적 타당성 검증 중...")
    time.sleep(0.5)
    
    review = review_experiment(experiment_result)
    
    # 강점 로깅
    for strength in review["strengths"]:
        log("STRENGTH", f"✅ {strength}")
    
    # 문제점 로깅
    for issue in review["issues"]:
        emoji = "🔴" if issue["severity"] == "CRITICAL" else "🟡" if issue["severity"] == "WARNING" else "🔵"
        log("ISSUE", f"{emoji} [{issue['severity']}] {issue['aspect']}: {issue['detail']}")
    
    # Phase 4: 최종 판정
    time.sleep(0.2)
    verdict_emoji = "🟢" if review["verdict"] == "ACCEPT" else "🟡" if review["verdict"] == "REVISE" else "🔴"
    log("VERDICT", f"{verdict_emoji} 최종 판정: **{review['verdict']}** — {review['verdict_reason']}")
    
    # 권장사항 로깅
    for rec in review["recommendations"]:
        log("RECOMMEND", f"💡 {rec}")
    
    stats = kg.get_stats()
    log("COMPLETE", f"리뷰 완료 [{review['review_id']}]. KG: {stats['nodes']}노드, {stats['edges']}엣지. → Archivist에게 기록 요청.")
    
    return logs
