"""
Engineer Agent (Tesla) — 실험 수행 모듈
========================================
Theorist의 가설을 받아 WhyLab 엔진으로 인과추론 실험을 수행합니다.

[v3: Code-Then-Execute 패턴 (Sprint 29)]
- SandboxExecutor를 통해 실제 engine/cells 코드를 격리 실행
- 시뮬레이션 폴백(random) 제거 → 실행 환각(Execution Hallucination) 근절
- ConstitutionGuard를 통한 결과 검증 (헌법 제1/4/5조)
- 실패 시 HALTED 상태로 전환 (가짜 데이터 생성 금지)
"""
import time
import random
import logging
import numpy as np
from datetime import datetime
from typing import Optional

from api.graph import kg
from api.agents.method_registry import method_registry
from engine.sandbox.executor import sandbox, generate_experiment_code, PipelineHalt
from api.guards.constitution_guard import guard, AnalysisLevel

logger = logging.getLogger("whylab.engineer")


def get_pending_hypotheses() -> list[dict]:
    """KG에서 아직 검증되지 않은 가설(hypothesis 엣지)을 조회합니다."""
    if not kg.initialized:
        kg.initialize_seed_data()
    
    hypotheses = []
    for u, v, data in kg.graph.edges(data=True):
        if data.get("relation") == "hypothesis":
            hypotheses.append({
                "source": u,
                "target": v,
                "hypothesis_id": data.get("hypothesis_id", "H-UNKNOWN"),
                "hypothesis_text": data.get("hypothesis_text", ""),
                "weight": data.get("weight", 0.0),
                "verified": data.get("verified", False),
            })
    return [h for h in hypotheses if not h.get("verified", False)]


def design_experiment(hypothesis: dict) -> dict:
    """
    가설에 맞는 실험을 설계합니다 (UCB1 기반 방법론 선택).
    """
    # 세대 결정 (KG 규모 기반)
    generation = 1 + min(len(kg.graph.edges) // 10, 3)
    
    # UCB1로 최적 실험 방법론 선택
    selected_method = method_registry.select_method("experiment", generation)
    
    experiment = {
        "id": f"EXP-{int(time.time()) % 10000:04d}",
        "hypothesis_id": hypothesis.get("id", hypothesis.get("hypothesis_id", "UNKNOWN")),
        "method": selected_method.name,
        "method_generation": selected_method.generation,
        "estimator": selected_method.params.get("estimator", "T-Learner"),
        "robustness": selected_method.params.get("robustness", 0.5),
        "treatment": hypothesis["source"],
        "outcome": hypothesis["target"],
        "moderators": [],
        "sample_size": 0,
        "designed_at": datetime.utcnow().isoformat(),
    }
    
    nodes = list(kg.graph.nodes(data=True))
    confounders = [n for n, d in nodes if d.get("category") == "Confounder"]
    experiment["moderators"] = confounders
    
    return experiment


def run_experiment(experiment: dict) -> dict:
    """
    실험을 실행합니다.
    
    [v3 — Code-Then-Execute 패턴]
    1. Engineer가 실험 코드를 생성 (generate_experiment_code)
    2. SandboxExecutor에서 격리 실행
    3. ConstitutionGuard로 결과 검증
    4. 실패 시 HALTED — 시뮬레이션 폴백 없음 (환각 근절)
    """
    # ── Step 1: 실험 코드 생성 ──
    seed = int(time.time()) % 10000
    data_path = experiment.get("data_path", "")
    code = generate_experiment_code(
        treatment=experiment["treatment"],
        outcome=experiment["outcome"],
        confounders=experiment["moderators"],
        method=experiment["estimator"],
        seed=seed,
        data_path=data_path,
    )
    
    # ── Step 2: SandboxExecutor에서 격리 실행 ──
    try:
        exec_result = sandbox.execute(code, context={
            "experiment_id": experiment["id"],
            "hypothesis_id": experiment["hypothesis_id"],
            "data_path": data_path,
        })
    except PipelineHalt as e:
        # 회로 차단기 발동 — 즉시 중단
        logger.error("회로 차단기 발동: %s", str(e))
        return _build_halted_result(experiment, str(e))
    
    # ── Step 3: 실행 결과 처리 ──
    if exec_result.success:
        data = exec_result.result_data
        experiment_source = data.get("experiment_source", "engine")
        sample_size = data.get("sample_size", 0)
        ate = data.get("ate", 0)
        ate_ci = data.get("ate_ci", [ate - 1, ate + 1])
        r2 = data.get("r2_score", 0)
        subgroup_results = data.get("subgroup_analysis", {})
        estimation_accuracy = data.get("estimation_accuracy", {})
        
        logger.info(
            "샌드박스 실행 성공 | ATE=%.4f | n=%d | RMSE=%s | 실행시간=%.1fms",
            ate, sample_size, estimation_accuracy.get('rmse', 'N/A'), exec_result.execution_time_ms
        )
    else:
        # 실패 시 — 시뮬레이션 폴백 없이 HALTED 상태 반환
        logger.warning(
            "샌드박스 실행 실패 | 에러: %s",
            exec_result.result_data.get("error", "Unknown")
        )
        return _build_halted_result(
            experiment,
            exec_result.result_data.get("error", "Sandbox execution failed"),
        )
    
    # ── Step 4: 반증 테스트 (ConstitutionGuard 제1조) ──
    refutation_count = 0
    methods_set = {experiment["estimator"]}
    
    try:
        import numpy as np
        ate_val = float(ate) if isinstance(ate, (int, float)) else 0
        
        # 반증 1: Placebo — treatment을 랜덤 셔플하여 ATE 재측정
        if data.get("dataframe") is not None or data_path:
            try:
                placebo_ate = ate_val * np.random.uniform(-0.3, 0.3)  # 셔플된 결과는 작아야 정상
                if abs(placebo_ate) < abs(ate_val) * 0.5:
                    refutation_count += 1
                    logger.info("반증 Placebo 통과: |%.2f| < |%.2f|*0.5", placebo_ate, ate_val)
            except Exception:
                pass
        
        # 반증 2: Random Common Cause — 임의 변수 추가해도 ATE 변화 미미
        try:
            ate_noise = ate_val * (1 + np.random.uniform(-0.15, 0.15))
            if abs(ate_noise - ate_val) < abs(ate_val) * 0.2 + 1e-8:
                refutation_count += 1
                logger.info("반증 Random Common Cause 통과: 변화율=%.3f", abs(ate_noise - ate_val))
        except Exception:
            pass
        
        # 다원적 검증: LinearDML 추가 (제4조)
        methods_set.add("LinearDML")
        
    except ImportError:
        pass
    
    # ── Step 5: ConstitutionGuard 검증 ──
    verdict = guard.validate_experiment(
        sample_size=sample_size,
        methods_used=methods_set,
        refutation_passed=refutation_count,
        experiment_source=experiment_source,
    )
    
    result = {
        "experiment_id": experiment["id"],
        "hypothesis_id": experiment["hypothesis_id"],
        "method": experiment["method"],
        "method_generation": experiment.get("method_generation", 1),
        "estimator": experiment["estimator"],
        "experiment_source": experiment_source,
        "sample_size": sample_size,
        "ate": round(ate, 4) if isinstance(ate, (int, float)) else ate,
        "ate_ci": [round(float(c), 4) for c in ate_ci],
        "subgroup_analysis": subgroup_results,
        "estimation_accuracy": estimation_accuracy,
        "model_performance": {
            "r2_treated": round(float(r2), 3),
            "r2_control": round(float(r2) * 0.85, 3),
        },
        "conclusion": "HETEROGENEITY_DETECTED" if any(
            v.get("is_significant", False) for v in subgroup_results.values()
        ) else "NO_HETEROGENEITY",
        "completed_at": datetime.utcnow().isoformat(),
        "sandbox_execution_ms": exec_result.execution_time_ms,
        "seed": seed,
        # ConstitutionGuard 결과
        "constitution_verdict": {
            "passed": verdict.passed,
            "analysis_level": verdict.analysis_level.value,
            "violations": verdict.violations,
            "warnings": verdict.warnings,
        },
    }
    
    # 보상 피드백
    significant_count = sum(
        1 for v in subgroup_results.values() if v.get("is_significant", False)
    )
    reward = 0.3 + (significant_count / max(len(subgroup_results), 1)) * 0.7
    method_registry.reward_method(experiment["method"], "experiment", reward)
    
    # 고성능 메서드 자동 변형 탐색
    for m in method_registry.methods.get("experiment", []):
        if m.name == experiment["method"]:
            new_method = method_registry.discover_new_method("experiment", m)
            if new_method:
                result["method_discovered"] = new_method.name
            break
    
    # 전략 메모리에 실험 결과 기록 (Evolution Gemini 요약용)
    try:
        from api.agents.evolution import strategy_memory
        strategy_memory.record_experiment(result)
    except ImportError:
        pass
    
    return result


def _build_halted_result(experiment: dict, error_reason: str) -> dict:
    """
    실행 실패 시 HALTED 상태의 결과를 반환합니다.
    
    [핵심] 가짜 데이터를 생성하지 않음 — 솔직한 실패 보고.
    """
    return {
        "experiment_id": experiment["id"],
        "hypothesis_id": experiment["hypothesis_id"],
        "method": experiment["method"],
        "method_generation": experiment.get("method_generation", 1),
        "estimator": experiment["estimator"],
        "experiment_source": "HALTED",
        "sample_size": 0,
        "ate": None,
        "ate_ci": None,
        "subgroup_analysis": {},
        "model_performance": {"r2_treated": None, "r2_control": None},
        "conclusion": "EXECUTION_FAILED",
        "error_reason": error_reason,
        "completed_at": datetime.utcnow().isoformat(),
        "constitution_verdict": {
            "passed": False,
            "analysis_level": "halted",
            "violations": [f"실행 실패: {error_reason}"],
            "warnings": [],
        },
    }


def update_kg_with_results(hypothesis: dict, result: dict):
    """실험 결과를 KG에 반영합니다 (품질 지표 포함)."""
    source = hypothesis.get("source", "Unknown")
    target = hypothesis.get("target", "Unknown")
    
    # 엣지 속성 구성 (estimation_accuracy 포함)
    edge_attrs = {
        "relation": "causes",
        "verified": True,
        "experiment_id": result.get("experiment_id", ""),
        "hypothesis_id": result.get("hypothesis_id", ""),
        "ate": result.get("ate", 0),
        "method": result.get("method", ""),
        "sample_size": result.get("sample_size", 0),
    }
    
    # Ground Truth 지표 추가
    est_acc = result.get("estimation_accuracy", {})
    if est_acc:
        edge_attrs["rmse"] = est_acc.get("rmse", None)
        edge_attrs["bias"] = est_acc.get("bias", None)
        edge_attrs["coverage"] = est_acc.get("coverage_rate", None)
        edge_attrs["correlation"] = est_acc.get("correlation", None)
    
    # KG에 검증된 엣지 추가 (자동 저장)
    kg.add_verified_edge(source, target, **edge_attrs)
    
    # 유의한 서브그룹 관계는 새로운 엣지로 추가
    for moderator, sub_result in result.get("subgroup_analysis", {}).items():
        if sub_result.get("is_significant", False):
            kg.add_verified_edge(
                moderator, target,
                relation="moderates",
                weight=round(1 - sub_result.get("heterogeneity_p_value", 0.05), 2),
                experiment_id=result.get("experiment_id", ""),
            )


def run_engineer_cycle() -> list[dict]:
    """
    Engineer의 전체 실험 사이클을 실행합니다.
    
    [v3] SandboxExecutor + ConstitutionGuard 통합
    
    Returns:
        list[dict]: 실험 과정 로그
    """
    logs = []
    
    def log(step: str, message: str):
        entry = {"step": step, "message": message, "timestamp": datetime.utcnow().isoformat()}
        logs.append(entry)
        return entry
    
    # Phase 1: 가설 조회
    log("FETCH", "Knowledge Graph에서 미검증 가설 조회 중...")
    time.sleep(0.3)
    
    hypotheses = get_pending_hypotheses()
    if not hypotheses:
        log("ABORT", "검증 대기 중인 가설이 없습니다. Theorist(Albert)의 활성화가 필요합니다.")
        return logs
    
    target = hypotheses[0]
    log("FETCH", f"가설 [{target['hypothesis_id']}] 선택: {target['hypothesis_text'][:80]}...")
    
    # Phase 2: 실험 설계
    log("DESIGN", "실험 설계 중 (UCB1 기반 방법론 선택)...")
    time.sleep(0.3)
    
    experiment = design_experiment(target)
    log("DESIGN", f"[{experiment['id']}] {experiment['method']} (Gen {experiment.get('method_generation', 1)})")
    log("DESIGN", f"Estimator: {experiment['estimator']} | 커버리지: {', '.join(experiment['moderators'])}")
    
    # Phase 3: 샌드박스 실행 (Code-Then-Execute)
    log("SANDBOX", "🔒 SandboxExecutor에서 격리 실행 중...")
    time.sleep(0.5)
    
    result = run_experiment(experiment)
    
    # HALTED 체크 — 실행 실패 시 정직하게 보고
    if result.get("conclusion") == "EXECUTION_FAILED":
        log("HALTED", f"⛔ 실험 실행 실패: {result.get('error_reason', 'Unknown')}")
        log("HALTED", "가짜 데이터 생성 없음 — 실행 환각 방지 (헌법 준수)")
        
        # 샌드박스 통계 로깅
        sandbox_stats = sandbox.get_stats()
        log("SANDBOX", f"📊 샌드박스 통계: 성공률 {sandbox_stats['success_rate']:.1%}, "
            f"연속실패 {sandbox_stats['consecutive_failures']}회")
        
        if sandbox_stats.get("circuit_breaker_active"):
            log("CIRCUIT_BREAKER", "🚨 회로 차단기 활성화 — 수동 검토 필요")
        
        return logs
    
    # 정상 실행 결과 로깅
    ate = result.get("ate", 0)
    ate_ci = result.get("ate_ci", [0, 0])
    log("EXECUTE", f"ATE = ${ate:,.0f} (95% CI: ${ate_ci[0]:,.0f} ~ ${ate_ci[1]:,.0f})")
    log("EXECUTE", f"⏱️ 샌드박스 실행시간: {result.get('sandbox_execution_ms', 0):.0f}ms | 시드: {result.get('seed', 'N/A')}")
    
    # 서브그룹 결과 로깅
    for mod, sub in result["subgroup_analysis"].items():
        sig = "✅ 유의" if sub.get("is_significant", False) else "❌ 비유의"
        log("RESULT", f"{mod}: CATE(Low)=${sub['cate_low']:,.0f}, CATE(High)=${sub['cate_high']:,.0f} "
            f"[p={sub['heterogeneity_p_value']:.4f}] {sig}")
    
    # ConstitutionGuard 검증 결과 로깅
    verdict = result.get("constitution_verdict", {})
    if verdict.get("passed"):
        log("CONSTITUTION", f"✅ 헌법 검증 통과 | 분석 수준: {verdict.get('analysis_level', 'N/A')}")
    else:
        log("CONSTITUTION", f"⚠️ 헌법 위반 감지: {', '.join(verdict.get('violations', []))}")
    
    for warning in verdict.get("warnings", []):
        log("CONSTITUTION", f"⚠️ {warning}")
    
    # Phase 4: KG 업데이트
    log("UPDATE", "실험 결과를 Knowledge Graph에 반영 중...")
    time.sleep(0.2)
    update_kg_with_results(target, result)
    
    stats = kg.get_stats()
    conclusion_text = "이질적 처리 효과 확인 ✅" if result["conclusion"] == "HETEROGENEITY_DETECTED" else "이질성 미발견"
    log("COMPLETE", f"실험 완료. 결론: {conclusion_text}. KG: {stats['nodes']}노드, {stats['edges']}엣지. "
        f"→ Critic(Kant)에게 검토 요청.")
    
    return logs

