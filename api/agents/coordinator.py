"""
Coordinator Agent v2 — E2E 연구 순환 오케스트레이션 (Sprint 36)
================================================================
STEAM → Theorist → Engineer → Sandbox → Critic → KG 반영까지
실제 데이터 기반 완전 순환을 구현합니다.

[설계 문서 §3.1 계층적 오케스트레이터-워커 패턴]
- Coordinator는 유일한 전역 오케스트레이터
- 모든 에이전트 간 메시지 패싱은 Coordinator를 경유

[7단계 파이프라인]
1. select_agenda    — Director에서 Grand Challenge 선택
2. supply_data      — STEAM Generator로 합성 데이터 생성
3. generate_hypo    — Theorist 호출 (KG gap 기반)
4. run_experiment   — Engineer에 data_path 전달, Sandbox 실행
5. review_result    — Critic 판정 + 재실행 루프 (최대 2회)
6. update_knowledge — KG에 결과 반영
7. log_cycle        — DB에 사이클 로그 기록
"""
import os
import time
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("whylab.coordinator")

# ── 설정 상수 ──
MAX_RETRY_ON_REVISE = 2          # Critic REVISE 시 재시도 횟수
CONSECUTIVE_TOPIC_LIMIT = 3      # 헌법 제13조: 같은 주제 연속 선택 제한
DEFAULT_SAMPLE_SIZE = 3000
DEFAULT_SEED = 42


class CoordinatorV2:
    """
    v2 Coordinator — 7단계 E2E 파이프라인.
    
    기존 run_coordinator_cycle() 함수를 대체합니다.
    """
    
    def __init__(self):
        self._recent_topics: list[str] = []
        self._cycle_count = 0
        self._last_cycle_at: Optional[str] = None
    
    def run_cycle(self) -> dict:
        """
        전체 연구 사이클 1회를 실행합니다.
        
        Returns:
            dict: 사이클 결과 (stages, hypothesis, experiment, verdict, metrics)
        """
        self._cycle_count += 1
        cycle_id = f"CYCLE-{self._cycle_count:04d}"
        start_time = time.time()
        
        result = {
            "cycle_id": cycle_id,
            "started_at": datetime.utcnow().isoformat(),
            "stages": [],
            "status": "IN_PROGRESS",
        }
        
        try:
            # ── Step 1: 아젠다 선택 ──
            challenge = self._select_agenda(result)
            
            # ── Step 2: STEAM 데이터 공급 ──
            data_info = self._supply_data(result, challenge)
            
            # ── Step 3: 가설 생성 (STEAM 데이터 컨텍스트 포함) ──
            hypothesis = self._generate_hypothesis(result, data_info)
            
            # ── Step 4+5: 실험 + 판정 (재시도 루프) ──
            experiment_result, verdict = self._experiment_review_loop(
                result, hypothesis, data_info
            )
            
            # ── Step 6: KG 업데이트 ──
            self._update_knowledge(result, hypothesis, experiment_result, verdict)
            
            # ── Step 7: 결과 종합 ──
            elapsed = time.time() - start_time
            result["status"] = "COMPLETE"
            result["elapsed_seconds"] = round(elapsed, 2)
            result["hypothesis"] = hypothesis
            result["experiment"] = experiment_result
            result["verdict"] = verdict
            result["metrics"] = {
                "cycle_id": cycle_id,
                "challenge": challenge.get("id", "unknown"),
                "data_quality": data_info.get("quality_grade", "N/A"),
                "ate": experiment_result.get("ate", None),
                "verdict_action": verdict.get("action", "UNKNOWN"),
                "elapsed_seconds": round(elapsed, 2),
            }
            
        except Exception as e:
            logger.error("연구 사이클 실패: %s", str(e))
            result["status"] = "ERROR"
            result["error"] = str(e)
        
        self._last_cycle_at = datetime.utcnow().isoformat()
        result["ended_at"] = self._last_cycle_at
        
        self._log_stage(result, "COMPLETE", f"사이클 {cycle_id} 종료 — {result['status']}")
        
        return result
    
    # ═══════════════════════════════════════════════
    # Step 1: 아젠다 선택 (헌법 제13조 적용)
    # ═══════════════════════════════════════════════
    def _select_agenda(self, result: dict) -> dict:
        """Director에서 Grand Challenge를 선택합니다."""
        self._log_stage(result, "AGENDA", "Grand Challenge 선택 중...")
        
        from engine.agents.director import LabDirector
        director = LabDirector()
        challenges = director.challenges  # Director는 challenges 리스트 필드 사용
        
        if not challenges:
            raise RuntimeError("Grand Challenges DB가 비어있습니다.")
        
        # 헌법 제13조: 같은 주제 3회 연속 금지
        available = [
            c for c in challenges
            if c.get("id", "") not in self._recent_topics[-CONSECUTIVE_TOPIC_LIMIT:]
        ] or challenges  # 모두 제외되면 전체 풀에서 선택
        
        # 영향도(Impact) × 난이도(Difficulty) 기반 선택
        impact_map = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        selected = max(available, key=lambda c: impact_map.get(c.get("impact", "Medium"), 2))
        
        self._recent_topics.append(selected.get("id", "unknown"))
        self._log_stage(
            result, "AGENDA",
            f"선택: [{selected.get('id', '?')}] {selected.get('title', 'Untitled')} (영향도: {selected.get('impact', 'N/A')})"
        )
        
        return selected
    
    # ═══════════════════════════════════════════════
    # Step 2: STEAM 합성 데이터 공급
    # ═══════════════════════════════════════════════
    def _supply_data(self, result: dict, challenge: dict) -> dict:
        """STEAM Generator로 합성 데이터를 생성합니다."""
        self._log_stage(result, "DATA", "STEAM 합성 데이터 생성 시작...")
        
        from engine.data.steam_generator import steam  # 모듈 레벨 싱글턴
        
        # Challenge 카테고리와 DGP 매핑
        dgp_list = steam.available_dgps  # @property → list[str]
        if not dgp_list:
            raise RuntimeError("STEAM DGP 템플릿이 없습니다.")
        
        # 카테고리 기반 매칭 시도 → 없으면 첫 번째 DGP 사용
        category = challenge.get("category", "").lower()
        matched_dgp = None
        if category:
            for dname in dgp_list:
                template = steam._templates.get(dname)
                if template and category in template.category.lower():
                    matched_dgp = dname
                    break
        
        dgp_name = matched_dgp or dgp_list[0]
        
        # 데이터 생성 → SyntheticData 객체 반환
        syn_data = steam.generate(
            dgp_name=dgp_name,
            n=DEFAULT_SAMPLE_SIZE,
            seed=DEFAULT_SEED,
        )
        
        # 품질 평가
        quality = steam.evaluate_quality(syn_data)
        quality_grade = quality.get("grade", "N/A")
        ate_true = syn_data.ate_true
        
        # CSV로 저장
        data_path = self._save_data_csv(syn_data, dgp_name)
        
        # STEAM 데이터의 실제 컬럼명 추출 (가설 변수명과 매핑용)
        all_cols = list(syn_data.df.columns)
        confounders = [c for c in all_cols 
                       if c not in (syn_data.treatment_col, syn_data.outcome_col)]
        
        data_info = {
            "dgp_name": dgp_name,
            "sample_size": syn_data.n,
            "quality_grade": quality_grade,
            "ate_true": ate_true,
            "data_path": data_path,
            "treatment": syn_data.treatment_col,
            "outcome": syn_data.outcome_col,
            "confounders": confounders[:6],  # 상위 6개로 제한 (과적합 방지)
        }
        
        self._log_stage(
            result, "DATA",
            f"STEAM 생성 완료: {dgp_name} (n={syn_data.n}, Grade={quality_grade}, ATE={ate_true:.3f})"
        )
        
        return data_info
    
    def _save_data_csv(self, syn_data, dgp_name: str) -> str:
        """SyntheticData의 DataFrame을 CSV로 저장합니다."""
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")
        os.makedirs(base_dir, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"steam_{dgp_name}_{timestamp}.csv"
        filepath = os.path.join(base_dir, filename)
        
        # SyntheticData.df (pandas DataFrame) → CSV
        syn_data.df.to_csv(filepath, index=False, encoding="utf-8")
        
        logger.info("STEAM 데이터 저장: %s (%d행)", filepath, len(syn_data.df))
        return filepath
    
    # ═══════════════════════════════════════════════
    # Step 3: 가설 생성
    # ═══════════════════════════════════════════════
    def _generate_hypothesis(self, result: dict, data_info: dict = None) -> dict:
        """Theorist를 호출하여 가설을 생성하고, STEAM 변수를 강제 매핑합니다."""
        self._log_stage(result, "HYPOTHESIS", "Theorist(Albert) 가설 생성 중...")
        
        from api.agents.theorist import generate_hypothesis
        hypothesis = generate_hypothesis()
        
        # STEAM 데이터의 treatment/outcome을 가설에 **강제 매핑**
        # Theorist가 KG seed 변수명을 사용하므로, STEAM 변수명으로 덮어쓰기 필수
        if data_info:
            hypothesis["source"] = data_info.get("treatment", hypothesis.get("iv", "Treatment"))
            hypothesis["target"] = data_info.get("outcome", hypothesis.get("dv", "Outcome"))
            hypothesis["iv"] = hypothesis["source"]
            hypothesis["dv"] = hypothesis["target"]
        
        self._log_stage(
            result, "HYPOTHESIS",
            f"[{hypothesis.get('id','?')}] {hypothesis.get('text','')[:80]}... "
            f"(source: {hypothesis.get('hypothesis_source', 'N/A')}, "
            f"IV={hypothesis.get('source','?')}, DV={hypothesis.get('target','?')})"
        )
        
        return hypothesis
    
    # ═══════════════════════════════════════════════
    # Step 4+5: 실험 + 판정 루프
    # ═══════════════════════════════════════════════
    def _experiment_review_loop(
        self, result: dict, hypothesis: dict, data_info: dict
    ) -> tuple[dict, dict]:
        """
        Engineer 실험 → Critic 판정 → 필요시 재실행 (최대 MAX_RETRY_ON_REVISE회).
        """
        from api.agents.engineer import design_experiment, run_experiment
        from api.agents.critic import review_experiment
        
        experiment_result = {}
        verdict = {"action": "REJECT", "reason": "실험이 수행되지 않았습니다."}
        
        # STEAM 데이터에서 moderators(공변량) 자동 추출
        data_path = data_info.get("data_path", "")
        steam_moderators = []
        if data_path:
            try:
                import pandas as pd
                df_cols = list(pd.read_csv(data_path, nrows=0).columns)
                treatment_col = data_info.get("treatment", "")
                outcome_col = hypothesis.get("target", "")
                # treatment/outcome 제외한 수치형 컬럼 = 공변량
                steam_moderators = [
                    c for c in df_cols
                    if c not in (treatment_col, outcome_col, "true_cate")
                ]
            except Exception:
                pass
        
        for attempt in range(1, MAX_RETRY_ON_REVISE + 2):
            # ── Step 4: 실험 ──
            self._log_stage(
                result, "EXPERIMENT",
                f"Engineer(Tesla) 실험 #{attempt} 실행 중... (data: {data_info['dgp_name']})"
            )
            
            experiment = design_experiment(hypothesis)
            # STEAM 데이터 경로 + 공변량 주입          # STEAM 데이터의 실제 컬럼명으로 오버라이드
            # (KG 가설 변수명 "Job Training Program" ≠ CSV 컬럼명 "alignment_training")
            experiment["data_path"] = data_path
            experiment["data_sample_size"] = data_info.get("sample_size", DEFAULT_SAMPLE_SIZE)
            if steam_moderators:
                experiment["moderators"] = steam_moderators
            
            if data_info.get("treatment"):
                experiment["treatment"] = data_info["treatment"]
            if data_info.get("outcome"):
                experiment["outcome"] = data_info["outcome"]
            if data_info.get("confounders"):
                experiment["moderators"] = data_info["confounders"]
            
            experiment_result = run_experiment(experiment)
            
            is_halted = experiment_result.get("experiment_source") == "HALTED"
            ate = experiment_result.get("ate", "N/A")
            
            self._log_stage(
                result, "EXPERIMENT",
                f"실험 #{attempt} 완료 — ATE={ate}, Source={'HALTED' if is_halted else 'engine'}"
            )
            
            # ── Step 5: 판정 ──
            self._log_stage(result, "REVIEW", f"Critic(Kant) 판정 #{attempt}...")
            verdict = review_experiment(experiment_result)
            action = verdict.get("verdict", verdict.get("action", "REJECT")) if isinstance(verdict, dict) else "REJECT"
            
            self._log_stage(
                result, "REVIEW",
                f"Critic 판정: {action} (시도 {attempt}/{MAX_RETRY_ON_REVISE + 1})"
            )
            
            if action == "ACCEPT":
                logger.info("실험 ACCEPT — 사이클 성공")
                break
            elif action == "REVISE" and attempt <= MAX_RETRY_ON_REVISE:
                logger.warning("REVISE — 재시도 %d/%d", attempt, MAX_RETRY_ON_REVISE)
                continue
            else:
                logger.warning("REJECT 또는 재시도 한도 초과 — 사이클 종료")
                break
        
        return experiment_result, verdict
    
    # ═══════════════════════════════════════════════
    # Step 6: Knowledge Graph 업데이트
    # ═══════════════════════════════════════════════
    def _update_knowledge(
        self, result: dict, hypothesis: dict, experiment: dict, verdict: dict
    ):
        """실험 결과를 Knowledge Graph에 반영합니다."""
        self._log_stage(result, "KG_UPDATE", "Knowledge Graph 업데이트 중...")
        
        from api.graph import kg
        
        action = verdict.get("verdict", verdict.get("action", "REJECT")) if isinstance(verdict, dict) else "REJECT"
        
        ate = experiment.get("ate")
        method = experiment.get("method", "unknown")
        exp_id = experiment.get("experiment_id", "")
        source = hypothesis.get("source", hypothesis.get("iv", "Unknown"))
        target = hypothesis.get("target", hypothesis.get("dv", "Unknown"))
        
        if action == "ACCEPT":
            # 성공 → 검증된 엣지
            kg.add_verified_edge(
                source, target,
                relation="causes", verified=True, weight=0.8,
                ate=ate, method=method, experiment_id=exp_id,
                verdict="ACCEPT", confidence="high",
            )
            self._log_stage(result, "KG_UPDATE", f"✅ 검증 완료: {source} → {target} (ATE={ate})")
        elif action == "REVISE":
            # 조건부 → LOW_CONFIDENCE 엣지
            kg.add_verified_edge(
                source, target,
                relation="may_cause", verified=False, weight=0.4,
                ate=ate, method=method, experiment_id=exp_id,
                verdict="REVISE", confidence="low",
            )
            self._log_stage(result, "KG_UPDATE", f"⚠️ 조건부: {source} → {target} (ATE={ate})")
        else:
            # REJECT → 탐색적 결과로 기록 (지식 축적 보장)
            kg.add_verified_edge(
                source, target,
                relation="explored", verified=False, weight=0.1,
                ate=ate, method=method, experiment_id=exp_id,
                verdict="REJECT", confidence="exploratory",
            )
            self._log_stage(result, "KG_UPDATE", f"🔍 탐색: {source} → {target} (ATE={ate}, REJECT)")
    
    # ═══════════════════════════════════════════════
    # 유틸리티
    # ═══════════════════════════════════════════════
    def _log_stage(self, result: dict, stage: str, message: str):
        """결과 딕셔너리에 단계 로그를 추가합니다."""
        entry = {
            "stage": stage,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }
        result["stages"].append(entry)
        logger.info("[%s] %s", stage, message)
    
    def get_status(self) -> dict:
        """Coordinator 현재 상태를 반환합니다."""
        return {
            "version": "v2",
            "cycle_count": self._cycle_count,
            "last_cycle_at": self._last_cycle_at,
            "recent_topics": self._recent_topics[-5:],
        }


# ── 싱글턴 ──
coordinator_v2 = CoordinatorV2()


# ── 하위 호환: 기존 run_coordinator_cycle() 인터페이스 유지 ──
def run_coordinator_cycle() -> list[dict]:
    """
    기존 API 호환용 래퍼.
    Coordinator v2를 실행하고, 기존 로그 포맷으로 변환합니다.
    """
    result = coordinator_v2.run_cycle()
    
    # 기존 포맷 호환: stages → logs 리스트 변환
    logs = []
    for stage in result.get("stages", []):
        logs.append({
            "step": stage["stage"],
            "message": stage["message"],
            "timestamp": stage["timestamp"],
        })
    
    return logs
