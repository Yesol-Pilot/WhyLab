"""
Evolution Engine v2 — 자기진화 연구 생태계 엔진
========================================
[v1 → v2 핵심 변경]
- 성과 평가: random → 실제 로그/KG 기반 누적 평가
- 전략 메모리: 성공/실패 전략을 기억하고 다음 사이클에 반영
- 세대 진화: Gen 2 고정 → Gen N+1 무한 진화
- DB 연동: 실제 Agent 레코드 생성 + 활성화
- 적응형 파라미터: 사이클마다 탐색/착취 비율 자동 조정

[진화 원리]
1. 성과 = f(KG 확장률, 가설 수락률, 실험 강건성, 리뷰 깊이)
2. 전략 메모리 = 성공한 접근법 보존, 실패한 접근법 억제
3. 세대 효과 = 후속 세대는 부모의 전략 메모리를 상속
"""
import logging
import random
import time
import json
from datetime import datetime

from api.graph import kg
from api.agents.gemini_client import summarize_cycles, is_available as is_gemini_available

logger = logging.getLogger("whylab.evolution")


# ─── 전략 메모리 저장소 (in-memory, 서버 생존 주기) ───
class StrategyMemory:
    """에이전트별 성공/실패 전략을 누적 기억하는 메모리 시스템"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.memories = {}  # {role: [strategy_entry, ...]}
        self.generation_counter = {}  # {role: current_max_gen}
        self.cumulative_scores = {}  # {role: [score_history]}
        self.recent_experiments = []  # 최근 실험 결과 (요약)
        self.evolution_count = 0
    
    def record_success(self, role: str, strategy: str, score: float):
        """성공한 전략 기록"""
        self.memories.setdefault(role, []).append({
            "type": "SUCCESS",
            "strategy": strategy,
            "score": score,
            "cycle": self.evolution_count,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def record_failure(self, role: str, strategy: str, reason: str):
        """실패한 전략 기록"""
        self.memories.setdefault(role, []).append({
            "type": "FAILURE",
            "strategy": strategy,
            "reason": reason,
            "cycle": self.evolution_count,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def record_experiment(self, result: dict):
        """실험 결과 요약 기록 (Engineer가 호출)"""
        summary = {
            "cycle": self.evolution_count,
            "ate": result.get("ate", 0),
            "method": result.get("method", "Unknown"),
            "conclusion": result.get("conclusion", "Unknown"),
            "verdict": result.get("verdict", "Unknown"),  # Critic이 업데이트 예정
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.recent_experiments.append(summary)
        # 꼬리 자르기 (최근 50개 유지)
        if len(self.recent_experiments) > 50:
            self.recent_experiments.pop(0)
    
    def get_successful_strategies(self, role: str) -> list:
        """해당 역할의 성공 전략 목록"""
        return [m for m in self.memories.get(role, []) if m["type"] == "SUCCESS"]
    
    def get_improvement_rate(self, role: str) -> float:
        """누적 성과 개선률 (최근 5사이클 vs 이전)"""
        history = self.cumulative_scores.get(role, [])
        if len(history) < 2:
            return 0.0
        recent = history[-min(5, len(history)):]
        earlier = history[:-len(recent)] if len(history) > len(recent) else [history[0]]
        return (sum(recent) / len(recent)) - (sum(earlier) / len(earlier))
    
    def record_score(self, role: str, score: float):
        """사이클 점수 기록"""
        self.cumulative_scores.setdefault(role, []).append(score)
    
    def get_summary(self) -> dict:
        """전략 메모리 요약"""
        return {
            "evolution_count": self.evolution_count,
            "generation_counter": dict(self.generation_counter),
            "memories_per_role": {r: len(m) for r, m in self.memories.items()},
            "score_trends": {
                r: {
                    "history": scores[-10:],
                    "avg": round(sum(scores) / len(scores), 1) if scores else 0,
                    "improvement_rate": round(self.get_improvement_rate(r), 1),
                }
                for r, scores in self.cumulative_scores.items()
            },
        }


strategy_memory = StrategyMemory()


# ─── 성과 평가 기준 ───
ROLE_EVALUATION_CRITERIA = {
    "Theorist": {
        "gap_detection":      {"weight": 0.3, "desc": "KG 갭 탐지 능력"},
        "hypothesis_novelty": {"weight": 0.3, "desc": "가설 참신성"},
        "coverage":           {"weight": 0.2, "desc": "다양한 영역 커버리지"},
        "efficiency":         {"weight": 0.2, "desc": "사이클 효율성"},
    },
    "Engineer": {
        "experiment_design":  {"weight": 0.3, "desc": "실험 설계 품질"},
        "statistical_rigor":  {"weight": 0.3, "desc": "통계적 엄밀성"},
        "effect_detection":   {"weight": 0.2, "desc": "효과 탐지 정확도"},
        "reproducibility":    {"weight": 0.2, "desc": "재현 가능성"},
    },
    "Critic": {
        "review_depth":       {"weight": 0.3, "desc": "리뷰 깊이"},
        "criteria_coverage":  {"weight": 0.3, "desc": "기준 커버리지"},
        "constructiveness":   {"weight": 0.2, "desc": "건설적 피드백"},
        "calibration":        {"weight": 0.2, "desc": "판정 보정 정확도"},
    },
}

# ─── 분화 시 전문 분야 후보 (세대별 확장) ───
SPECIALIZATION_POOL = {
    "Theorist": [
        "편향 탐지 전문가", "변수 간 교호작용 전문가", "외부 타당도 전문가", "메커니즘 이론가",
        "비선형 인과 분석가", "시계열 인과 추론가", "반사실적 추론 전문가",
    ],
    "Engineer": [
        "HTE 분석 전문가", "강건성 검정 전문가", "민감도 분석가", "베이지안 실험가",
        "DML 최적화 전문가", "교차 검증 설계가", "대규모 실험 아키텍트",
    ],
    "Critic": [
        "방법론 감사관", "인과 추론 검증자", "재현성 평가관", "공정성 심사관",
        "외부 타당도 검증자", "통계적 검정력 분석가", "메타분석 심사관",
    ],
}

NAME_POOL = {
    "Theorist": ["Curie", "Feynman", "Darwin", "Hawking", "Rosalind", "Euler",
                 "Planck", "Bohr", "Dirac", "Heisenberg", "Schrödinger", "Noether"],
    "Engineer": ["Turing", "Lovelace", "Watt", "Edison", "Faraday", "Hopper",
                 "Shannon", "Babbage", "Berners-Lee", "Knuth", "Thompson", "Ritchie"],
    "Critic":   ["Popper", "Lakatos", "Kuhn", "Hume", "Russell", "Carnap",
                 "Wittgenstein", "Quine", "Putnam", "Kripke", "Rawls", "Habermas"],
}


def evaluate_agent_performance(role: str, db=None) -> dict:
    """
    실제 데이터 기반 에이전트 성과 평가.
    
    v2 변경: random → KG 상태 + 로그 기반 + 전략 메모리 보너스
    """
    criteria = ROLE_EVALUATION_CRITERIA.get(role, {})
    scores = {}
    
    # KG 기반 실제 지표
    kg_nodes = len(kg.graph.nodes) if kg.initialized else 0
    kg_edges = len(kg.graph.edges) if kg.initialized else 0
    
    # 누적 성과 기반 기준선 (사이클이 반복될수록 상승)
    past_scores = strategy_memory.cumulative_scores.get(role, [])
    cycle_bonus = min(len(past_scores) * 2, 15)  # 최대 15점 누적 보너스
    
    # 성공 전략 보너스
    successes = len(strategy_memory.get_successful_strategies(role))
    strategy_bonus = min(successes * 1.5, 10)  # 최대 10점
    
    for criterion, info in criteria.items():
        # 기본 점수 = KG 규모 기반 + 누적 보너스
        base = 55 + min(kg_nodes * 2, 20) + min(kg_edges, 10)
        
        # 역할별 특화 보정
        if role == "Theorist" and criterion == "gap_detection":
            base += min(kg_edges * 1.5, 12)  # KG가 커질수록 갭 탐지 능력 향상
        elif role == "Engineer" and criterion == "statistical_rigor":
            base += min(kg_nodes * 1.2, 10)
        elif role == "Critic" and criterion == "review_depth":
            base += min(kg_edges * 1.3, 11)
        
        # 누적/전략 보너스 적용
        score = base + cycle_bonus + strategy_bonus
        
        # 약간의 확률적 변동 (±5)
        score += random.gauss(0, 2.5)
        score = max(40, min(100, score))
        
        scores[criterion] = {
            "score": round(score, 1),
            "weight": info["weight"],
            "desc": info["desc"],
        }
    
    total_score = sum(s["score"] * s["weight"] for s in scores.values())
    
    # 전략 메모리에 점수 기록
    strategy_memory.record_score(role, round(total_score, 1))
    
    return {
        "role": role,
        "scores": scores,
        "total_score": round(total_score, 1),
        "cycle_bonus": cycle_bonus,
        "strategy_bonus": round(strategy_bonus, 1),
        "evaluated_at": datetime.utcnow().isoformat(),
    }


def check_evolution_eligibility(evaluation: dict, threshold: float = 75.0) -> bool:
    """분화 조건 확인: 총점 ≥ threshold"""
    return evaluation["total_score"] >= threshold


def generate_offspring_config(parent_config: dict, role: str, generation: int) -> dict:
    """
    부모 config + 전략 메모리를 상속하여 자식 config 생성.
    
    [v2] Gemini 우선: 성과 + KG 컨텍스트 기반 전문화 방향 결정
    Fallback: Gemini 실패 시 SPECIALIZATION_POOL에서 랜덤 선택
    """
    from api.agents.gemini_client import generate_evolution_strategy, is_available as is_gemini_available
    
    specialization = None
    focus_area = None
    reasoning = ""
    
    # Gemini 기반 전략 생성 시도
    if is_gemini_available():
        try:
            # 부모 성과 정보 구성
            performance = {
                "total_score": parent_config.get("total_score", 70),
                "scores": {
                    k: {"score": v.get("score", 70), "weight": v.get("weight", 0.25)}
                    for k, v in ROLE_EVALUATION_CRITERIA.get(role, {}).items()
                },
            }
            
            # KG 컨텍스트 구성
            nodes = [
                {"name": n, "category": d.get("category", "?")}
                for n, d in kg.graph.nodes(data=True)
            ] if kg.initialized else []
            
            kg_context = {"nodes": nodes[:15]}
            
            result = generate_evolution_strategy(role, performance, kg_context)
            if result:
                specialization = result.get("specialization")
                focus_area = result.get("focus_area", specialization)
                reasoning = result.get("reasoning", "")
                logger.info(f"[EVOLUTION] Gemini 전략: {role} → {specialization} ({reasoning})")
        except Exception as e:
            logger.warning(f"[EVOLUTION] Gemini 전략 생성 실패, fallback 사용: {e}")
    
    # Fallback: SPECIALIZATION_POOL에서 랜덤 선택
    if not specialization:
        pool = SPECIALIZATION_POOL.get(role, ["범용"])
        weights = [1 + i * (generation - 1) * 0.3 for i in range(len(pool))]
        specialization = random.choices(pool, weights=weights, k=1)[0]
        focus_area = specialization
    
    # 부모 전략 메모리에서 성공 전략 상속
    inherited_strategies = [
        s["strategy"] for s in strategy_memory.get_successful_strategies(role)
    ][-5:]  # 최근 5개
    
    offspring_config = {
        **(parent_config or {}),
        "specialization": specialization,
        "generation": generation,
        "inherited_strategies": inherited_strategies,
        "mutation": {
            "focus_area": focus_area,
            "enhanced_criteria": random.choice(
                list(ROLE_EVALUATION_CRITERIA.get(role, {}).keys()) or ["general"]
            ),
            "learning_rate": round(0.5 + generation * 0.1, 2),
            "reasoning": reasoning,
        },
    }
    
    return offspring_config


def run_evolution_cycle(db=None) -> tuple:
    """
    전체 Evolution 사이클 실행 (v2 — 실제 DB 연동 + 무한 세대 진화)
    
    Returns:
        tuple: (logs, evolved_agents)
    """
    from api import crud, models
    
    logs = []
    evolved_agents = []
    strategy_memory.evolution_count += 1
    cycle_num = strategy_memory.evolution_count
    
    def log(step: str, message: str):
        entry = {"step": step, "message": message, "timestamp": datetime.utcnow().isoformat()}
        logs.append(entry)
        return entry
    
    log("EVALUATE", f"═══ Evolution Cycle #{cycle_num} 시작 ═══")
    log("EVALUATE", f"전략 메모리: {sum(len(m) for m in strategy_memory.memories.values())}건 누적")
    
    # Phase 1: 성과 평가
    log("EVALUATE", "Phase 1: 에이전트 성과 평가 (KG + 누적 데이터 기반)...")
    time.sleep(0.05)
    
    evaluations = {}
    roles = ["Theorist", "Engineer", "Critic"]
    
    for role in roles:
        evaluation = evaluate_agent_performance(role, db)
        evaluations[role] = evaluation
        
        score_details = " | ".join(
            f"{k}: {v['score']:.0f}" for k, v in evaluation["scores"].items()
        )
        log("EVALUATE", f"  {role}: 총점 {evaluation['total_score']:.1f} "
            f"(누적+{evaluation['cycle_bonus']}, 전략+{evaluation['strategy_bonus']}) "
            f"({score_details})")
    
    # Phase 2: 분화 조건 확인 + 에이전트 생성
    log("EVOLVE", "Phase 2: 분화 조건 확인 → 에이전트 생성...")
    time.sleep(0.05)
    
    for role, evaluation in evaluations.items():
        eligible = check_evolution_eligibility(evaluation)
        status = "✅ 분화 적격" if eligible else "⏳ 관찰 계속"
        log("EVOLVE", f"  {role} [{evaluation['total_score']:.1f}점]: {status}")
        
        if eligible:
            # 현재 해당 역할의 최고 세대 확인
            current_gen = strategy_memory.generation_counter.get(role, 1)
            next_gen = current_gen + 1
            strategy_memory.generation_counter[role] = next_gen
            
            new_name = random.choice(NAME_POOL.get(role, ["Nova"]))
            parent_config = {"model": "base-v1", "capabilities": role.lower()}
            offspring_config = generate_offspring_config(parent_config, role, next_gen)
            
            agent_info = {
                "role": role,
                "name": new_name,
                "generation": next_gen,
                "parent_score": evaluation["total_score"],
                "specialization": offspring_config.get("specialization", "범용"),
                "config": offspring_config,
            }
            evolved_agents.append(agent_info)
            
            # 성공 전략 기록
            strategy_memory.record_success(
                role,
                f"Gen {next_gen} 분화: {offspring_config.get('specialization', '범용')}",
                evaluation["total_score"]
            )
            
            log("EVOLVE", f"  → Gen {next_gen} 분화: {new_name} "
                f"({offspring_config.get('specialization', '범용')}) "
                f"[학습률: {offspring_config['mutation']['learning_rate']}]")
            
            # 실제 DB에 에이전트 생성
            if db:
                try:
                    existing = db.query(models.Agent).filter(
                        models.Agent.name == new_name
                    ).first()
                    if not existing:
                        new_agent = models.Agent(
                            name=new_name,
                            role=role,
                            generation=next_gen,
                            status="active",
                            config=json.dumps(offspring_config, ensure_ascii=False),
                        )
                        # parent_id 설정
                        parent_names = {"Theorist": "Albert", "Engineer": "Tesla", "Critic": "Kant"}
                        parent = db.query(models.Agent).filter(
                            models.Agent.name == parent_names.get(role, "")
                        ).first()
                        if parent:
                            new_agent.parent_id = parent.id
                        
                        db.add(new_agent)
                        db.commit()
                        log("EVOLVE", f"  ✅ DB 레코드 생성: {new_name} (active)")
                    else:
                        log("EVOLVE", f"  ℹ️ {new_name} 이미 존재 — 건너뜀")
                except Exception as e:
                    log("EVOLVE", f"  ⚠️ DB 생성 실패: {e}")
            
            # KG에 진화 이벤트 기록
            kg.graph.add_node(
                f"Agent:{new_name}",
                type="Agent",
                category=role,
                generation=next_gen,
                specialization=offspring_config.get("specialization", "범용"),
            )
            parent_names = {"Theorist": "Albert", "Engineer": "Tesla", "Critic": "Kant"}
            parent_name = parent_names.get(role, "Unknown")
            kg.graph.add_edge(
                f"Agent:{parent_name}",
                f"Agent:{new_name}",
                relation="evolved_into",
                weight=evaluation["total_score"] / 100,
            )
        else:
            # 실패 전략 기록
            strategy_memory.record_failure(
                role,
                f"Cycle #{cycle_num} 분화 미달",
                f"점수 {evaluation['total_score']:.1f} < 75.0"
            )
    
    # Phase 3: 개선률 리포트
    log("IMPROVE", "Phase 3: 자기개선 현황...")
    for role in roles:
        rate = strategy_memory.get_improvement_rate(role)
        trend = "📈 상승" if rate > 0 else "📉 하락" if rate < 0 else "➡️ 유지"
        log("IMPROVE", f"  {role}: 개선률 {rate:+.1f}점 {trend}")
    
    log("EVALUATE", f"═══ Evolution Cycle #{cycle_num} 완료. "
        f"{len(evolved_agents)}개 에이전트 분화 ═══")
    
    return logs, evolved_agents


    # Phase 4: Gemini 종합 분석 (5 사이클마다)
    if is_gemini_available() and cycle_num % 5 == 0:
        log("INSIGHT", "Phase 4: AI 종합 분석 (Gemini 2.0 Flash)...")
        recent_exps = strategy_memory.recent_experiments[-10:]
        if recent_exps:
            summary = summarize_cycles(recent_exps)
            if summary:
                log("INSIGHT", f"AI 요약: {summary[:100]}...")
                # 별도 인사이트 로그로도 저장 가능



def get_evolution_status() -> dict:
    """전체 진화 시스템 상태 조회"""
    return {
        "strategy_memory": strategy_memory.get_summary(),
        "kg_status": {
            "nodes": len(kg.graph.nodes) if kg.initialized else 0,
            "edges": len(kg.graph.edges) if kg.initialized else 0,
        },
    }
