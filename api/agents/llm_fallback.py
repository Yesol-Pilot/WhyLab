"""
LLM Fallback Chain — 3단계 계층적 폴백 전략 (Sprint 38)
=========================================================
Gemini API 실패 시 품질 저하를 최소화하는 폴백 체인.

[3단계 계층]
- Level 1 (L1): Gemini API → 최고 품질
- Level 2 (L2): KG 캐시 유사 응답 재활용 → 중간 품질  
- Level 3 (L3): 구조화 템플릿 → 최소 품질 보장

[적용 대상]
- Theorist: 가설 생성 (L1 + L2 + L3)
- Critic:   판정 생성 (L1 + L2 + L3)
- Debate:   토론 응답 (L1 + L3)
"""
import logging
import time
from typing import Optional, Callable
from datetime import datetime, timezone, timedelta

# 한국 표준시 (KST = UTC+9)
KST = timezone(timedelta(hours=9))

logger = logging.getLogger("whylab.llm_fallback")


class FallbackLevel:
    """폴백 레벨 라벨."""
    GEMINI = "gemini"           # 최상위: Gemini API
    KG_CACHE = "kg_cache"       # 중간: KG 유사 응답 캐시
    TEMPLATE = "template"       # 최하위: 규칙 기반 템플릿


class FallbackResult:
    """폴백 체인 결과."""
    
    def __init__(self, content: dict, source: str, level: int, latency_ms: float = 0):
        self.content = content            # 실제 응답 데이터
        self.source = source              # FallbackLevel 라벨
        self.level = level                # 1, 2, 3
        self.latency_ms = latency_ms      # 소요 시간
        self.generated_at = datetime.now(KST).isoformat()
    
    @property
    def quality_stars(self) -> str:
        """레벨에 따른 품질 등급."""
        return {1: "⭐⭐⭐", 2: "⭐⭐", 3: "⭐"}.get(self.level, "❓")
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "source": self.source,
            "level": self.level,
            "quality": self.quality_stars,
            "latency_ms": round(self.latency_ms, 1),
            "generated_at": self.generated_at,
        }


class LLMFallbackChain:
    """
    3단계 폴백 체인.
    
    각 레벨은 (시도 함수, 레벨 번호, 소스 라벨) 튜플로 등록됩니다.
    순서대로 시도하며, 첫 번째 성공한 결과를 반환합니다.
    
    사용 예:
        chain = LLMFallbackChain("hypothesis")
        chain.add_level(gemini_generate, 1, FallbackLevel.GEMINI)
        chain.add_level(kg_cache_search, 2, FallbackLevel.KG_CACHE) 
        chain.add_level(template_generate, 3, FallbackLevel.TEMPLATE)
        result = chain.execute(context)
    """
    
    def __init__(self, task_type: str):
        self.task_type = task_type
        self._levels: list[tuple[Callable, int, str]] = []
        self._stats = {
            "total_calls": 0,
            "level_hits": {1: 0, 2: 0, 3: 0},
            "failures": 0,
        }
    
    def add_level(self, fn: Callable, level: int, source: str) -> "LLMFallbackChain":
        """폴백 레벨을 등록합니다."""
        self._levels.append((fn, level, source))
        return self
    
    def execute(self, context: dict) -> FallbackResult:
        """
        폴백 체인을 실행합니다.
        
        Args:
            context: 폴백 함수에 전달할 컨텍스트
            
        Returns:
            FallbackResult: 첫 번째 성공한 레벨의 결과
        """
        self._stats["total_calls"] += 1
        
        for fn, level, source in self._levels:
            start = time.time()
            try:
                result = fn(context)
                latency = (time.time() - start) * 1000
                
                if result and isinstance(result, dict) and result.get("text"):
                    self._stats["level_hits"][level] = self._stats["level_hits"].get(level, 0) + 1
                    
                    logger.info(
                        "[%s] L%d(%s) 성공 | %.1fms",
                        self.task_type, level, source, latency
                    )
                    
                    return FallbackResult(
                        content=result,
                        source=source,
                        level=level,
                        latency_ms=latency,
                    )
                else:
                    logger.warning("[%s] L%d(%s) 빈 결과", self.task_type, level, source)
                    
            except Exception as e:
                latency = (time.time() - start) * 1000
                logger.warning(
                    "[%s] L%d(%s) 실패 [%.1fms]: %s",
                    self.task_type, level, source, latency, str(e)
                )
        
        # 모든 레벨 실패
        self._stats["failures"] += 1
        logger.error("[%s] 모든 폴백 레벨 실패", self.task_type)
        
        return FallbackResult(
            content={"text": "[FALLBACK EXHAUSTED] 모든 생성 레벨이 실패했습니다.", "error": True},
            source="exhausted",
            level=0,
        )
    
    def get_stats(self) -> dict:
        """폴백 통계를 반환합니다."""
        return {
            "task_type": self.task_type,
            **self._stats,
        }


# ═══════════════════════════════════════════════════════
# 가설 생성용 폴백 함수들
# ═══════════════════════════════════════════════════════

def hypothesis_l1_gemini(context: dict) -> Optional[dict]:
    """L1: Gemini API로 가설 생성."""
    from api.agents.gemini_client import generate_hypothesis, is_available
    
    if not is_available():
        return None
    
    return generate_hypothesis(context.get("kg_context", {}))


def hypothesis_l2_kg_cache(context: dict) -> Optional[dict]:
    """
    L2: KG에서 동일 도메인의 기존 가설 검색 → 변수 조합 변경하여 재활용.
    
    기존 KG에 검증된 가설이 있으면 변수만 바꿔서 새 가설로 변환합니다.
    """
    import random
    
    try:
        from api.graph import kg
    except ImportError:
        return None
    
    if not kg.initialized or not kg.graph.edges:
        return None
    
    # 기존 hypothesis 엣지에서 검색
    existing_hypotheses = [
        (u, v, data) for u, v, data in kg.graph.edges(data=True)
        if data.get("relation") == "hypothesis" and data.get("hypothesis_text")
    ]
    
    if not existing_hypotheses:
        return None
    
    # 가장 최근 가설을 기반으로 변형
    base = random.choice(existing_hypotheses)
    base_text = base[2]["hypothesis_text"]
    
    # 변수 교체 시도
    nodes = list(kg.graph.nodes(data=True))
    treatments = [n for n, d in nodes if d.get("category") == "Treatment"]
    confounders = [n for n, d in nodes if d.get("category") == "Confounder"]
    
    if treatments and confounders:
        new_treatment = random.choice(treatments)
        new_confounder = random.choice(confounders)
        text = f"[KG 기반 변형] {new_confounder}이(가) {new_treatment}에 미치는 인과적 영향 가설"
    else:
        text = f"[KG 기반] {base_text}의 심화 분석"
    
    return {
        "text": text,
        "source": FallbackLevel.KG_CACHE,
        "reasoning": f"KG의 기존 가설({base[0]}→{base[1]})에서 변수를 교체하여 재생성",
    }


def hypothesis_l3_template(context: dict) -> Optional[dict]:
    """L3: 구조화 템플릿으로 가설 생성 (최후의 수단)."""
    import random
    
    templates = context.get("templates", [
        "{confounder}이(가) {treatment}의 인과 효과를 조절하는 교란 변수로 작용할 수 있다.",
        "{treatment}이(가) {outcome}에 미치는 처치 효과에서 {confounder}의 이질적 효과가 존재한다.",
    ])
    
    variables = context.get("variables", {})
    confounder = variables.get("confounder", "Unknown")
    treatment = variables.get("treatment", "Unknown")
    outcome = variables.get("outcome", "Unknown")
    
    text = random.choice(templates).format(
        confounder=confounder, treatment=treatment, outcome=outcome
    )
    
    return {
        "text": text,
        "source": FallbackLevel.TEMPLATE,
        "reasoning": "템플릿 기반 자동 생성",
    }


# ═══════════════════════════════════════════════════════
# Critic 판정용 폴백 함수들
# ═══════════════════════════════════════════════════════

def critic_l1_gemini(context: dict) -> Optional[dict]:
    """L1: Gemini API로 판정 생성."""
    from api.agents.gemini_client import is_available
    
    if not is_available():
        return None
    
    # Gemini를 통한 구조화 판정
    try:
        from api.agents.gemini_client import generate_critique
        return generate_critique(context.get("experiment_result", {}))
    except (ImportError, AttributeError):
        return None


def critic_l2_kg_cache(context: dict) -> Optional[dict]:
    """L2: 유사 실험 결과의 판정을 KG에서 검색."""
    try:
        from api.graph import kg
    except ImportError:
        return None
    
    exp = context.get("experiment_result", {})
    method = exp.get("method", "")
    
    # KG에서 같은 방법론의 이전 판정 검색
    similar_edges = [
        data for _, _, data in kg.graph.edges(data=True)
        if data.get("method") == method and data.get("verdict")
    ]
    
    if not similar_edges:
        return None
    
    prev_verdict = similar_edges[-1]["verdict"]
    return {
        "text": f"[KG 유사 판정] 이전 {method} 실험과 동일 방법론 기반 판정: {prev_verdict}",
        "source": FallbackLevel.KG_CACHE,
        "action": prev_verdict,
    }


def critic_l3_rules(context: dict) -> Optional[dict]:
    """L3: ConstitutionGuard 체크리스트만으로 규칙 기반 판정."""
    exp = context.get("experiment_result", {})
    
    # 규칙 기반 판정
    issues = []
    action = "ACCEPT"
    
    # 규칙 1: HALTED → 즉시 REJECT
    if exp.get("experiment_source") == "HALTED":
        return {"text": "HALTED — 자동 REJECT", "source": FallbackLevel.TEMPLATE, "action": "REJECT"}
    
    # 규칙 2: 표본 수 < 500 → REVISE
    if exp.get("sample_size", 0) < 500:
        issues.append(f"표본 수 부족: {exp.get('sample_size', 0)}")
        action = "REVISE"
    
    # 규칙 3: ConstitutionGuard 위반 → REJECT
    cv = exp.get("constitution_verdict", {})
    if not cv.get("passed", True):
        issues.append(f"헌법 위반: {cv.get('violations', [])}")
        action = "REJECT"
    
    # 규칙 4: R² < 0.1 → REVISE
    perf = exp.get("model_performance", {})
    if perf.get("r2_treated", 1) < 0.1:
        issues.append(f"R² 매우 낮음: {perf.get('r2_treated')}")
        if action != "REJECT":
            action = "REVISE"
    
    text = f"[규칙 기반] 판정: {action}" + (f" | 이슈: {', '.join(issues)}" if issues else "")
    
    return {
        "text": text,
        "source": FallbackLevel.TEMPLATE,
        "action": action,
        "issues": issues,
    }


# ═══════════════════════════════════════════════════════
# 사전 구성된 체인 팩토리
# ═══════════════════════════════════════════════════════

def create_hypothesis_chain() -> LLMFallbackChain:
    """가설 생성용 3단계 폴백 체인을 생성합니다."""
    chain = LLMFallbackChain("hypothesis")
    chain.add_level(hypothesis_l1_gemini, 1, FallbackLevel.GEMINI)
    chain.add_level(hypothesis_l2_kg_cache, 2, FallbackLevel.KG_CACHE)
    chain.add_level(hypothesis_l3_template, 3, FallbackLevel.TEMPLATE)
    return chain


def create_critic_chain() -> LLMFallbackChain:
    """Critic 판정용 3단계 폴백 체인을 생성합니다."""
    chain = LLMFallbackChain("critic")
    chain.add_level(critic_l1_gemini, 1, FallbackLevel.GEMINI)
    chain.add_level(critic_l2_kg_cache, 2, FallbackLevel.KG_CACHE)
    chain.add_level(critic_l3_rules, 3, FallbackLevel.TEMPLATE)
    return chain
