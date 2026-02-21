"""
Academic Forum v2 — Gemini 기반 에이전트 간 학술 토론 시스템
========================================
Theorist, Engineer, Critic이 연구 결과에 대해 Gemini LLM을 통해
실시간으로 토론하고 합의를 도출하는 다중 에이전트 학술 포럼.

[토론 구조 (3라운드)]
1. Opening: 각 에이전트가 논제에 대한 초기 입장 발표
2. Rebuttal: 이전 발언에 대한 반론/보충
3. Closing: 최종 입장 정리 + 합의 도출

[v2 변경: 하드코딩 → Gemini 전면 전환]
- 토론 주제: KG 컨텍스트 기반 Gemini 동적 생성
- 발언 내용: 에이전트 페르소나 + 멀티턴 컨텍스트 기반 Gemini 생성
- 합의 도출: 전체 토론 내용 종합하여 Gemini가 판단
- Fallback: API 실패 시만 간소화된 규칙 기반 응답 사용
"""
import logging
from datetime import datetime, timezone, timedelta

from api.graph import kg
from api.agents.gemini_client import (
    generate_debate_topic,
    generate_debate_response,
    generate_consensus,
    is_available as is_gemini_available,
    AGENT_PERSONAS,
)

logger = logging.getLogger("whylab.forum")

# 한국 표준시 (KST = UTC+9)
KST = timezone(timedelta(hours=9))

# 토론 참여 에이전트 순서
DEBATE_ROLES = ["Theorist", "Engineer", "Critic"]

# 토론 라운드
DEBATE_PHASES = ["opening", "rebuttal", "closing"]


def _get_kg_context() -> dict:
    """현재 KG 상태를 토론 컨텍스트로 변환합니다."""
    if not kg.initialized:
        kg.initialize_seed_data()
    
    nodes = [
        {"name": n, "category": d.get("category", "?")}
        for n, d in kg.graph.nodes(data=True)
    ]
    edges = [
        {
            "source": u,
            "target": v,
            "relation": data.get("relation", "?"),
            "weight": data.get("weight", 0),
        }
        for u, v, data in kg.graph.edges(data=True)
    ]
    
    # 최근 실험 결과 수집
    recent_results = []
    for u, v, data in kg.graph.edges(data=True):
        if data.get("verified") or data.get("experiment_id"):
            recent_results.append({
                "hypothesis": data.get("hypothesis_text", f"{u} → {v}"),
                "ate": data.get("ate", "N/A"),
                "verdict": data.get("verdict", "N/A"),
            })
    
    return {
        "nodes": nodes[:15],
        "edges": edges[:10],
        "recent_results": recent_results[:5],
    }


def _fallback_topic() -> dict:
    """Gemini 실패 시 KG 기반 간소화된 토론 주제 생성."""
    if not kg.initialized:
        kg.initialize_seed_data()
    
    edges = list(kg.graph.edges(data=True))
    if edges:
        u, v, data = edges[0]
        return {
            "topic": f"'{u}'와 '{v}' 사이의 {data.get('relation', '인과')} 관계가 외부 타당도를 갖는가?",
            "context": f"KG에서 신뢰도 {data.get('weight', 0):.0%}로 관측된 관계에 대한 검증 필요",
            "domain": "인과추론",
        }
    
    return {
        "topic": "현재 Knowledge Graph의 인과 구조가 충분히 탐색되었는가?",
        "context": "초기 연구 단계에서의 KG 확장 전략 논의",
        "domain": "연구 설계",
    }


def _fallback_response(role: str, topic: str, phase: str) -> str:
    """Gemini 실패 시 간소화된 규칙 기반 응답."""
    persona = AGENT_PERSONAS.get(role, AGENT_PERSONAS["Theorist"])
    
    phase_responses = {
        "opening": f"{persona['bias']} 관점에서 '{topic}'에 대해 추가 검증이 필요하다고 봅니다.",
        "rebuttal": f"앞선 주장에 대해 {persona['bias']}의 관점에서 보완할 점이 있습니다.",
        "closing": f"종합적으로 {persona['bias']} 측면의 추가 연구가 필요합니다.",
    }
    return phase_responses.get(phase, phase_responses["opening"])


def run_forum_debate() -> dict:
    """
    Academic Forum 토론을 실행합니다 (v2: Gemini 기반).
    
    Returns:
        dict: 토론 결과 (논제, 발언록, 합의)
    """
    now = datetime.now(KST)
    logger.info("[FORUM] 학술 토론 세션 시작")
    
    kg_context = _get_kg_context()
    all_statements = []
    debate_log = []
    
    def add_entry(phase: str, role: str, content: str):
        """발언 기록"""
        entry = {
            "phase": phase,
            "role": role,
            "speaker": AGENT_PERSONAS.get(role, {}).get("name", role),
            "content": content,
            "timestamp": datetime.now(KST).isoformat(),
        }
        all_statements.append(entry)
        debate_log.append(entry)
        logger.info(f"[FORUM] [{phase}] {role}({entry['speaker']}): {content[:80]}...")
    
    # ── Phase 0: 토론 주제 생성 ──
    topic_data = None
    if is_gemini_available():
        topic_data = generate_debate_topic(kg_context)
    
    if not topic_data:
        topic_data = _fallback_topic()
        logger.warning("[FORUM] 토론 주제 생성 fallback 사용")
    
    topic = topic_data["topic"]
    topic_source = "gemini" if is_gemini_available() and topic_data else "fallback"
    logger.info(f"[FORUM] 논제: {topic} (출처: {topic_source})")
    
    # ── Phase 1~3: 3라운드 토론 ──
    for phase in DEBATE_PHASES:
        for role in DEBATE_ROLES:
            response = None
            
            if is_gemini_available():
                response = generate_debate_response(
                    role=role,
                    topic=topic,
                    phase=phase,
                    previous_statements=[
                        {"role": s["role"], "content": s["content"]}
                        for s in all_statements
                    ],
                    kg_context=kg_context,
                )
            
            if not response:
                response = _fallback_response(role, topic, phase)
            
            add_entry(phase, role, response)
    
    # ── Phase 4: 합의 도출 ──
    consensus = None
    if is_gemini_available():
        consensus = generate_consensus(topic, all_statements)
    
    if not consensus:
        # 간소화된 규칙 기반 합의
        consensus = {
            "label": "조건부 합의",
            "summary": f"'{topic}'에 대해 추가 검증이 필요하다는 점에 합의하였으나, "
                       "방법론적 접근에 대한 이견이 남아 있음.",
            "next_steps": ["추가 데이터 수집", "민감도 분석 실시"],
        }
    
    # KG에 토론 결과 기록
    _record_forum_to_kg(topic_data, consensus)
    
    result = {
        "session_id": f"FORUM-{int(now.timestamp()) % 100000:05d}",
        "topic": topic_data,
        "topic_source": topic_source,
        "debate_log": debate_log,
        "consensus": consensus,
        "participants": [
            {"role": r, **AGENT_PERSONAS.get(r, {})} for r in DEBATE_ROLES
        ],
        "total_statements": len(all_statements),
        "gemini_used": is_gemini_available(),
        "started_at": now.isoformat(),
        "ended_at": datetime.now(KST).isoformat(),
    }
    
    logger.info(
        f"[FORUM] 토론 완료: {consensus.get('label', '?')} | "
        f"발언 {len(all_statements)}건 | Gemini: {is_gemini_available()}"
    )
    
    return result


def _record_forum_to_kg(topic_data: dict, consensus: dict):
    """토론 결과를 Knowledge Graph에 기록합니다."""
    try:
        # 토론 주제 노드 추가
        forum_node = f"Forum:{topic_data.get('domain', 'general')}"
        if not kg.graph.has_node(forum_node):
            kg.graph.add_node(
                forum_node,
                category="Forum",
                topic=topic_data.get("topic", ""),
            )
        
        # 합의 결과 반영
        kg.graph.nodes[forum_node]["last_consensus"] = consensus.get("label", "")
        kg.graph.nodes[forum_node]["last_summary"] = consensus.get("summary", "")
        kg.graph.nodes[forum_node]["updated_at"] = datetime.now(KST).isoformat()
        
    except Exception as e:
        logger.error(f"[FORUM] KG 기록 실패: {e}")
