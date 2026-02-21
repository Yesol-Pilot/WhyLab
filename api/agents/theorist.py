"""
Theorist Agent (Albert) — 가설 생성 모듈
========================================
Knowledge Graph의 빈틈(Gap)을 탐색하여 새로운 인과 가설을 자율적으로 생성합니다.

[v2: 적응형 가설 생성]
- MethodRegistry(UCB1 밴디트)로 가설 전략을 적응적으로 선택
- 성공한 전략은 가중치가 상승하여 더 자주 선택됨
- 세대가 높을수록 고급 전략(비선형 인과, 교호작용 네트워크) 접근 가능
"""
import time
import random
from datetime import datetime
from typing import Optional

from api.graph import kg
from api.agents.method_registry import method_registry


# 기본 템플릿 (레거시 호환용, 실제로는 MethodRegistry에서 선택)
HYPOTHESIS_TEMPLATES = [
    "'{confounder}'이(가) '{outcome}'에 미치는 인과 효과는 '{moderator}'에 의해 조절(moderation)될 수 있다.",
    "'{treatment}'의 효과가 '{confounder}' 수준에 따라 이질적(heterogeneous)일 가능성이 있다.",
    "'{confounder}'와 '{outcome}' 사이의 관계는 '{treatment}'를 매개(mediation)로 하여 간접적으로 작용할 수 있다.",
    "'{confounder}'이(가) '{treatment}' 배정에 영향을 미쳐 선택 편향(selection bias)을 유발할 수 있다.",
]


def analyze_knowledge_gaps():
    """
    KG에서 연구 갭을 분석합니다.
    
    Returns:
        list[dict]: 발견된 갭 목록 (낮은 신뢰도 엣지, 미연결 노드 등)
    """
    if not kg.initialized:
        kg.initialize_seed_data()
    
    gaps = []
    
    # 1단계: 낮은 신뢰도 엣지 탐색 (weight < 0.3)
    for u, v, data in kg.graph.edges(data=True):
        weight = data.get("weight", 0.0)
        relation = data.get("relation", "unknown")
        if weight < 0.3 or relation == "hypothesis":
            gaps.append({
                "type": "low_confidence",
                "source": u,
                "target": v,
                "relation": relation,
                "weight": weight,
                "description": f"'{u}' → '{v}' 관계의 신뢰도가 {weight:.1%}로 낮음 (검증 필요)"
            })
    
    # 2단계: 연결되지 않은 노드 쌍 탐색
    nodes = list(kg.graph.nodes(data=True))
    confounders = [n for n, d in nodes if d.get("category") == "Confounder"]
    treatments = [n for n, d in nodes if d.get("category") == "Treatment"]
    
    for c in confounders:
        for t in treatments:
            if not kg.graph.has_edge(c, t) and not kg.graph.has_edge(t, c):
                gaps.append({
                    "type": "missing_link",
                    "source": c,
                    "target": t,
                    "relation": "unknown",
                    "weight": 0.0,
                    "description": f"'{c}' ↔ '{t}' 사이의 관계가 아직 탐색되지 않음"
                })
    
    return gaps


def generate_hypothesis(gap: Optional[dict] = None) -> dict:
    """
    분석된 갭을 기반으로 가설을 생성합니다.
    
    [v3] Gemini API 우선 → 실패 시 MethodRegistry+템플릿 fallback
    """
    if not kg.initialized:
        kg.initialize_seed_data()
    
    gaps = analyze_knowledge_gaps()
    if not gaps:
        return {
            "id": "H-000",
            "text": "현재 Knowledge Graph에 탐색할 갭이 없습니다. 추가 데이터가 필요합니다.",
            "confidence": 0.0,
            "source_gap": None,
            "created_at": datetime.utcnow().isoformat(),
        }
    
    selected_gap = gap or sorted(gaps, key=lambda g: g["weight"])[0]
    
    nodes = list(kg.graph.nodes(data=True))
    confounders = [n for n, d in nodes if d.get("category") == "Confounder"]
    treatments = [n for n, d in nodes if d.get("category") == "Treatment"]
    outcomes = [n for n, d in nodes if d.get("category") == "Outcome"]
    
    generation = 1 + min(len(kg.graph.edges) // 10, 3)
    selected_method = method_registry.select_method("hypothesis", generation)
    
    # ── LLM Fallback Chain 실행 (L1:Gemini → L2:KG → L3:Template) ──
    from api.agents.llm_fallback import create_hypothesis_chain
    
    # 컨텍스트 구성
    fallback_context = {
        "kg_context": {
            "nodes": [{"name": n, "category": d.get("category", "?")} for n, d in nodes[:15]],
            "edges": [
                {"source": u, "target": v, "relation": data.get("relation", "?"), "weight": data.get("weight", 0)}
                for u, v, data in kg.graph.edges(data=True)
            ][:10],
            "gaps": [{"source": g["source"], "target": g["target"], "description": g["description"]} for g in gaps[:5]],
            "recent_results": [],
        },
        "variables": {
            "confounder": random.choice(confounders) if confounders else "Unknown",
            "treatment": random.choice(treatments) if treatments else "Unknown",
            "outcome": random.choice(outcomes) if outcomes else "Unknown",
        },
        "templates": selected_method.params.get("templates", HYPOTHESIS_TEMPLATES),
    }
    
    chain = create_hypothesis_chain()
    result = chain.execute(fallback_context)
    
    hypothesis_text = result.content.get("text", "")
    hypothesis_source = result.source
    gemini_reasoning = result.content.get("reasoning", "")
    
    if not hypothesis_text:
        hypothesis_text = "가설 생성 실패 (모든 폴백 소진)"
    
    hypothesis_id = f"H-{int(time.time()) % 10000:04d}"
    
    kg.graph.add_edge(
        selected_gap["source"],
        selected_gap["target"],
        relation="hypothesis",
        weight=0.0,
        hypothesis_id=hypothesis_id,
        hypothesis_text=hypothesis_text,
    )
    
    return {
        "id": hypothesis_id,
        "text": hypothesis_text,
        "source": selected_gap["source"],
        "target": selected_gap["target"],
        "method_used": selected_method.name,
        "method_generation": selected_method.generation,
        "hypothesis_source": hypothesis_source,  # "gemini" 또는 "template"
        "gemini_reasoning": gemini_reasoning,
        "confidence": 0.0,
        "source_gap": selected_gap,
        "gaps_found": len(gaps),
        "kg_stats": kg.get_stats(),
        "created_at": datetime.utcnow().isoformat(),
    }


def run_theorist_cycle() -> list[dict]:
    """
    Theorist의 전체 연구 사이클을 실행합니다.
    
    Returns:
        list[dict]: 연구 과정 로그 (step, message, timestamp)
    """
    logs = []
    
    def log(step: str, message: str):
        entry = {
            "step": step,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }
        logs.append(entry)
        return entry
    
    # Phase 1: KG 분석
    log("SCAN", "Knowledge Graph 스캔 시작...")
    time.sleep(0.3)  # 시뮬레이션 딜레이
    
    gaps = analyze_knowledge_gaps()
    log("SCAN", f"총 {len(gaps)}개의 연구 갭 발견.")
    
    if not gaps:
        log("ABORT", "탐색할 갭이 없습니다. 추가 데이터가 필요합니다.")
        return logs
    
    # Phase 2: 갭 분석 상세
    for i, gap in enumerate(gaps[:3]):  # 상위 3개만 분석
        log("ANALYZE", f"Gap #{i+1}: {gap['description']}")
        time.sleep(0.2)
    
    # Phase 3: 가설 생성
    log("HYPOTHESIZE", "가설 생성 중...")
    time.sleep(0.5)
    
    hypothesis = generate_hypothesis()
    log("HYPOTHESIZE", f"[전략: {hypothesis.get('method_used', 'N/A')} (Gen {hypothesis.get('method_generation', 1)})]")
    log("HYPOTHESIZE", f"[{hypothesis['id']}] {hypothesis['text']}")
    
    # 보상 피드백: 검증 전이므로 탐색 보상 (0.5)
    method_registry.reward_method(hypothesis.get('method_used', ''), "hypothesis", 0.5)
    
    # Phase 4: 결론
    stats = kg.get_stats()
    log("COMPLETE", f"연구 사이클 완료. KG: {stats['nodes']}노드, {stats['edges']}엣지. 다음 단계: Engineer에게 검증 요청.")
    
    return logs
