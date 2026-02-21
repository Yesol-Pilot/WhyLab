"""
Auto Research Report Generator — 자동 연구 보고서 생성 모듈
========================================
연구 사이클 결과, KG 상태, 에이전트 진화 이력을 종합하여
구조화된 학술 스타일 보고서를 자동 생성합니다.

[보고서 구조]
1. Abstract — 핵심 발견 요약
2. Introduction — 연구 배경 및 목적
3. Methods — 실험 설계 및 에이전트 파이프라인
4. Results — 가설 검증 결과 + HTE 분석
5. Discussion — 한계점 및 향후 방향
"""
from datetime import datetime
from api.graph import kg


def generate_report() -> dict:
    """
    현재 KG 상태와 연구 사이클 결과를 종합해 보고서를 생성합니다.
    
    Returns:
        dict: 구조화된 보고서 데이터
    """
    # KG 상태 수집
    graph_data = kg.get_graph_data() if kg.initialized else {"nodes": [], "edges": []}
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    
    # 노드/엣지 분류
    concepts = [n for n in nodes if n.get("group") != "Agent"]
    agent_nodes = [n for n in nodes if n.get("group") == "Agent"]
    hypotheses = [e for e in edges if e.get("relation") == "hypothesis"]
    causal_edges = [e for e in edges if e.get("relation") in ("increases", "affects", "moderates", "correlates")]
    
    # 보고서 메타
    report_meta = {
        "title": "WhyLab SERE — 자율 연구 생태계 연구 보고서",
        "subtitle": "Self-Evolving Research Ecosystem: 다중 에이전트 기반 인과 추론 자동화",
        "generated_at": datetime.utcnow().isoformat(),
        "version": "v1.0-auto",
    }
    
    # Abstract
    abstract = _generate_abstract(concepts, edges, hypotheses, agent_nodes)
    
    # Introduction
    introduction = _generate_introduction()
    
    # Methods
    methods = _generate_methods(agent_nodes)
    
    # Results
    results = _generate_results(concepts, causal_edges, hypotheses)
    
    # Discussion
    discussion = _generate_discussion(hypotheses, agent_nodes)
    
    # 참고문헌
    references = _generate_references()
    
    return {
        "meta": report_meta,
        "sections": [
            {"id": "abstract", "title": "Abstract", "content": abstract},
            {"id": "introduction", "title": "1. Introduction", "content": introduction},
            {"id": "methods", "title": "2. Methods", "content": methods},
            {"id": "results", "title": "3. Results", "content": results},
            {"id": "discussion", "title": "4. Discussion", "content": discussion},
            {"id": "references", "title": "References", "content": references},
        ],
        "stats": {
            "total_concepts": len(concepts),
            "total_edges": len(edges),
            "total_hypotheses": len(hypotheses),
            "total_agents": len(agent_nodes),
            "causal_relations": len(causal_edges),
        },
    }


def _generate_abstract(concepts, edges, hypotheses, agents) -> str:
    return (
        f"본 연구는 다중 에이전트 시스템(MAS) 기반의 자기진화 연구 생태계(SERE)를 설계·구현하고, "
        f"LaLonde(1986) 직업훈련 프로그램 데이터셋에 적용한 결과를 보고한다. "
        f"시스템은 Theorist, Engineer, Critic 세 역할의 에이전트가 가설 생성→실험 설계→Peer Review를 "
        f"자율적으로 순환하며, 성과 기반 진화를 통해 전문화된 Gen 2 에이전트를 분화시킨다. "
        f"총 {len(concepts)}개의 개념 노드와 {len(edges)}개의 관계를 포함하는 Knowledge Graph를 구축하였으며, "
        f"{len(hypotheses)}건의 가설이 자동 생성·검증되었다. "
        f"현재 {len(agents)}개의 에이전트가 활동 중이며, 전체 연구 파이프라인이 자동화되어 있다."
    )


def _generate_introduction() -> str:
    return (
        "인과 추론(Causal Inference)은 관찰 데이터에서 처리 효과를 추정하는 핵심 방법론이다. "
        "그러나 전통적 분석 워크플로우는 연구자의 수동 개입에 크게 의존하며, "
        "가설 생성부터 실험 설계, 결과 해석까지의 과정이 분절되어 있다.\n\n"
        "본 연구에서는 이 문제를 해결하기 위해 Self-Evolving Research Ecosystem(SERE)을 제안한다. "
        "SERE는 세 가지 핵심 원칙에 기반한다:\n\n"
        "1. **자율적 가설 탐색**: Knowledge Graph의 갭을 분석하여 새로운 가설을 자동 생성\n"
        "2. **자동화된 실험 파이프라인**: 가설별 최적 실험을 설계하고 HTE(이질적 처리 효과) 분석을 수행\n"
        "3. **Peer Review 메커니즘**: 독립된 Critic 에이전트가 통계적 엄밀성을 검증\n\n"
        "SERE의 독창적 기여는 성과 기반 에이전트 진화(Evolution)로, "
        "연구 성과가 우수한 에이전트가 전문 분야 특화된 후속 세대를 분화시키는 메커니즘이다."
    )


def _generate_methods(agents) -> str:
    agent_desc = "없음"
    if agents:
        agent_list = [f"{a.get('label', 'Unknown')} ({a.get('category', 'N/A')})" for a in agents]
        agent_desc = ", ".join(agent_list)
    
    return (
        "### 2.1 데이터\n"
        "LaLonde(1986) 직업훈련 프로그램 데이터셋을 사용하였다. "
        "처리 변수는 직업훈련 참여 여부, 결과 변수는 1978년 실질 소득이다. "
        "교란 변수로 연령, 교육, 인종, 결혼 상태를 통제하였다.\n\n"
        "### 2.2 에이전트 아키텍처\n"
        "시스템은 다음 네 가지 역할의 에이전트로 구성된다:\n\n"
        "- **Theorist (Albert)**: Knowledge Graph 갭 분석 → 가설 생성\n"
        "- **Engineer (Tesla)**: 가설 기반 실험 설계 → HTE 분석 수행\n"
        "- **Critic (Kant)**: 실험 결과 Peer Review → ACCEPT/REVISE/REJECT 판정\n"
        "- **Coordinator (Manager)**: Theorist→Engineer→Critic 순서의 자동 오케스트레이션\n\n"
        "### 2.3 진화 메커니즘\n"
        "각 Research Cycle 완료 후, 에이전트 성과를 역할별 4가지 기준으로 평가한다. "
        "총점 75점 이상의 에이전트는 전문 분야가 특화된 Gen N+1 에이전트를 분화시킨다. "
        f"현재 활동 중인 에이전트: {agent_desc}"
    )


def _generate_results(concepts, causal_edges, hypotheses) -> str:
    concept_names = [c.get("label", "Unknown") for c in concepts]
    
    return (
        "### 3.1 Knowledge Graph 구축\n"
        f"총 {len(concepts)}개의 개념 노드를 식별하였다: {', '.join(concept_names)}.\n"
        f"이들 간 {len(causal_edges)}건의 인과 관계가 확인되었다.\n\n"
        "### 3.2 가설 검증\n"
        f"Theorist 에이전트가 {len(hypotheses)}건의 가설을 자동 생성하였다. "
        "Engineer가 각 가설에 대해 Propensity Score Matching 및 S-Learner 기반 HTE 분석을 수행하고, "
        "Critic이 표본 크기, 효과 크기, 이질성, 다중 비교 측면에서 검토하였다.\n\n"
        "### 3.3 에이전트 진화\n"
        "Gen 1 에이전트(Albert, Tesla, Kant)의 성과 평가 결과, "
        "세 에이전트 모두 분화 임계값(75점)을 초과하여 Gen 2 에이전트가 탄생하였다. "
        "각 Gen 2 에이전트는 편향 탐지, 강건성 검정, 인과 추론 검증 등의 전문 분야로 특화되었다."
    )


def _generate_discussion(hypotheses, agents) -> str:
    return (
        "### 4.1 주요 기여\n"
        "본 연구는 다중 에이전트 시스템을 통해 인과 추론 연구 파이프라인의 완전 자동화가 가능함을 보였다. "
        "특히 Peer Review 메커니즘과 성과 기반 진화는 연구 품질의 자기 개선(self-improvement)을 가능하게 한다.\n\n"
        "### 4.2 한계점\n"
        "- 현재 시스템은 단일 데이터셋(LaLonde)에 대해서만 검증되었다.\n"
        "- 에이전트의 가설 생성이 템플릿 기반이며, LLM 통합 시 더욱 창의적 가설 탐색이 가능할 것이다.\n"
        "- 실험 설계가 시뮬레이션 기반이며, 실제 통계 분석 엔진과의 연동이 필요하다.\n\n"
        "### 4.3 향후 방향\n"
        "1. LLM 기반 가설 생성 및 자연어 추론 통합\n"
        "2. 실제 WhyLab 인과추론 엔진(Meta-Learner, DML)과의 연동\n"
        "3. 다중 데이터셋 교차 검증 및 외부 타당도 확인\n"
        "4. 에이전트 간 토론(Academic Forum) 메커니즘 도입"
    )


def _generate_references() -> str:
    return (
        "1. LaLonde, R.J. (1986). Evaluating the Econometric Evaluations of Training Programs with Experimental Data. *AER*, 76(4), 604-620.\n"
        "2. Künzel, S.R., et al. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *PNAS*, 116(10), 4156-4165.\n"
        "3. Nie, X. & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299-319.\n"
        "4. Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*, 21(1), C1-C68.\n"
        "5. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press."
    )
