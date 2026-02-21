"""
Agent Registry — 에이전트 역할 정의 및 통합 레지스트리 (Sprint 30)
================================================================
기존 engine/agents/와 api/agents/의 이원화 문제를 해결하기 위한
중앙 에이전트 등록소.

[설계 문서 §3.1 계층적 오케스트레이터-워커 패턴]

계층 구조:
┌───────────────────────────────────────────┐
│  지휘 및 제어 (Control Plane)              │
│  - Director: 연구 아젠다 선정              │
│  - Coordinator: 유일한 전역 오케스트레이터  │
├───────────────────────────────────────────┤
│  연구 및 실행 (Execution Plane)            │
│  - Theorist: KG 기반 가설 생성             │
│  - Engineer: 샌드박스 실험 실행             │
│  - Discovery: 인과 구조 발견 (PC/GES)      │
├───────────────────────────────────────────┤
│  평가 및 검증 (Evaluation Plane)           │
│  - Critic: 방법론 검증 + 헌법 가드          │
│  - Debate: Growth/Risk/PO 3자 토론         │
├───────────────────────────────────────────┤
│  진화 및 자동화 (Meta Plane)               │
│  - Evolution: 세대 분화 + 성과 평가         │
│  - Autopilot: 무한 자율 순환               │
│  - Method Registry: UCB1 메서드 선택        │
│  - Forum: 에이전트 간 학술 토론             │
│  - Report: 자동 연구 보고서 생성            │
└───────────────────────────────────────────┘
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AgentPlane(Enum):
    """에이전트 실행 계층."""
    CONTROL = "control"       # 지휘 및 제어
    EXECUTION = "execution"   # 연구 및 실행
    EVALUATION = "evaluation" # 평가 및 검증
    META = "meta"             # 진화 및 자동화


@dataclass
class AgentSpec:
    """에이전트 사양."""
    id: str
    name: str
    role: str
    plane: AgentPlane
    module_path: str          # 실제 코드 위치
    uses_llm: bool = False    # Gemini 등 LLM 사용 여부
    description: str = ""


# ── 에이전트 레지스트리 (전체 목록) ──
AGENT_REGISTRY: list[AgentSpec] = [
    # 지휘 및 제어 계층
    AgentSpec(
        id="director",
        name="Director",
        role="Lab Director",
        plane=AgentPlane.CONTROL,
        module_path="engine.agents.director",
        description="Grand Challenges에서 연구 주제를 선정하고 아젠다를 설정",
    ),
    AgentSpec(
        id="coordinator-1",
        name="Manager",
        role="Coordinator",
        plane=AgentPlane.CONTROL,
        module_path="api.agents.coordinator",
        description="유일한 전역 오케스트레이터. 모든 에이전트 간 메시지 패싱을 중재",
    ),

    # 연구 및 실행 계층
    AgentSpec(
        id="theorist-1",
        name="Albert",
        role="Theorist",
        plane=AgentPlane.EXECUTION,
        module_path="api.agents.theorist",
        uses_llm=True,
        description="KG Gap 분석 → 가설 생성 (Gemini 통합)",
    ),
    AgentSpec(
        id="engineer-1",
        name="Tesla",
        role="Engineer",
        plane=AgentPlane.EXECUTION,
        module_path="api.agents.engineer",
        description="Code-Then-Execute 패턴으로 SandboxExecutor에서 실험 실행",
    ),
    AgentSpec(
        id="discovery",
        name="Discovery",
        role="Causal Discovery",
        plane=AgentPlane.EXECUTION,
        module_path="engine.agents.discovery",
        description="PC/GES 알고리즘으로 데이터 기반 인과 구조(DAG) 발견",
    ),

    # 평가 및 검증 계층
    AgentSpec(
        id="critic-1",
        name="Kant",
        role="Critic",
        plane=AgentPlane.EVALUATION,
        module_path="api.agents.critic",
        uses_llm=True,
        description="방법론적 타당성 검증 + 연구 헌법 가드레일 (LLM-as-a-Judge)",
    ),
    AgentSpec(
        id="debate",
        name="Debate Cell",
        role="Debate",
        plane=AgentPlane.EVALUATION,
        module_path="engine.agents.debate",
        uses_llm=True,
        description="Growth/Risk/PO 3자 토론 → Rollout 의사결정",
    ),

    # 진화 및 자동화 계층
    AgentSpec(
        id="evolution",
        name="Evolution Engine",
        role="Evolution",
        plane=AgentPlane.META,
        module_path="api.agents.evolution",
        description="에이전트 성과 평가 → 세대 분화 (자연 선택)",
    ),
    AgentSpec(
        id="autopilot",
        name="Autopilot",
        role="Autopilot",
        plane=AgentPlane.META,
        module_path="api.agents.autopilot",
        description="무한 자율 순환 엔진 (가설→실험→비판→진화)",
    ),
    AgentSpec(
        id="method-registry",
        name="Method Registry",
        role="Method Selection",
        plane=AgentPlane.META,
        module_path="api.agents.method_registry",
        description="UCB1 밴디트 기반 최적 추정 알고리즘 자율 선택",
    ),
    AgentSpec(
        id="forum",
        name="Forum",
        role="Academic Forum",
        plane=AgentPlane.META,
        module_path="api.agents.forum",
        description="에이전트 간 학술 토론 엔진",
    ),
    AgentSpec(
        id="report",
        name="Report Generator",
        role="Report",
        plane=AgentPlane.META,
        module_path="api.agents.report_generator",
        description="자동 연구 보고서 생성",
    ),
]


def get_agent_by_id(agent_id: str) -> Optional[AgentSpec]:
    """에이전트 ID로 사양 조회."""
    for spec in AGENT_REGISTRY:
        if spec.id == agent_id:
            return spec
    return None


def get_agents_by_plane(plane: AgentPlane) -> list[AgentSpec]:
    """계층별 에이전트 목록 조회."""
    return [s for s in AGENT_REGISTRY if s.plane == plane]


def get_registry_summary() -> dict:
    """레지스트리 요약 조회."""
    summary = {}
    for plane in AgentPlane:
        agents = get_agents_by_plane(plane)
        summary[plane.value] = [
            {
                "id": a.id,
                "name": a.name,
                "role": a.role,
                "uses_llm": a.uses_llm,
                "module": a.module_path,
            }
            for a in agents
        ]
    return summary
