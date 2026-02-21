# -*- coding: utf-8 -*-
"""MACDiscoveryAgent 테스트."""

import pytest
import numpy as np

from engine.agents.mac_discovery import (
    MACDiscoveryAgent,
    PCSpecialist,
    GESSpecialist,
    LiNGAMSpecialist,
    VoteAggregator,
    Edge,
    DiscoveryResult,
)


@pytest.fixture
def linear_scm():
    """선형 SCM: X1 → X2 → X3, X1 → X3.

    X1 = noise1
    X2 = 0.8*X1 + noise2
    X3 = 0.5*X2 + 0.3*X1 + noise3
    """
    np.random.seed(42)
    n = 500
    x1 = np.random.normal(0, 1, n)
    x2 = 0.8 * x1 + np.random.normal(0, 0.5, n)
    x3 = 0.5 * x2 + 0.3 * x1 + np.random.normal(0, 0.5, n)
    data = np.column_stack([x1, x2, x3])
    names = ["X1", "X2", "X3"]
    return data, names


@pytest.fixture
def independent_data():
    """독립 변수 3개 (인과 관계 없음)."""
    np.random.seed(42)
    n = 300
    data = np.random.normal(0, 1, (n, 3))
    names = ["A", "B", "C"]
    return data, names


# ──────────────────────────────────────────────
# Specialist 테스트
# ──────────────────────────────────────────────

class TestPCSpecialist:

    def test_basic(self, linear_scm):
        """PC가 선형 SCM에서 엣지 발견."""
        data, names = linear_scm
        pc = PCSpecialist()
        result = pc.discover(data, names)
        assert result.algorithm == "PC"
        assert len(result.edges) > 0
        assert result.adjacency is not None

    def test_independent_sparse(self, independent_data):
        """독립 데이터에서 엣지 적음."""
        data, names = independent_data
        pc = PCSpecialist()
        result = pc.discover(data, names, alpha=0.01)
        # 독립이므로 엣지가 거의 없어야 함
        assert len(result.edges) <= 3


class TestGESSpecialist:

    def test_basic(self, linear_scm):
        """GES가 엣지 발견."""
        data, names = linear_scm
        ges = GESSpecialist()
        result = ges.discover(data, names)
        assert result.algorithm == "GES"
        assert len(result.edges) > 0

    def test_has_score(self, linear_scm):
        """BIC 점수 존재."""
        data, names = linear_scm
        ges = GESSpecialist()
        result = ges.discover(data, names)
        assert result.score != 0


class TestLiNGAMSpecialist:

    def test_basic(self, linear_scm):
        """LiNGAM이 인과 순서 발견."""
        data, names = linear_scm
        lingam = LiNGAMSpecialist()
        result = lingam.discover(data, names)
        assert result.algorithm == "LiNGAM"
        assert len(result.edges) > 0


# ──────────────────────────────────────────────
# Aggregator 테스트
# ──────────────────────────────────────────────

class TestVoteAggregator:

    def test_unanimous(self):
        """3개 에이전트가 동일 엣지에 투표 → 합의."""
        names = ["A", "B"]
        results = []
        for alg in ["PC", "GES", "LiNGAM"]:
            results.append(DiscoveryResult(
                algorithm=alg,
                edges=[Edge(source="A", target="B", discovered_by=[alg])],
                adjacency=np.array([[0, 1], [0, 0]]),
            ))
        agg = VoteAggregator(threshold=0.5)
        dag = agg.aggregate(results, names)
        assert len(dag.edges) >= 1
        assert dag.consensus_level > 0

    def test_no_agreement(self):
        """에이전트들이 다른 엣지에 투표 → 합의 낮음."""
        names = ["A", "B", "C"]
        r1 = DiscoveryResult(
            algorithm="PC",
            edges=[Edge(source="A", target="B")],
            adjacency=np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
        )
        r2 = DiscoveryResult(
            algorithm="GES",
            edges=[Edge(source="B", target="C")],
            adjacency=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
        )
        r3 = DiscoveryResult(
            algorithm="LiNGAM",
            edges=[Edge(source="C", target="A")],
            adjacency=np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
        )
        agg = VoteAggregator(threshold=0.6)
        dag = agg.aggregate([r1, r2, r3], names)
        # 2/3 이상 투표 없으므로 합의 엣지 없음
        assert len(dag.edges) == 0

    def test_stability_scores(self):
        """안정성 점수 범위 검증."""
        names = ["X", "Y"]
        results = [
            DiscoveryResult(
                algorithm="PC",
                edges=[Edge(source="X", target="Y")],
                adjacency=np.array([[0, 1], [0, 0]]),
            ),
            DiscoveryResult(
                algorithm="GES",
                edges=[Edge(source="X", target="Y")],
                adjacency=np.array([[0, 1], [0, 0]]),
            ),
        ]
        agg = VoteAggregator(threshold=0.5)
        dag = agg.aggregate(results, names)
        for score in dag.stability_scores.values():
            assert 0 <= score <= 1


# ──────────────────────────────────────────────
# MACDiscoveryAgent 통합
# ──────────────────────────────────────────────

class TestMACDiscoveryAgent:

    def test_full_pipeline(self, linear_scm):
        """전체 MAC 파이프라인."""
        data, names = linear_scm
        agent = MACDiscoveryAgent()
        dag = agent.discover(data, names)
        assert len(dag.edges) > 0
        assert dag.consensus_level > 0
        assert len(dag.variable_names) == 3

    def test_custom_specialists(self, linear_scm):
        """Specialist 선택."""
        data, names = linear_scm
        agent = MACDiscoveryAgent(specialists=["PC", "GES"])
        assert len(agent.specialists) == 2
        dag = agent.discover(data, names)
        assert len(dag.edges) >= 0

    def test_execute_interface(self, linear_scm):
        """파이프라인 execute 인터페이스."""
        data, names = linear_scm
        import pandas as pd
        df = pd.DataFrame(data, columns=names)
        agent = MACDiscoveryAgent()
        result = agent.execute({
            "dataframe": df,
            "feature_names": ["X1"],
            "treatment_col": "X2",
            "outcome_col": "X3",
        })
        assert "mac_discovery" in result
        assert "dag_edges" in result

    def test_execute_no_data(self):
        """데이터 없으면 None."""
        agent = MACDiscoveryAgent()
        result = agent.execute({})
        assert result["mac_discovery"] is None

    def test_name_attribute(self):
        """name 속성."""
        agent = MACDiscoveryAgent()
        assert agent.name == "MACDiscovery"
