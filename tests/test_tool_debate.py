# -*- coding: utf-8 -*-
"""ToolAugmentedDebate 테스트."""

import pytest
import numpy as np

from engine.agents.tool_debate import (
    ToolAugmentedDebate,
    Tool,
    tool_cate_variance,
    tool_effect_size_check,
    tool_placebo_refutation,
    tool_overlap_check,
)
from engine.agents.dav_protocol import DaVClaim, Evidence


@pytest.fixture
def verified_context():
    """검증되어야 할 강한 인과 컨텍스트."""
    return {
        "treatment_col": "T",
        "outcome_col": "Y",
        "ate": {"point_estimate": 2.5, "ci_lower": 1.0, "ci_upper": 4.0},
        "sensitivity": {"e_value": {"point": 3.5}},
        "refutation": {"placebo": {"passed": True, "p_value": 0.02}},
        "meta_learners": {
            "S": {"ate": 2.3}, "T": {"ate": 2.7}, "X": {"ate": 2.4},
        },
        "dag_edges": [["T", "Y"]],
        "outcome_std": 1.0,
    }


@pytest.fixture
def refuted_context():
    """기각되어야 할 약한 인과 컨텍스트."""
    return {
        "treatment_col": "T",
        "outcome_col": "Y",
        "ate": {"point_estimate": 0.01, "ci_lower": -0.5, "ci_upper": 0.52},
        "sensitivity": {"e_value": {"point": 1.0}},
        "refutation": {"placebo": {"passed": False, "p_value": 0.8}},
        "meta_learners": {
            "S": {"ate": 0.01}, "T": {"ate": -0.3}, "X": {"ate": 0.5},
        },
        "dag_edges": [],
        "outcome_std": 5.0,
    }


# ──────────────────────────────────────────────
# 개별 도구 테스트
# ──────────────────────────────────────────────

class TestTools:

    def test_cate_variance_consistent(self, verified_context):
        """일관된 CATE → supports."""
        claim = DaVClaim(statement="test", treatment="T", outcome="Y", ate=2.5)
        ev = tool_cate_variance(verified_context, claim)
        assert ev.direction == "supports"
        assert ev.strength > 0.5

    def test_cate_variance_inconsistent(self, refuted_context):
        """불일치 CATE → contradicts."""
        claim = DaVClaim(statement="test", treatment="T", outcome="Y", ate=0.01)
        ev = tool_cate_variance(refuted_context, claim)
        assert ev.direction == "contradicts"

    def test_effect_size_large(self, verified_context):
        """큰 효과 크기 → supports."""
        claim = DaVClaim(statement="test", treatment="T", outcome="Y", ate=2.5)
        ev = tool_effect_size_check(verified_context, claim)
        assert ev.direction == "supports"

    def test_effect_size_negligible(self, refuted_context):
        """무시할 효과 크기 → contradicts."""
        claim = DaVClaim(statement="test", treatment="T", outcome="Y", ate=0.01)
        ev = tool_effect_size_check(refuted_context, claim)
        assert ev.direction == "contradicts"

    def test_placebo_pass(self, verified_context):
        """위약 통과 → supports."""
        claim = DaVClaim(statement="test", treatment="T", outcome="Y")
        ev = tool_placebo_refutation(verified_context, claim)
        assert ev.direction == "supports"

    def test_placebo_fail(self, refuted_context):
        """위약 실패 → contradicts."""
        claim = DaVClaim(statement="test", treatment="T", outcome="Y")
        ev = tool_placebo_refutation(refuted_context, claim)
        assert ev.direction == "contradicts"

    def test_overlap_good(self):
        """좋은 겹침 → supports."""
        ctx = {"propensity_scores": np.random.beta(2, 2, 200)}
        claim = DaVClaim(statement="test", treatment="T", outcome="Y")
        ev = tool_overlap_check(ctx, claim)
        assert ev.direction == "supports"

    def test_overlap_missing(self):
        """성향점수 없으면 neutral."""
        claim = DaVClaim(statement="test", treatment="T", outcome="Y")
        ev = tool_overlap_check({}, claim)
        assert ev.direction == "neutral"


# ──────────────────────────────────────────────
# ToolAugmentedDebate 통합
# ──────────────────────────────────────────────

class TestToolAugmentedDebate:

    def test_verified_verdict(self, verified_context):
        """강한 컨텍스트 → VERIFIED."""
        debate = ToolAugmentedDebate(n_rounds=1)
        verdict = debate.verify(verified_context)
        assert verdict.verdict == "VERIFIED"
        assert verdict.confidence > 0.5

    def test_refuted_verdict(self, refuted_context):
        """약한 컨텍스트 → REFUTED."""
        debate = ToolAugmentedDebate(n_rounds=1)
        verdict = debate.verify(refuted_context)
        assert verdict.verdict == "REFUTED"

    def test_tool_log(self, verified_context):
        """도구 호출 로그 기록."""
        debate = ToolAugmentedDebate(n_rounds=1)
        debate.verify(verified_context)
        log = debate.get_tool_log()
        assert len(log) > 0
        assert all("tool" in entry for entry in log)
        assert all("agent" in entry for entry in log)

    def test_more_evidence_than_base(self, verified_context):
        """도구 강화 → 기본 DaV보다 증거 많음."""
        from engine.agents.dav_protocol import DaVProtocol
        base = DaVProtocol()
        base_verdict = base.verify(verified_context)

        debate = ToolAugmentedDebate(n_rounds=2)
        aug_verdict = debate.verify(verified_context)

        assert len(aug_verdict.evidence_chain) > len(base_verdict.evidence_chain)

    def test_custom_tools(self, verified_context):
        """커스텀 도구 사용."""
        custom = Tool(
            name="custom_check",
            description="커스텀 검증",
            role="both",
            execute=lambda ctx, claim: Evidence(
                source="custom",
                claim="커스텀",
                direction="supports",
                strength=0.9,
            ),
        )
        debate = ToolAugmentedDebate(tools=[custom], n_rounds=1)
        verdict = debate.verify(verified_context)
        log = debate.get_tool_log()
        assert any(e["tool"] == "custom_check" for e in log)

    def test_inherits_dav(self):
        """DaVProtocol 상속 확인."""
        from engine.agents.dav_protocol import DaVProtocol
        debate = ToolAugmentedDebate()
        assert isinstance(debate, DaVProtocol)
