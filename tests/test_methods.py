# -*- coding: utf-8 -*-
"""인과 추론 메서드 모듈 테스트.

base, lightweight, causal_impact, gsc, dml, blame_attribution
모든 메서드의 기본 동작을 검증합니다.
"""

import statistics
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from engine.audit.methods.base import AnalysisResult, BaseMethod
from engine.audit.methods.lightweight import LightweightMethod
from engine.audit.methods.causal_impact_method import CausalImpactMethod
from engine.audit.methods.gsc_method import GSCMethod
from engine.audit.methods.dml_method import DMLMethod
from engine.audit.methods.blame_attribution import BlameAttributionMethod


# ── 헬퍼 ──

def _make_series(base: float, n: int, noise: float = 0.05, seed: int = 42) -> List[float]:
    import random
    random.seed(seed)
    return [round(base + random.gauss(0, base * noise), 2) for _ in range(n)]


# ── Base ──

class TestBase:
    def test_analysis_result_significance(self):
        r = AnalysisResult(method="test", p_value=0.01)
        assert r.is_significant

    def test_analysis_result_not_significant(self):
        r = AnalysisResult(method="test", p_value=0.1)
        assert not r.is_significant

    def test_base_method_not_implemented(self):
        m = BaseMethod()
        with pytest.raises(NotImplementedError):
            m.analyze([], [])


# ── Lightweight ──

class TestLightweight:
    def test_clear_effect(self):
        pre = _make_series(100, 14)
        post = _make_series(150, 7)
        result = LightweightMethod().analyze(pre, post)
        assert result.ate > 20
        assert result.method == "lightweight_t_test"

    def test_no_effect(self):
        pre = _make_series(100, 14)
        post = _make_series(100, 7, seed=99)
        result = LightweightMethod().analyze(pre, post)
        assert abs(result.ate) < 20

    def test_has_diagnostics(self):
        result = LightweightMethod().analyze(_make_series(100, 10), _make_series(120, 5))
        assert "t_statistic" in result.diagnostics
        assert "n_pre" in result.diagnostics


# ── CausalImpact ──

class TestCausalImpact:
    def test_insufficient_data_fallback(self):
        pre = _make_series(100, 10)  # < 21일
        post = _make_series(120, 5)
        result = CausalImpactMethod().analyze(pre, post)
        assert result.diagnostics.get("fallback_from") == "causal_impact"

    def test_data_validation(self):
        m = CausalImpactMethod()
        assert not m._validate_data(_make_series(100, 10), _make_series(100, 5))
        assert m._validate_data(_make_series(100, 21), _make_series(100, 7))


# ── GSC ──

class TestGSC:
    def test_self_synthetic(self):
        pre = _make_series(100, 14)
        post = _make_series(130, 7)
        result = GSCMethod(n_bootstrap=50).analyze(pre, post)
        assert result.method == "gsc"
        assert result.ate != 0
        assert "n_factors" in result.diagnostics

    def test_with_donors(self):
        pre = _make_series(100, 14)
        post = _make_series(130, 7)
        donors = [
            _make_series(95, 21, seed=i) for i in range(3)
        ]
        result = GSCMethod(n_bootstrap=50).analyze(pre, post, donor_pool=donors)
        assert result.diagnostics["has_donor_pool"]
        assert result.diagnostics["n_donors"] == 3

    def test_placebo(self):
        m = GSCMethod()
        pre = _make_series(100, 14)
        counterfactual_pre = _make_series(100, 14, seed=99)
        assert m._placebo_test(pre, counterfactual_pre)


# ── DML ──

class TestDML:
    def test_single_treatment_fallback(self):
        pre = _make_series(100, 14)
        post = _make_series(120, 7)
        result = DMLMethod().analyze(pre, post)  # no treatments
        assert "fallback_from" in result.diagnostics

    def test_lightweight_multi(self):
        pre = _make_series(100, 14)
        post = _make_series(130, 7)
        treatments = [
            _make_series(0, 21, seed=1),
            _make_series(0, 21, seed=2),
        ]
        result = DMLMethod().analyze(pre, post, treatments=treatments)
        assert "n_treatments" in result.diagnostics
        assert result.diagnostics["n_treatments"] == 2


# ── Blame Attribution ──

class TestBlameAttribution:
    def test_single_agent(self):
        pre = _make_series(100, 14)
        post = _make_series(130, 7)
        result = BlameAttributionMethod().analyze(
            pre, post,
            agent_decisions={"agent_A": {"treatment_value": 1.0}}
        )
        assert result.diagnostics["blame_scores"]["agent_A"] == 1.0

    def test_multi_agent_shapley(self):
        pre = _make_series(100, 14)
        post = _make_series(130, 7)
        result = BlameAttributionMethod().analyze(
            pre, post,
            agent_decisions={
                "hive_mind": {"treatment_value": 2.0, "expected_effect": "positive"},
                "cro_agent": {"treatment_value": 1.0, "expected_effect": "positive"},
            }
        )
        scores = result.diagnostics["blame_scores"]
        assert len(scores) == 2
        assert "hive_mind" in scores
        assert "cro_agent" in scores
        # hive_mind이 더 큰 treatment_value이므로 더 큰 책임
        assert abs(scores["hive_mind"]) >= abs(scores["cro_agent"])

    def test_synergy(self):
        pre = _make_series(100, 14)
        post = _make_series(130, 7)
        result = BlameAttributionMethod().analyze(
            pre, post,
            agent_decisions={
                "A": {"treatment_value": 1.0},
                "B": {"treatment_value": 1.0},
                "C": {"treatment_value": 1.0},
            }
        )
        assert "synergy" in result.diagnostics
