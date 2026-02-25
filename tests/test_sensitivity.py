# -*- coding: utf-8 -*-
"""R2 테스트 — 민감도 분석 (E-value + Partial R²)."""

import pytest

from engine.audit.sensitivity import SensitivityAnalyzer, SensitivityResult


class TestEValue:
    """E-value 계산 검증."""

    def setup_method(self):
        self.analyzer = SensitivityAnalyzer()
        self.pre = [100 + i * 0.5 for i in range(14)]

    def test_strong_effect_high_evalue(self):
        """큰 효과 → 높은 E-value (강건)."""
        post = [v + 30 for v in self.pre[:7]]
        result = self.analyzer.analyze(
            ate=30.0, ate_ci=[20.0, 40.0],
            pre_values=self.pre, post_values=post,
        )
        assert result.e_value >= 2.0
        assert result.is_robust
        assert result.robustness_level in ("strong", "very_strong")

    def test_weak_effect_low_evalue(self):
        """작은 효과 → 낮은 E-value (약함)."""
        post = [v + 0.1 for v in self.pre[:7]]
        result = self.analyzer.analyze(
            ate=0.1, ate_ci=[-0.5, 0.7],
            pre_values=self.pre, post_values=post,
        )
        # CI가 0 포함 → 강건하지 않음
        assert result.e_value_ci_lower <= 1.5

    def test_negative_ate(self):
        """음의 ATE도 정상 처리."""
        post = [v - 20 for v in self.pre[:7]]
        result = self.analyzer.analyze(
            ate=-20.0, ate_ci=[-30.0, -10.0],
            pre_values=self.pre, post_values=post,
        )
        assert result.e_value >= 1.5
        assert result.risk_ratio > 1.0

    def test_evalue_formula(self):
        """E-value 공식: E = RR + sqrt(RR*(RR-1))."""
        # RR=2 → E = 2 + sqrt(2*1) = 3.414
        e = self.analyzer._compute_e_value(2.0)
        assert abs(e - 3.414) < 0.01

    def test_evalue_rr_one(self):
        """RR ≤ 1 → E-value = 1 (효과 없음)."""
        e = self.analyzer._compute_e_value(1.0)
        assert e == 1.0
        e = self.analyzer._compute_e_value(0.5)
        assert e == 1.0


class TestPartialR2:
    """Partial R² 경계 검증."""

    def setup_method(self):
        self.analyzer = SensitivityAnalyzer()

    def test_rv_q_positive(self):
        """유의한 효과 → RV_q > 0."""
        pre = [100 + i for i in range(14)]
        post = [v + 25 for v in pre[:7]]
        result = self.analyzer.analyze(
            ate=25.0, ate_ci=[15.0, 35.0],
            pre_values=pre, post_values=post,
        )
        assert result.rv_q > 0
        assert result.partial_r2_treatment > 0

    def test_interpretation_generated(self):
        """자연어 해석이 생성됨."""
        pre = [100] * 14
        pre[-1] = 101  # stdev > 0 보장
        post = [120] * 7
        result = self.analyzer.analyze(
            ate=20.0, ate_ci=[10.0, 30.0],
            pre_values=pre, post_values=post,
        )
        interp = result.diagnostics.get("interpretation", "")
        assert "교란 변수" in interp or "Risk Ratio" in interp


class TestSensitivityResult:
    """SensitivityResult 구조 검증."""

    def test_default_values(self):
        r = SensitivityResult()
        assert r.e_value == 0.0
        assert r.robustness_level == "unknown"
        assert not r.is_robust

    def test_robustness_classification(self):
        analyzer = SensitivityAnalyzer()
        assert analyzer._classify_robustness(3.5) == "very_strong"
        assert analyzer._classify_robustness(2.5) == "strong"
        assert analyzer._classify_robustness(1.7) == "moderate"
        assert analyzer._classify_robustness(1.2) == "weak"
        assert analyzer._classify_robustness(0.8) == "not_robust"
