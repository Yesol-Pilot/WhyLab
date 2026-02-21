# -*- coding: utf-8 -*-
"""DoseResponseCell 테스트."""

import pytest
import numpy as np
import pandas as pd

from engine.cells.dose_response_cell import (
    DoseResponseCell,
    DoseResponseConfig,
    estimate_gps_gaussian,
    kernel_dose_response,
)


@pytest.fixture
def synthetic_continuous():
    """연속 처치 합성 데이터 (비선형 용량-반응)."""
    np.random.seed(42)
    n = 500

    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    X = np.column_stack([x1, x2])

    # 연속 처치: T = 0.5*x1 + noise
    T = 0.5 * x1 + np.random.normal(0, 0.5, n)

    # 비선형 용량-반응: Y = 2*T - 0.5*T^2 + x2 + noise
    # 최적 용량: dY/dT = 2 - T = 0 → T* = 2
    Y = 2 * T - 0.5 * T ** 2 + 0.3 * x2 + np.random.normal(0, 0.5, n)

    return X, T, Y


@pytest.fixture
def synthetic_df(synthetic_continuous):
    """데이터프레임 형식."""
    X, T, Y = synthetic_continuous
    return pd.DataFrame({
        "x1": X[:, 0],
        "x2": X[:, 1],
        "dose": T,
        "outcome": Y,
    })


# ──────────────────────────────────────────────
# GPS 추정
# ──────────────────────────────────────────────

class TestGPS:

    def test_gps_gaussian(self, synthetic_continuous):
        """가우시안 GPS가 유효한 밀도를 반환."""
        X, T, _ = synthetic_continuous
        gps, sigma, residuals = estimate_gps_gaussian(T, X)
        assert len(gps) == len(T)
        assert sigma > 0
        assert np.all(gps >= 0)
        assert np.all(np.isfinite(gps))

    def test_gps_captures_confounding(self, synthetic_continuous):
        """GPS가 교란(x1 → T) 구조를 포착."""
        X, T, _ = synthetic_continuous
        _, _, residuals = estimate_gps_gaussian(T, X)
        # 잔차 분산 < 원래 T 분산 (공변량으로 일부 설명)
        assert np.var(residuals) < np.var(T)


# ──────────────────────────────────────────────
# 커널 용량-반응
# ──────────────────────────────────────────────

class TestKernelDoseResponse:

    def test_curve_shape(self, synthetic_continuous):
        """커널 곡선이 비선형 형태를 포착."""
        X, T, Y = synthetic_continuous
        t_grid = np.linspace(np.quantile(T, 0.05), np.quantile(T, 0.95), 30)
        curve = kernel_dose_response(T, Y, X, t_grid)
        assert len(curve) == 30
        assert not np.all(np.isnan(curve))

    def test_curve_not_flat(self, synthetic_continuous):
        """곡선이 평평하지 않음 (처치 효과 존재)."""
        X, T, Y = synthetic_continuous
        t_grid = np.linspace(np.quantile(T, 0.05), np.quantile(T, 0.95), 30)
        curve = kernel_dose_response(T, Y, X, t_grid)
        assert np.nanstd(curve) > 0.1  # 곡선에 변동 있음


# ──────────────────────────────────────────────
# DoseResponseCell
# ──────────────────────────────────────────────

class TestDoseResponseCell:

    def test_estimate_basic(self, synthetic_continuous):
        """기본 추정이 올바른 구조 반환."""
        X, T, Y = synthetic_continuous
        cfg = DoseResponseConfig(n_bootstrap=20, n_grid_points=30)
        cell = DoseResponseCell(dr_config=cfg)
        result = cell.estimate(X, T, Y)

        assert "t_grid" in result
        assert "dr_curve" in result
        assert "marginal_effect" in result
        assert "optimal_dose" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert len(result["t_grid"]) == 30
        assert len(result["dr_curve"]) == 30
        assert len(result["marginal_effect"]) == 30

    def test_optimal_dose_direction(self, synthetic_continuous):
        """최적 용량이 합리적인 범위."""
        X, T, Y = synthetic_continuous
        cfg = DoseResponseConfig(n_bootstrap=10, n_grid_points=50)
        cell = DoseResponseCell(dr_config=cfg)
        result = cell.estimate(X, T, Y)

        # 이론적 최적: T* ≈ 2. 커널 추정이므로 정확하진 않지만 양수여야 함
        assert result["optimal_dose"] > 0
        assert result["has_effect"] is True

    def test_ci_bracket(self, synthetic_continuous):
        """신뢰구간이 곡선을 포함."""
        X, T, Y = synthetic_continuous
        cfg = DoseResponseConfig(n_bootstrap=30, n_grid_points=20)
        cell = DoseResponseCell(dr_config=cfg)
        result = cell.estimate(X, T, Y)

        ci_low = np.array(result["ci_lower"])
        ci_up = np.array(result["ci_upper"])
        curve = np.array(result["dr_curve"])

        valid = ~np.isnan(curve) & ~np.isnan(ci_low) & ~np.isnan(ci_up)
        # 대부분 곡선이 CI 안에 있어야 함
        inside = np.sum((curve[valid] >= ci_low[valid]) & (curve[valid] <= ci_up[valid]))
        assert inside / np.sum(valid) > 0.5

    def test_polynomial_method(self, synthetic_continuous):
        """다항 결과 표면 방법 동작."""
        X, T, Y = synthetic_continuous
        cfg = DoseResponseConfig(
            outcome_method="polynomial",
            polynomial_degree=3,
            bootstrap_ci=False,
            n_grid_points=20,
        )
        cell = DoseResponseCell(dr_config=cfg)
        result = cell.estimate(X, T, Y)
        assert len(result["dr_curve"]) == 20
        assert result["has_effect"] is True


# ──────────────────────────────────────────────
# 파이프라인 인터페이스
# ──────────────────────────────────────────────

class TestCellInterface:

    def test_execute_continuous(self, synthetic_df):
        """execute가 연속 처치에서 정상 동작."""
        cfg = DoseResponseConfig(n_bootstrap=10, n_grid_points=20)
        cell = DoseResponseCell(dr_config=cfg)
        result = cell.execute({
            "dataframe": synthetic_df,
            "treatment_col": "dose",
            "outcome_col": "outcome",
            "feature_names": ["x1", "x2"],
        })
        assert "dose_response" in result
        dr = result["dose_response"]
        assert "t_grid" in dr
        assert dr["has_effect"] is True

    def test_execute_binary_skip(self):
        """이진 처치는 건너뜀."""
        np.random.seed(0)
        df = pd.DataFrame({
            "t": np.random.binomial(1, 0.5, 100),
            "y": np.random.normal(0, 1, 100),
            "x": np.random.normal(0, 1, 100),
        })
        cell = DoseResponseCell()
        result = cell.execute({
            "dataframe": df,
            "treatment_col": "t",
            "outcome_col": "y",
            "feature_names": ["x"],
        })
        assert result["dose_response"]["skipped"] is True

    def test_execute_no_df(self):
        """데이터프레임 없으면 None."""
        cell = DoseResponseCell()
        result = cell.execute({})
        assert result["dose_response"] is None

    def test_name_attribute(self):
        """name 속성 존재."""
        cell = DoseResponseCell()
        assert cell.name == "DoseResponse"
