# -*- coding: utf-8 -*-
"""FairnessAuditCell 테스트."""

import pytest
import numpy as np
import pandas as pd

from engine.cells.fairness_audit_cell import (
    FairnessAuditCell,
    FairnessConfig,
    compute_subgroup_metrics,
    compute_causal_parity_gap,
    compute_disparate_impact_ratio,
    compute_equalized_cate_score,
    compute_counterfactual_fairness_index,
    generate_fairness_report,
)


@pytest.fixture
def fair_data():
    """공정한 데이터: 성별 간 CATE 차이 없음."""
    np.random.seed(42)
    n = 400
    gender = np.array([0, 1] * (n // 2))
    cate = np.random.normal(0.5, 0.2, n)  # 동일 분포
    df = pd.DataFrame({
        "gender": gender,
        "age_group": np.random.choice([0, 1, 2], n),  # 3그룹
        "x1": np.random.normal(0, 1, n),
    })
    return cate, df


@pytest.fixture
def unfair_data():
    """불공정 데이터: 그룹 0에 높은 CATE, 그룹 1에 낮은 CATE."""
    np.random.seed(42)
    n = 400
    group = np.array([0] * (n // 2) + [1] * (n // 2))
    cate = np.where(group == 0,
                    np.random.normal(0.8, 0.1, n),    # 그룹 0: 높음
                    np.random.normal(0.1, 0.1, n))     # 그룹 1: 낮음
    df = pd.DataFrame({
        "protected": group,
        "x1": np.random.normal(0, 1, n),
    })
    return cate, df


# ──────────────────────────────────────────────
# 서브그룹 지표
# ──────────────────────────────────────────────

class TestSubgroupMetrics:

    def test_basic(self, fair_data):
        """서브그룹 지표 기본 동작."""
        cate, df = fair_data
        metrics = compute_subgroup_metrics(cate, df["gender"].values)
        assert len(metrics) == 2
        assert all(m.n_samples > 0 for m in metrics)
        assert all(m.positive_ratio >= 0 for m in metrics)

    def test_unfair_groups(self, unfair_data):
        """불공정 그룹에서 평균 CATE 차이 감지."""
        cate, df = unfair_data
        metrics = compute_subgroup_metrics(cate, df["protected"].values)
        means = [m.mean_cate for m in metrics]
        assert abs(means[0] - means[1]) > 0.5  # 큰 격차


# ──────────────────────────────────────────────
# 공정성 지표
# ──────────────────────────────────────────────

class TestFairnessMetrics:

    def test_causal_parity_fair(self, fair_data):
        """공정 데이터에서 Causal Parity 격차 작음."""
        cate, df = fair_data
        sg = compute_subgroup_metrics(cate, df["gender"].values)
        gap = compute_causal_parity_gap(sg)
        assert gap < 0.1  # 공정

    def test_causal_parity_unfair(self, unfair_data):
        """불공정 데이터에서 Causal Parity 격차 큼."""
        cate, df = unfair_data
        sg = compute_subgroup_metrics(cate, df["protected"].values)
        gap = compute_causal_parity_gap(sg)
        assert gap > 0.5  # 불공정

    def test_dir_fair(self, fair_data):
        """공정 데이터에서 DIR ≈ 1."""
        cate, df = fair_data
        sg = compute_subgroup_metrics(cate, df["gender"].values)
        dir_ratio = compute_disparate_impact_ratio(sg)
        assert dir_ratio > 0.8

    def test_equalized_cate(self, fair_data):
        """공정 데이터에서 Equalized CATE 점수 높음."""
        cate, df = fair_data
        sg = compute_subgroup_metrics(cate, df["gender"].values)
        score = compute_equalized_cate_score(sg)
        assert score > 0.8

    def test_cf_index_fair(self, fair_data):
        """공정 데이터에서 반사실 공정성 높음."""
        cate, df = fair_data
        sg = compute_subgroup_metrics(cate, df["gender"].values)
        idx = compute_counterfactual_fairness_index(sg)
        assert idx > 0.7

    def test_cf_index_unfair(self, unfair_data):
        """불공정 데이터에서 반사실 공정성 낮음."""
        cate, df = unfair_data
        sg = compute_subgroup_metrics(cate, df["protected"].values)
        idx = compute_counterfactual_fairness_index(sg)
        assert idx < 0.5


# ──────────────────────────────────────────────
# FairnessAuditCell
# ──────────────────────────────────────────────

class TestFairnessAuditCell:

    def test_audit_fair(self, fair_data):
        """공정 데이터 감사 통과."""
        cate, df = fair_data
        cfg = FairnessConfig(sensitive_attrs=["gender"])
        cell = FairnessAuditCell(fairness_config=cfg)
        results = cell.audit(cate, df, ["gender"])
        assert len(results) == 1
        assert results[0].is_fair is True
        assert len(results[0].violations) == 0

    def test_audit_unfair(self, unfair_data):
        """불공정 데이터 감사 위반 감지."""
        cate, df = unfair_data
        cfg = FairnessConfig(sensitive_attrs=["protected"])
        cell = FairnessAuditCell(fairness_config=cfg)
        results = cell.audit(cate, df, ["protected"])
        assert len(results) == 1
        assert results[0].is_fair is False
        assert len(results[0].violations) > 0

    def test_missing_attr(self, fair_data):
        """존재하지 않는 속성 무시."""
        cate, df = fair_data
        cell = FairnessAuditCell()
        results = cell.audit(cate, df, ["nonexistent"])
        assert len(results) == 0

    def test_report_generation(self, unfair_data):
        """보고서 생성."""
        cate, df = unfair_data
        cell = FairnessAuditCell()
        results = cell.audit(cate, df, ["protected"])
        report = generate_fairness_report(results)
        assert "인과 공정성 감사 보고서" in report
        assert "위반" in report


# ──────────────────────────────────────────────
# 파이프라인 인터페이스
# ──────────────────────────────────────────────

class TestCellInterface:

    def test_execute_with_cate(self, fair_data):
        """execute에 cate 직접 전달."""
        cate, df = fair_data
        cfg = FairnessConfig(sensitive_attrs=["gender"])
        cell = FairnessAuditCell(fairness_config=cfg)
        result = cell.execute({
            "dataframe": df,
            "cate": cate,
            "treatment_col": "x1",
            "outcome_col": "x1",
        })
        assert "fairness_audit" in result
        audit = result["fairness_audit"]
        assert audit["overall_fair"] is True

    def test_execute_auto_detect(self, fair_data):
        """민감 속성 자동 탐지."""
        cate, df = fair_data
        cell = FairnessAuditCell(fairness_config=FairnessConfig())
        result = cell.execute({
            "dataframe": df,
            "cate": cate,
            "treatment_col": "x1",
            "outcome_col": "x1",
        })
        assert result["fairness_audit"] is not None

    def test_execute_no_data(self):
        """데이터 없으면 None."""
        cell = FairnessAuditCell()
        result = cell.execute({})
        assert result["fairness_audit"] is None

    def test_name_attribute(self):
        """name 속성."""
        cell = FairnessAuditCell()
        assert cell.name == "FairnessAudit"
