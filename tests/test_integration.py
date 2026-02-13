# -*- coding: utf-8 -*-
"""Integration Test - E2E Pipeline Chain Verification.

DataCell -> CausalCell -> SensitivityCell -> ExportCell
전체 파이프라인이 에러 없이 동작하고, 올바른 JSON을 생성하는지 검증합니다.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from engine.config import WhyLabConfig
from engine.cells.data_cell import DataCell
from engine.cells.causal_cell import CausalCell
from engine.cells.sensitivity_cell import SensitivityCell
from engine.cells.export_cell import ExportCell


class TestDataCell(unittest.TestCase):
    """DataCell 단위 테스트."""

    def setUp(self):
        self.config = WhyLabConfig()
        self.cell = DataCell(self.config)

    def test_scenario_A_output_schema(self):
        """Scenario A: DataCell 출력 스키마 검증"""
        out = self.cell.execute({"scenario": "A"})

        required_keys = [
            "dataframe", "feature_names", "treatment_col",
            "outcome_col", "dag_edges", "scenario"
        ]
        for key in required_keys:
            self.assertIn(key, out, f"Missing key: {key}")

        self.assertIsInstance(out["dataframe"], pd.DataFrame)
        self.assertGreater(len(out["dataframe"]), 0)
        self.assertEqual(out["treatment_col"], "credit_limit")
        self.assertEqual(out["outcome_col"], "is_default")

    def test_scenario_B_binary_treatment(self):
        """Scenario B: 바이너리 treatment 검증"""
        out = self.cell.execute({"scenario": "B"})

        self.assertEqual(out["treatment_col"], "coupon_sent")
        self.assertEqual(out["outcome_col"], "is_joined")
        unique_values = set(out["dataframe"]["coupon_sent"].unique())
        self.assertTrue(unique_values.issubset({0, 1}))


class TestCausalCell(unittest.TestCase):
    """CausalCell 단위 테스트."""

    def setUp(self):
        self.config = WhyLabConfig()
        data_cell = DataCell(self.config)
        self.data_out = data_cell.execute({"scenario": "A"})

    def test_ate_output(self):
        """ATE 추정치가 올바른 형식인지 확인"""
        causal_cell = CausalCell(self.config)
        out = causal_cell.execute(self.data_out)

        # ATE는 float
        self.assertIn("ate", out)
        self.assertIsInstance(out["ate"], float)

        # 신뢰구간
        self.assertIn("ate_ci_lower", out)
        self.assertIn("ate_ci_upper", out)
        self.assertLessEqual(out["ate_ci_lower"], out["ate"])
        self.assertLessEqual(out["ate"], out["ate_ci_upper"])

    def test_cate_predictions(self):
        """CATE 예측값 배열이 올바른 형식인지 확인"""
        causal_cell = CausalCell(self.config)
        out = causal_cell.execute(self.data_out)

        self.assertIn("cate_predictions", out)
        cate = out["cate_predictions"]
        self.assertIsInstance(cate, np.ndarray)
        self.assertEqual(len(cate), len(self.data_out["dataframe"]))


class TestExportJSON(unittest.TestCase):
    """ExportCell JSON 스키마 검증."""

    def setUp(self):
        self.config = WhyLabConfig()
        # 임시 프로젝트 루트를 설정하여 dashboard_data_dir을 우회
        self.temp_dir = tempfile.mkdtemp()
        self.config.paths.project_root = Path(self.temp_dir)
        # dashboard/public/data 디렉토리 생성 (dashboard_data_dir property가 참조)
        (Path(self.temp_dir) / "dashboard" / "public" / "data").mkdir(parents=True)
        # paper/data 디렉토리도 생성 (CSV 백업용)
        (Path(self.temp_dir) / "paper" / "data").mkdir(parents=True)

    def test_json_schema_matches_typescript(self):
        """생성된 JSON이 TypeScript CausalAnalysisResult 타입과 일치하는지 검증"""
        # Pipeline 실행
        data_cell = DataCell(self.config)
        causal_cell = CausalCell(self.config)
        sensitivity_cell = SensitivityCell(self.config)
        export_cell = ExportCell(self.config)

        ctx = data_cell.execute({"scenario": "A"})
        ctx.update(causal_cell.execute(ctx))
        ctx.update(sensitivity_cell.execute(ctx))
        result = export_cell.execute(ctx)

        # JSON 파일 로드
        json_path = result["json_path"]
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # TypeScript CausalAnalysisResult 타입 검증
        # 1. ate
        ate = data["ate"]
        self.assertIn("value", ate)
        self.assertIn("ci_lower", ate)
        self.assertIn("ci_upper", ate)
        self.assertIn("alpha", ate)
        self.assertIn("description", ate)

        # 2. cate_distribution
        cate = data["cate_distribution"]
        self.assertIn("mean", cate)
        self.assertIn("std", cate)
        self.assertIn("min", cate)
        self.assertIn("max", cate)
        self.assertIn("histogram", cate)
        self.assertIn("bin_edges", cate["histogram"])
        self.assertIn("counts", cate["histogram"])

        # 3. segments
        self.assertIsInstance(data["segments"], list)
        if len(data["segments"]) > 0:
            seg = data["segments"][0]
            self.assertIn("name", seg)
            self.assertIn("dimension", seg)
            self.assertIn("n", seg)
            self.assertIn("cate_mean", seg)
            self.assertIn("cate_ci_lower", seg)
            self.assertIn("cate_ci_upper", seg)

        # 4. dag
        dag = data["dag"]
        self.assertIn("nodes", dag)
        self.assertIn("edges", dag)
        if len(dag["nodes"]) > 0:
            node = dag["nodes"][0]
            self.assertIn("id", node)
            self.assertIn("label", node)
            self.assertIn("role", node)

        # 5. metadata
        meta = data["metadata"]
        self.assertIn("generated_at", meta)
        self.assertIn("scenario", meta)
        self.assertIn("model_type", meta)
        self.assertIn("n_samples", meta)
        self.assertIn("feature_names", meta)
        self.assertIn("treatment_col", meta)
        self.assertIn("outcome_col", meta)

        # 6. sensitivity
        sens = data["sensitivity"]
        self.assertIn("status", sens)
        self.assertIn("placebo_test", sens)
        self.assertIn("random_common_cause", sens)

        # 7. 불필요 키 없음 확인 (meta, cate 중복 키)
        self.assertNotIn("meta", data, "중복 키 'meta' 가 여전히 존재")

    def test_no_meta_or_cate_duplicates(self):
        """'meta'와 'cate' 중복 키가 제거되었는지 확인"""
        data_cell = DataCell(self.config)
        export_cell = ExportCell(self.config)
        causal_cell = CausalCell(self.config)

        ctx = data_cell.execute({"scenario": "A"})
        ctx.update(causal_cell.execute(ctx))
        result = export_cell.execute(ctx)

        with open(result["json_path"], "r", encoding="utf-8") as f:
            data = json.load(f)

        # 'meta' 키는 'metadata'로 통합되어야 함
        self.assertNotIn("meta", data)
        # 'cate' 키는 'cate_distribution'으로 통합되어야 함
        self.assertNotIn("cate", data)


if __name__ == "__main__":
    unittest.main()
