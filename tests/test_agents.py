# -*- coding: utf-8 -*-
"""Test Agents & Diagnostics — 에이전트 + 통계 진단 테스트."""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from engine.config import WhyLabConfig
from engine.agents.discovery import DiscoveryAgent
from engine.cells.sensitivity_cell import SensitivityCell


class TestDiscoveryAgent(unittest.TestCase):
    def setUp(self):
        self.config = WhyLabConfig()
        self.agent = DiscoveryAgent(self.config)
        
        # 더미 데이터 생성 (A -> B -> C)
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            'A': np.random.normal(0, 1, n),
        })
        self.df['B'] = self.df['A'] * 0.5 + np.random.normal(0, 0.1, n)
        self.df['C'] = self.df['B'] * 0.5 + np.random.normal(0, 0.1, n)
        
        self.metadata = {
            "feature_names": ["A", "B"],
            "treatment_col": "B",
            "outcome_col": "C"
        }

    def test_discovery_run(self):
        """DiscoveryAgent가 에러 없이 DAG를 반환하는지 테스트"""
        dag = self.agent.discover(self.df, self.metadata)
        
        # DAG 객체 반환 확인
        self.assertIsNotNone(dag)
        # 노드 존재 확인
        self.assertIn("A", dag.nodes)
        self.assertIn("B", dag.nodes)
        self.assertIn("C", dag.nodes)

    def test_auto_discover(self):
        """auto_discover가 treatment/outcome/confounders를 반환하는지 테스트"""
        roles = self.agent.auto_discover(self.df)
        
        self.assertIn("treatment", roles)
        self.assertIn("outcome", roles)
        self.assertIn("confounders", roles)
        self.assertIn("dag", roles)
        # treatment과 outcome이 실제 컬럼명인지 확인
        all_cols = self.df.columns.tolist()
        self.assertIn(roles["treatment"], all_cols)
        self.assertIn(roles["outcome"], all_cols)

    def test_heuristic_fallback(self):
        """LLM 없이 휴리스틱 역할 탐색이 작동하는지 테스트"""
        roles = self.agent._discover_roles_heuristic(self.df)
        
        self.assertIn("treatment", roles)
        self.assertIn("outcome", roles)
        self.assertIn("reasoning", roles)


class TestSensitivityDiagnostics(unittest.TestCase):
    """Phase 4 통계 진단 (E-value, Overlap, GATES) 테스트."""

    def setUp(self):
        self.config = WhyLabConfig()
        self.cell = SensitivityCell(self.config)
        np.random.seed(42)

    def test_e_value_computation(self):
        """E-value 계산 정확성 테스트"""
        e = self.cell._compute_e_value(0.035)
        # E-value > 1 (어떤 ATE든 E-value > 1)
        self.assertGreater(e, 1.0)
        # ATE가 클수록 E-value도 커야 함
        e_large = self.cell._compute_e_value(0.5)
        self.assertGreater(e_large, e)

    def test_overlap_diagnosis(self):
        """Overlap 진단이 올바른 구조를 반환하는지 테스트"""
        n = 500
        df = pd.DataFrame({
            "X1": np.random.normal(0, 1, n),
            "X2": np.random.normal(0, 1, n),
            "T": np.random.binomial(1, 0.5, n),
        })

        result = self.cell._diagnose_overlap(df, "T", ["X1", "X2"])
        
        self.assertIn("overlap_score", result)
        self.assertIn("ps_stats", result)
        self.assertIn("status", result)
        self.assertGreater(result["overlap_score"], 0)
        self.assertLessEqual(result["overlap_score"], 1.0)
        # 무작위 처치이므로 Overlap이 높아야 함
        self.assertEqual(result["status"], "Pass")

    def test_gates_clan(self):
        """GATES/CLAN 분석이 올바른 구조를 반환하는지 테스트"""
        n = 400
        cate_preds = np.random.normal(0, 0.05, n)
        df = pd.DataFrame({
            "X1": np.random.normal(0, 1, n),
            "X2": np.random.normal(0, 1, n),
        })

        result = self.cell._compute_gates_clan(df, cate_preds, ["X1", "X2"], 4)
        
        self.assertIn("n_groups", result)
        self.assertIn("groups", result)
        self.assertIn("f_statistic", result)
        self.assertEqual(result["n_groups"], 4)
        self.assertEqual(len(result["groups"]), 4)
        # 각 그룹이 올바른 필드를 가지는지
        for g in result["groups"]:
            self.assertIn("mean_cate", g)
            self.assertIn("ci_lower", g)
            self.assertIn("ci_upper", g)
            self.assertIn("clan_features", g)


if __name__ == '__main__':
    unittest.main()

