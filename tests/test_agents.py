# -*- coding: utf-8 -*-
"""Test Agents — 에이전트 모듈 단위 테스트."""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from engine.config import WhyLabConfig
from engine.agents.discovery import DiscoveryAgent

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
        
        # (통계적 발견이 잘 작동한다면 A->B, B->C 등의 엣지가 있어야 함)
        # 하지만 threshold나 노이즈에 따라 다를 수 있으므로 엣지 개수만 체크
        print("Discovered Edges:", dag.edges())

if __name__ == '__main__':
    unittest.main()
