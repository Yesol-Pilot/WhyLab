# -*- coding: utf-8 -*-
"""Test Cytoplasm Workflow."""

import unittest
from engine.workflow.graph import build_graph

class TestCytoplasm(unittest.TestCase):
    def test_basic_workflow(self):
        """기본 워크플로우(A->B->C->End) 실행 테스트"""
        app = build_graph()
        
        # 초기 상태 주입
        initial_state = {
            "scenario": "A",
            "data_summary": "Test Data",
            "history": []
        }
        
        # 워크플로우 실행
        print("\nStarting Cytoplasm Workflow Test...")
        final_state = app.invoke(initial_state)
        
        print(f"Workflow Finished. Final History: {final_state['history']}")
        
        # 검증
        self.assertIn("Discovery Completed", final_state["history"])
        self.assertIn("Estimation Completed", final_state["history"])
        self.assertIn("Refutation Completed", final_state["history"])
        self.assertTrue(final_state["refutation_result"])  # Mock은 항상 True

if __name__ == '__main__':
    unittest.main()
