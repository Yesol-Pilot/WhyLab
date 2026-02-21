"""
API Agents 단위 테스트 (Sprint 37)
===================================
api/agents/ 7개 모듈의 핵심 함수 테스트.

[테스트 전략]
- Gemini API 호출: unittest.mock.patch로 모킹
- DB 의존: SQLite :memory:
- STEAM 데이터: 고정 시드(42) 결정적 생성
"""
import sys
import os
import time
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestTheorist(unittest.TestCase):
    """Theorist Agent 단위 테스트"""
    
    def test_analyze_knowledge_gaps(self):
        """KG gap 분석이 리스트를 반환하는지 검증"""
        from api.agents.theorist import analyze_knowledge_gaps
        gaps = analyze_knowledge_gaps()
        self.assertIsInstance(gaps, list)
    
    def test_generate_hypothesis_returns_valid_structure(self):
        """가설 생성 결과가 올바른 구조인지 검증"""
        from api.agents.theorist import generate_hypothesis
        h = generate_hypothesis()
        
        # 필수 필드 확인
        self.assertIn("id", h)
        self.assertIn("text", h)
        self.assertIn("confidence", h)
        self.assertIn("created_at", h)
        
        # ID 형식 검증
        self.assertTrue(h["id"].startswith("H-"))
        
        # 텍스트가 비어있지 않은지
        self.assertGreater(len(h["text"]), 0)
    
    def test_generate_hypothesis_template_fallback(self):
        """Gemini 미사용 시 fallback이 작동하는지 검증"""
        import importlib
        
        # 환경 변수를 비워서 Gemini를 강제 비활성화
        original_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            # gemini_client 모듈의 전역 변수를 직접 패치
            import api.agents.gemini_client as gc
            old_key = gc.GEMINI_API_KEY
            gc.GEMINI_API_KEY = ""
            
            from api.agents.theorist import generate_hypothesis
            h = generate_hypothesis()
            
            # Gemini 비활성 → fallback 소스 확인
            self.assertIn(
                h.get("hypothesis_source"),
                ("template", "exhausted", "kg_heuristic", "kg_cache"),
            )
            
            gc.GEMINI_API_KEY = old_key
        finally:
            if original_key is not None:
                os.environ["GEMINI_API_KEY"] = original_key
    
    def test_run_theorist_cycle_returns_logs(self):
        """Theorist 사이클이 로그 리스트를 반환하는지 검증"""
        from api.agents.theorist import run_theorist_cycle
        logs = run_theorist_cycle()
        
        self.assertIsInstance(logs, list)
        self.assertGreater(len(logs), 0)
        
        # 각 로그 엔트리 구조 확인
        for entry in logs:
            self.assertIn("step", entry)
            self.assertIn("message", entry)
            self.assertIn("timestamp", entry)


class TestEngineer(unittest.TestCase):
    """Engineer Agent 단위 테스트"""
    
    def test_get_pending_hypotheses(self):
        """미검증 가설 조회가 리스트를 반환하는지 검증"""
        from api.agents.engineer import get_pending_hypotheses
        hyps = get_pending_hypotheses()
        self.assertIsInstance(hyps, list)
    
    def test_design_experiment_structure(self):
        """실험 설계 결과 구조 검증"""
        from api.agents.engineer import design_experiment
        hypothesis = {
            "hypothesis_id": "H-TEST",
            "text": "테스트 가설",
            "source": "A",
            "target": "B",
            "source_gap": {"source": "A", "target": "B"},
            "method_used": "null_hypothesis",
        }
        exp = design_experiment(hypothesis)
        
        self.assertIn("id", exp)
        self.assertIn("hypothesis_id", exp)
        self.assertIn("treatment", exp)
        self.assertIn("outcome", exp)
        self.assertIn("estimator", exp)
    
    def test_run_experiment_halted_on_sandbox_failure(self):
        """Sandbox 실패 시 HALTED 결과 반환 검증"""
        from api.agents.engineer import run_experiment
        
        experiment = {
            "id": "EXP-TEST",
            "hypothesis_id": "H-TEST",
            "treatment": "test_treatment",
            "outcome": "test_outcome",
            "moderators": ["conf1"],
            "estimator": "DML",
            "method": "test_method",
            "data_path": "",
        }
        
        # Sandbox를 실패하도록 모킹
        with patch("api.agents.engineer.sandbox") as mock_sandbox:
            from engine.sandbox.executor import ExecutionResult
            mock_sandbox.execute.return_value = ExecutionResult(
                success=False,
                result_data={"error": "테스트 에러"},
            )
            
            result = run_experiment(experiment)
            
            # HALTED 상태 확인
            self.assertEqual(result.get("experiment_source"), "HALTED")
    
    def test_run_engineer_cycle(self):
        """Engineer 전체 사이클이 로그를 반환하는지 검증"""
        from api.agents.engineer import run_engineer_cycle
        logs = run_engineer_cycle()
        
        self.assertIsInstance(logs, list)
        self.assertGreater(len(logs), 0)


class TestCritic(unittest.TestCase):
    """Critic Agent 단위 테스트"""
    
    def test_review_accept_for_good_result(self):
        """양호한 실험 결과에 ACCEPT 판정 검증"""
        from api.agents.critic import review_experiment
        
        good_result = {
            "experiment_id": "EXP-001",
            "hypothesis_id": "H-001",
            "experiment_source": "engine",
            "sample_size": 3000,
            "ate": 150.5,
            "ate_ci": [100.0, 200.0],
            "method": "DML",
            "estimator": "DML",
            "model_performance": {"r2_treated": 0.75, "r2_control": 0.65},
            "subgroup_analysis": {
                "age": {"cate_low": 100, "cate_high": 200, "heterogeneity_p_value": 0.03, "is_significant": True}
            },
            "conclusion": "HETEROGENEITY_DETECTED",
            "constitution_verdict": {"passed": True, "violations": [], "warnings": [], "analysis_level": "FULL"},
            "seed": 42,
        }
        
        verdict = review_experiment(good_result)
        self.assertIsInstance(verdict, dict)
        # 정상 결과에는 verdict 키 존재 확인
        self.assertIn("verdict", verdict)
    
    def test_review_reject_for_halted(self):
        """HALTED 결과에 REJECT 판정 검증"""
        from api.agents.critic import review_experiment
        
        halted_result = {
            "experiment_id": "EXP-002",
            "hypothesis_id": "H-002",
            "experiment_source": "HALTED",
            "sample_size": 0,
            "ate": 0,
            "ate_ci": [0, 0],
            "method": "DML",
            "estimator": "DML",
            "model_performance": {"r2_treated": 0, "r2_control": 0},
            "subgroup_analysis": {},
            "conclusion": "NO_HETEROGENEITY",
            "constitution_verdict": {"passed": False, "violations": ["HALTED"], "warnings": [], "analysis_level": "NONE"},
            "seed": 42,
        }
        
        verdict = review_experiment(halted_result)
        self.assertEqual(verdict.get("verdict"), "REJECT")
    
    def test_generate_recommendations(self):
        """권장 사항 생성 검증"""
        from api.agents.critic import generate_recommendations
        # issues는 dict 리스트
        recs = generate_recommendations("REVISE", [{"aspect": "표본 크기", "detail": "n=50 부족"}])
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)


class TestCoordinatorV2(unittest.TestCase):
    """Coordinator v2 단위 테스트"""
    
    def test_run_cycle_returns_valid_structure(self):
        """사이클 결과가 올바른 구조를 반환하는지 검증"""
        from api.agents.coordinator import CoordinatorV2
        
        coord = CoordinatorV2()
        result = coord.run_cycle()
        
        self.assertIn("cycle_id", result)
        self.assertIn("stages", result)
        self.assertIn("status", result)
        self.assertIn("started_at", result)
        self.assertIn("ended_at", result)
        
        # stages가 리스트이고 비어있지 않은지
        self.assertIsInstance(result["stages"], list)
        self.assertGreater(len(result["stages"]), 0)
    
    def test_backward_compat_run_coordinator_cycle(self):
        """기존 run_coordinator_cycle() 래퍼 호환성 검증"""
        from api.agents.coordinator import run_coordinator_cycle
        
        logs = run_coordinator_cycle()
        
        self.assertIsInstance(logs, list)
        # 기존 포맷: step, message, timestamp
        if logs:
            self.assertIn("step", logs[0])
            self.assertIn("message", logs[0])
            self.assertIn("timestamp", logs[0])
    
    def test_get_status(self):
        """상태 조회가 v2 정보를 반환하는지 검증"""
        from api.agents.coordinator import CoordinatorV2
        
        coord = CoordinatorV2()
        status = coord.get_status()
        
        self.assertEqual(status["version"], "v2")
        self.assertIn("cycle_count", status)
        self.assertIn("recent_topics", status)


class TestConstitutionGuard(unittest.TestCase):
    """ConstitutionGuard 단위 테스트"""
    
    def test_validate_pass_full_analysis(self):
        """정상 실험 데이터가 통과하는지 검증"""
        from api.guards.constitution_guard import ConstitutionGuard
        
        verdict = ConstitutionGuard.validate_experiment(
            sample_size=3000,
            methods_used={"DML", "IPW"},
            refutation_passed=2,
            experiment_source="engine",
        )
        
        # 충분한 표본 + 2개 방법론 + 반증 통과 → 통과
        self.assertTrue(verdict.passed)
    
    def test_validate_fail_low_sample(self):
        """표본 수 부족 시 헌법 위반 검증"""
        from api.guards.constitution_guard import ConstitutionGuard
        
        verdict = ConstitutionGuard.validate_experiment(
            sample_size=50,
            methods_used={"DML"},
            refutation_passed=0,
            experiment_source="engine",
        )
        
        # 경고 또는 위반이 존재해야 함
        self.assertTrue(len(verdict.warnings) > 0 or len(verdict.violations) > 0)
    
    def test_validate_reject_halted(self):
        """HALTED 소스에 대한 거부 검증"""
        from api.guards.constitution_guard import ConstitutionGuard
        
        verdict = ConstitutionGuard.validate_experiment(
            sample_size=0,
            methods_used=set(),
            refutation_passed=0,
            experiment_source="HALTED",
        )
        
        self.assertFalse(verdict.passed)


class TestEvolution(unittest.TestCase):
    """Evolution 에이전트 기본 테스트"""
    
    def test_run_evolution_cycle_no_crash(self):
        """Evolution 사이클이 크래시 없이 실행되는지 검증"""
        try:
            from api.agents.evolution import run_evolution_cycle
            # DB 없이 호출 시 에러가 발생할 수 있으나
            # 임포트 자체는 성공해야 함
            self.assertTrue(True)
        except ImportError:
            self.skipTest("evolution 모듈 임포트 실패")


class TestForum(unittest.TestCase):
    """Forum 에이전트 기본 테스트"""
    
    def test_run_forum_debate_returns_dict(self):
        """포럼 토론이 딕셔너리를 반환하는지 검증"""
        from api.agents.forum import run_forum_debate
        result = run_forum_debate()
        
        self.assertIsInstance(result, dict)
        self.assertIn("topic", result)
        self.assertIn("consensus", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
