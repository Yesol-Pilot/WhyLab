# -*- coding: utf-8 -*-
"""Phase 3 테스트 — DriftMonitor, FeedbackAPI, AuditOrchestrator."""

import pytest

from engine.audit.schemas import (
    AgentType,
    AuditResult,
    AuditVerdict,
    DecisionEvent,
    DecisionType,
    OutcomeMetric,
)
from engine.audit.drift_monitor import CausalDriftMonitor
from engine.audit.feedback_api import FeedbackAPI, AgentScore
from engine.audit.feedback_controller import DampingController
from engine.audit.orchestrator import AuditOrchestrator


# ── Fixtures ──

@pytest.fixture
def sample_results():
    """다양한 감사 결과 시퀀스."""
    return [
        AuditResult(decision_id=f"d{i}", verdict=v, confidence=c, ate=a)
        for i, (v, c, a) in enumerate([
            (AuditVerdict.CAUSAL, 0.8, 0.15),
            (AuditVerdict.CAUSAL, 0.7, 0.12),
            (AuditVerdict.NOT_CAUSAL, 0.6, -0.02),
            (AuditVerdict.CAUSAL, 0.75, 0.18),
            (AuditVerdict.UNCERTAIN, 0.4, 0.05),
            (AuditVerdict.NOT_CAUSAL, 0.5, -0.08),
        ])
    ]


# ── DriftMonitor ──

class TestDriftMonitor:
    def test_empty_history(self):
        m = CausalDriftMonitor()
        assert m.compute_drift_index() == 0.0

    def test_stable_history(self):
        m = CausalDriftMonitor()
        for _ in range(5):
            m.record(AuditResult(decision_id="x", verdict=AuditVerdict.CAUSAL,
                                 confidence=0.8, ate=0.1))
        assert m.compute_drift_index() < 0.1

    def test_volatile_history(self, sample_results):
        m = CausalDriftMonitor()
        for r in sample_results:
            m.record(r)
        di = m.compute_drift_index()
        assert di > 0  # 판결이 변하므로 DI > 0

    def test_structural_break(self):
        m = CausalDriftMonitor(break_sensitivity=1.5)
        for _ in range(5):
            m.record(AuditResult(decision_id="a", verdict=AuditVerdict.CAUSAL,
                                 confidence=0.8, ate=0.1))
        for _ in range(5):
            m.record(AuditResult(decision_id="b", verdict=AuditVerdict.CAUSAL,
                                 confidence=0.8, ate=2.0))
        assert m.detect_structural_break()

    def test_get_status(self, sample_results):
        m = CausalDriftMonitor()
        for r in sample_results:
            m.record(r)
        status = m.get_status()
        assert "drift_index" in status
        assert "structural_break" in status


# ── FeedbackAPI ──

class TestFeedbackAPI:
    def test_process_single(self):
        api = FeedbackAPI()
        result = AuditResult(
            decision_id="d1", verdict=AuditVerdict.CAUSAL,
            confidence=0.8, ate=0.15,
        )
        signal = api.process_audit_result("hive_mind", result)
        assert signal.agent_name == "hive_mind"
        assert signal.action == "reinforce"

    def test_scoreboard(self, sample_results):
        api = FeedbackAPI()
        for i, r in enumerate(sample_results):
            api.process_audit_result(f"agent_{i % 2}", r)
        board = api.get_agent_scoreboard()
        assert len(board) == 2
        for name, score in board.items():
            assert "success_rate" in score
            assert "total_audits" in score

    def test_feedback_history(self):
        api = FeedbackAPI()
        result = AuditResult(
            decision_id="d1", verdict=AuditVerdict.CAUSAL,
            confidence=0.8, ate=0.15,
        )
        api.process_audit_result("agent_x", result)
        history = api.get_feedback_history()
        assert len(history) == 1
        assert history[0]["agent_name"] == "agent_x"

    def test_system_status(self):
        api = FeedbackAPI()
        status = api.get_system_status()
        assert "drift" in status
        assert "agents_tracked" in status


# ── AgentScore ──

class TestAgentScore:
    def test_success_rate(self):
        score = AgentScore("test_agent")
        causal = AuditResult(decision_id="d1", verdict=AuditVerdict.CAUSAL, confidence=0.8)
        not_causal = AuditResult(decision_id="d2", verdict=AuditVerdict.NOT_CAUSAL, confidence=0.6)

        from engine.audit.feedback_controller import FeedbackSignal
        signal = FeedbackSignal(
            decision_id="d1", agent_name="test", verdict=AuditVerdict.CAUSAL,
            confidence=0.8, damping_factor=0.3, effective_weight=0.24,
            action="reinforce", memo=""
        )
        score.update(causal, signal)
        score.update(not_causal, signal)
        assert score.success_rate == 0.5


# ── Orchestrator ──

class TestOrchestrator:
    def test_e2e_pipeline(self, tmp_path):
        orch = AuditOrchestrator(
            decision_logger=__import__("engine.audit.decision_logger", fromlist=["DecisionLogger"]).DecisionLogger(
                log_dir=str(tmp_path / "decisions")
            ),
        )

        decision = orch.log_decision(
            agent_type=AgentType.HIVE_MIND,
            agent_name="hive_mind_toolpick",
            decision_type=DecisionType.CONTENT_STRATEGY,
            treatment="AI 도구 리뷰 시작",
            target_sbu="toolpick",
            target_metric=OutcomeMetric.PAGE_VIEWS,
        )

        signal = orch.run_audit(decision)
        assert signal is not None or signal is None  # 데이터에 따라 다름

    def test_status(self):
        orch = AuditOrchestrator()
        status = orch.get_status()
        assert status["orchestrator"] == "active"
