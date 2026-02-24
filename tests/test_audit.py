# -*- coding: utf-8 -*-
"""Causal Audit 인프라 통합 테스트.

schemas → decision_logger → ga4_connector → matcher → causal_auditor
전체 파이프라인 흐름을 검증합니다.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engine.audit.schemas import (
    AgentType,
    AuditResult,
    AuditVerdict,
    DecisionEvent,
    DecisionOutcomePair,
    DecisionType,
    OutcomeEvent,
    OutcomeMetric,
)
from engine.audit.decision_logger import DecisionLogger
from engine.audit.matcher import DecisionOutcomeMatcher
from engine.audit.causal_auditor import CausalAuditor
from engine.connectors.ga4_connector import GA4Connector


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def tmp_log_dir(tmp_path):
    """임시 로그 디렉토리."""
    return str(tmp_path / "decisions")


@pytest.fixture
def decision():
    """샘플 결정 이벤트."""
    return DecisionEvent(
        agent_type=AgentType.HIVE_MIND,
        agent_name="hive_mind_toolpick",
        decision_type=DecisionType.CONTENT_STRATEGY,
        treatment="키워드 전략을 AI Tools로 변경",
        target_sbu="toolpick",
        target_metric=OutcomeMetric.ORGANIC_TRAFFIC,
        timestamp="2026-02-10T00:00:00+00:00",
        observation_window_days=7,
    )


def _make_outcomes(sbu, metric, base, n_days, start_date, noise_factor=0.1):
    """관측 데이터 생성 헬퍼."""
    import random
    random.seed(42)
    outcomes = []
    for i in range(n_days):
        dt = start_date + timedelta(days=i)
        noise = random.gauss(0, base * noise_factor)
        outcomes.append(OutcomeEvent(
            metric=metric,
            value=round(base + noise, 2),
            sbu=sbu,
            timestamp=dt.isoformat(),
            source="test",
        ))
    return outcomes


# ──────────────────────────────────────────────
# Schema 테스트
# ──────────────────────────────────────────────

class TestSchemas:
    def test_decision_event_to_dict(self, decision):
        d = decision.to_dict()
        assert d["agent_type"] == "hive_mind"
        assert d["decision_type"] == "content_strategy"
        assert d["target_metric"] == "organic_traffic"

    def test_outcome_event_to_dict(self):
        o = OutcomeEvent(
            metric=OutcomeMetric.PAGE_VIEWS,
            value=350.5,
            sbu="toolpick",
        )
        d = o.to_dict()
        assert d["metric"] == "page_views"
        assert d["value"] == 350.5

    def test_decision_outcome_pair_ready(self, decision):
        start = datetime.fromisoformat(decision.timestamp)
        pre = _make_outcomes("toolpick", OutcomeMetric.ORGANIC_TRAFFIC, 100, 14,
                             start - timedelta(days=14))
        post = _make_outcomes("toolpick", OutcomeMetric.ORGANIC_TRAFFIC, 120, 7,
                              start)
        pair = DecisionOutcomePair(decision=decision, pre_outcomes=pre, post_outcomes=post)
        assert pair.is_ready_for_audit
        assert len(pair.pre_values) == 14
        assert len(pair.post_values) == 7

    def test_decision_outcome_pair_not_ready(self, decision):
        pair = DecisionOutcomePair(decision=decision)
        assert not pair.is_ready_for_audit


# ──────────────────────────────────────────────
# Decision Logger 테스트
# ──────────────────────────────────────────────

class TestDecisionLogger:
    def test_log_and_retrieve(self, tmp_log_dir):
        logger = DecisionLogger(log_dir=tmp_log_dir)
        d = logger.log_decision(
            agent_type=AgentType.CRO_AGENT,
            agent_name="cro_toolpick",
            decision_type=DecisionType.UI_CHANGE,
            treatment="CTA 버튼 색상 빨간색으로 변경",
            target_sbu="toolpick",
            target_metric=OutcomeMetric.CLICK_RATE,
        )
        assert d.decision_id

        decisions = logger.get_decisions()
        assert len(decisions) >= 1

    def test_filter_by_sbu(self, tmp_log_dir):
        logger = DecisionLogger(log_dir=tmp_log_dir)
        logger.log_decision(
            agent_type=AgentType.HIVE_MIND,
            agent_name="hm_toolpick",
            decision_type=DecisionType.CONTENT_STRATEGY,
            treatment="test1",
            target_sbu="toolpick",
            target_metric=OutcomeMetric.PAGE_VIEWS,
        )
        logger.log_decision(
            agent_type=AgentType.HIVE_MIND,
            agent_name="hm_urwrong",
            decision_type=DecisionType.CONTENT_STRATEGY,
            treatment="test2",
            target_sbu="ur-wrong",
            target_metric=OutcomeMetric.PAGE_VIEWS,
        )

        toolpick = logger.get_decisions(sbu="toolpick")
        assert len(toolpick) == 1

    def test_jsonl_persistence(self, tmp_log_dir):
        logger = DecisionLogger(log_dir=tmp_log_dir)
        logger.log_decision(
            agent_type=AgentType.FARMING_BOT,
            agent_name="farming_1",
            decision_type=DecisionType.PROTOCOL_INTERACTION,
            treatment="프로토콜 A에 유동성 공급",
            target_sbu="profit",
            target_metric=OutcomeMetric.REVENUE,
        )

        filepath = Path(tmp_log_dir) / "decisions.jsonl"
        assert filepath.exists()
        with open(filepath, encoding="utf-8") as f:
            data = json.loads(f.readline())
        assert data["agent_name"] == "farming_1"


# ──────────────────────────────────────────────
# GA4 Connector 테스트
# ──────────────────────────────────────────────

class TestGA4Connector:
    def test_mock_outcomes(self):
        connector = GA4Connector()
        outcomes = connector.fetch_outcomes(
            metric=OutcomeMetric.ORGANIC_TRAFFIC,
            start_date="2026-02-01",
            end_date="2026-02-14",
            sbu="toolpick",
        )
        assert len(outcomes) == 14
        assert all(o.source == "mock" for o in outcomes)
        assert all(o.metric == OutcomeMetric.ORGANIC_TRAFFIC for o in outcomes)

    def test_not_connected(self):
        connector = GA4Connector()
        assert not connector.is_connected


# ──────────────────────────────────────────────
# Matcher 테스트
# ──────────────────────────────────────────────

class TestMatcher:
    def test_match_basic(self, decision):
        start = datetime.fromisoformat(decision.timestamp)
        outcomes = (
            _make_outcomes("toolpick", OutcomeMetric.ORGANIC_TRAFFIC, 100, 14,
                           start - timedelta(days=14))
            + _make_outcomes("toolpick", OutcomeMetric.ORGANIC_TRAFFIC, 120, 7,
                             start)
        )

        matcher = DecisionOutcomeMatcher()
        pairs = matcher.match([decision], outcomes)
        assert len(pairs) == 1
        assert pairs[0].is_ready_for_audit

    def test_match_filters_sbu(self, decision):
        start = datetime.fromisoformat(decision.timestamp)
        wrong_sbu = _make_outcomes("ur-wrong", OutcomeMetric.ORGANIC_TRAFFIC, 100, 21,
                                    start - timedelta(days=14))

        matcher = DecisionOutcomeMatcher()
        pairs = matcher.match([decision], wrong_sbu)
        assert len(pairs) == 0

    def test_match_filters_metric(self, decision):
        start = datetime.fromisoformat(decision.timestamp)
        wrong_metric = _make_outcomes("toolpick", OutcomeMetric.BOUNCE_RATE, 0.5, 21,
                                      start - timedelta(days=14))

        matcher = DecisionOutcomeMatcher()
        pairs = matcher.match([decision], wrong_metric)
        assert len(pairs) == 0


# ──────────────────────────────────────────────
# Causal Auditor 테스트
# ──────────────────────────────────────────────

class TestCausalAuditor:
    def test_audit_insufficient_data(self, decision):
        pair = DecisionOutcomePair(decision=decision)
        auditor = CausalAuditor()
        result = auditor.audit(pair)
        assert result.verdict == AuditVerdict.INSUFFICIENT_DATA

    def test_audit_with_effect(self, decision):
        """효과가 큰 경우 CAUSAL 판결."""
        start = datetime.fromisoformat(decision.timestamp)
        pair = DecisionOutcomePair(
            decision=decision,
            pre_outcomes=_make_outcomes("toolpick", OutcomeMetric.ORGANIC_TRAFFIC,
                                        100, 14, start - timedelta(days=14), 0.05),
            post_outcomes=_make_outcomes("toolpick", OutcomeMetric.ORGANIC_TRAFFIC,
                                         150, 7, start, 0.05),
        )
        auditor = CausalAuditor()
        result = auditor.audit(pair)
        assert result.verdict == AuditVerdict.CAUSAL
        assert result.ate > 0
        assert result.confidence > 0.5
        assert "Causal Audit Report" in result.recommendation

    def test_audit_no_effect(self, decision):
        """효과 없는 경우 NOT_CAUSAL 판결."""
        start = datetime.fromisoformat(decision.timestamp)
        pair = DecisionOutcomePair(
            decision=decision,
            pre_outcomes=_make_outcomes("toolpick", OutcomeMetric.ORGANIC_TRAFFIC,
                                        100, 14, start - timedelta(days=14), 0.05),
            post_outcomes=_make_outcomes("toolpick", OutcomeMetric.ORGANIC_TRAFFIC,
                                         100, 7, start, 0.05),
        )
        auditor = CausalAuditor()
        result = auditor.audit(pair)
        assert result.verdict in (AuditVerdict.NOT_CAUSAL, AuditVerdict.UNCERTAIN)

    def test_audit_report_markdown(self, decision):
        """감사 보고서가 마크다운 형식."""
        start = datetime.fromisoformat(decision.timestamp)
        pair = DecisionOutcomePair(
            decision=decision,
            pre_outcomes=_make_outcomes("toolpick", OutcomeMetric.ORGANIC_TRAFFIC,
                                        100, 14, start - timedelta(days=14)),
            post_outcomes=_make_outcomes("toolpick", OutcomeMetric.ORGANIC_TRAFFIC,
                                         130, 7, start),
        )
        result = CausalAuditor().audit(pair)
        assert "## " in result.recommendation
        assert "ATE" in result.recommendation
        assert "p-value" in result.recommendation


# ──────────────────────────────────────────────
# 통합 테스트: 전체 파이프라인
# ──────────────────────────────────────────────

class TestFullPipeline:
    def test_end_to_end(self, tmp_log_dir):
        """Decision → Log → GA4 Mock → Match → Audit 전체."""
        # 1. 결정 기록
        dl = DecisionLogger(log_dir=tmp_log_dir)
        decision = dl.log_decision(
            agent_type=AgentType.HIVE_MIND,
            agent_name="hive_mind_toolpick",
            decision_type=DecisionType.CONTENT_STRATEGY,
            treatment="AI 도구 리뷰 시리즈 시작",
            target_sbu="toolpick",
            target_metric=OutcomeMetric.PAGE_VIEWS,
            observation_window_days=7,
        )

        # 2. GA4 데이터 수집 (Mock)
        connector = GA4Connector()
        start = datetime.fromisoformat(decision.timestamp)
        pre_date = (start - timedelta(days=14)).strftime("%Y-%m-%d")
        post_date = (start + timedelta(days=7)).strftime("%Y-%m-%d")

        outcomes = connector.fetch_outcomes(
            metric=OutcomeMetric.PAGE_VIEWS,
            start_date=pre_date,
            end_date=post_date,
            sbu="toolpick",
        )
        assert len(outcomes) >= 14

        # 3. 매칭
        matcher = DecisionOutcomeMatcher()
        pairs = matcher.match([decision], outcomes)
        assert len(pairs) == 1

        # 4. 감사
        auditor = CausalAuditor()
        pair = pairs[0]
        if pair.is_ready_for_audit:
            result = auditor.audit(pair)
            assert result.verdict in (
                AuditVerdict.CAUSAL,
                AuditVerdict.NOT_CAUSAL,
                AuditVerdict.UNCERTAIN,
            )
            assert result.recommendation


# ──────────────────────────────────────────────
# 피드백 컨트롤러 테스트
# ──────────────────────────────────────────────

from engine.audit.feedback_controller import DampingController, FeedbackSignal


class TestDampingController:
    def test_high_confidence_high_damping(self):
        """높은 신뢰도 → 높은 감쇠 → 공격적 업데이트."""
        ctrl = DampingController()
        zeta = ctrl.compute_damping(confidence=0.9, drift_index=0.1)
        assert zeta > ctrl.base_damping

    def test_high_drift_low_damping(self):
        """높은 드리프트 → 낮은 감쇠 → 보수적 유지."""
        ctrl = DampingController()
        zeta = ctrl.compute_damping(confidence=0.5, drift_index=0.8)
        assert zeta < ctrl.base_damping

    def test_sparse_data_reduces_damping(self):
        """데이터 희소 → 감쇠 하향."""
        ctrl = DampingController()
        dense = ctrl.compute_damping(confidence=0.7, data_density=1.0)
        sparse = ctrl.compute_damping(confidence=0.7, data_density=0.2)
        assert sparse < dense

    def test_damping_within_bounds(self):
        """감쇠 인자가 항상 [min, max] 범위 내."""
        ctrl = DampingController(min_damping=0.05, max_damping=0.8)
        for conf in [0.0, 0.5, 1.0]:
            for drift in [0.0, 0.5, 1.0]:
                for density in [0.1, 0.5, 1.0]:
                    z = ctrl.compute_damping(conf, drift, density)
                    assert 0.05 <= z <= 0.8

    def test_generate_feedback_causal(self, decision):
        """CAUSAL 감사 결과 → reinforce 피드백."""
        ctrl = DampingController()
        result = AuditResult(
            decision_id=decision.decision_id,
            verdict=AuditVerdict.CAUSAL,
            confidence=0.8,
            ate=0.15,
        )
        signal = ctrl.generate_feedback(result)
        assert signal.action == "reinforce"
        assert signal.effective_weight > 0

    def test_generate_feedback_not_causal(self, decision):
        """NOT_CAUSAL 감사 결과 → suppress 피드백."""
        ctrl = DampingController()
        result = AuditResult(
            decision_id=decision.decision_id,
            verdict=AuditVerdict.NOT_CAUSAL,
            confidence=0.7,
            ate=-0.02,
        )
        signal = ctrl.generate_feedback(result)
        assert signal.action == "suppress"

    def test_feedback_history(self, decision):
        """피드백 이력이 기록됨."""
        ctrl = DampingController()
        result = AuditResult(
            decision_id=decision.decision_id,
            verdict=AuditVerdict.UNCERTAIN,
            confidence=0.4,
        )
        ctrl.generate_feedback(result)
        assert len(ctrl.history) == 1
        assert ctrl.history[0]["action"] == "hold"
