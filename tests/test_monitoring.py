# -*- coding: utf-8 -*-
"""모니터링 패키지 단위 테스트.

DriftDetector, Alerter, DriftResult를 검증합니다.
"""

import pytest

from engine.monitoring.drift_detector import DriftDetector, DriftResult
from engine.monitoring.alerter import Alerter, Alert, AlertLevel


# ──────────────────────────────────────────────
# DriftResult 테스트
# ──────────────────────────────────────────────
class TestDriftResult:
    """DriftResult 데이터클래스 테스트."""

    def test_default_values(self):
        """기본값이 올바르게 설정됩니다."""
        result = DriftResult()
        assert result.drifted is False
        assert result.metric == ""
        assert result.score == 0.0
        assert result.threshold == 0.0
        assert result.details == {}

    def test_custom_values(self):
        """커스텀 값이 올바르게 설정됩니다."""
        result = DriftResult(
            drifted=True,
            metric="ate_change_rate",
            score=0.75,
            threshold=0.5,
        )
        assert result.drifted is True
        assert result.score == 0.75


# ──────────────────────────────────────────────
# DriftDetector 테스트
# ──────────────────────────────────────────────
class TestDriftDetector:
    """드리프트 탐지기 테스트."""

    def test_insufficient_data(self):
        """스냅샷이 부족할 때 드리프트를 감지하지 않습니다."""
        detector = DriftDetector(min_snapshots=2)
        detector.add_snapshot(ate=0.05)
        result = detector.check_drift()
        assert result.drifted is False
        assert result.metric == "insufficient_data"

    def test_no_drift_stable_ate(self):
        """ATE가 안정적이면 드리프트를 감지하지 않습니다."""
        detector = DriftDetector(ate_change_threshold=0.5)
        detector.add_snapshot(ate=0.10)
        detector.add_snapshot(ate=0.11)
        detector.add_snapshot(ate=0.10)
        result = detector.check_drift()
        assert result.drifted is False

    def test_drift_ate_large_change(self):
        """ATE가 크게 변하면 드리프트를 감지합니다."""
        detector = DriftDetector(ate_change_threshold=0.3)
        detector.add_snapshot(ate=0.10)
        detector.add_snapshot(ate=0.10)
        detector.add_snapshot(ate=0.50)  # 400% 변화
        result = detector.check_drift()
        assert result.drifted == True

    def test_drift_sign_flip(self):
        """ATE 부호가 반전되면 드리프트를 감지합니다."""
        detector = DriftDetector()
        detector.add_snapshot(ate=0.10)
        detector.add_snapshot(ate=0.08)
        detector.add_snapshot(ate=-0.05)  # 부호 반전
        result = detector.check_drift()
        assert result.drifted == True

    def test_kl_divergence_no_drift(self):
        """CATE 분포가 비슷하면 KL-Div 드리프트를 감지하지 않습니다."""
        import numpy as np
        rng = np.random.default_rng(42)

        detector = DriftDetector(kl_threshold=0.5)
        dist1 = rng.normal(0, 1, 200).tolist()
        dist2 = rng.normal(0, 1, 200).tolist()

        detector.add_snapshot(ate=0.10, cate_distribution=dist1)
        detector.add_snapshot(ate=0.10, cate_distribution=dist2)

        result = detector.check_drift()
        assert result.drifted is False

    def test_snapshot_count(self):
        """스냅샷 수가 올바르게 추적됩니다."""
        detector = DriftDetector()
        assert detector.snapshot_count == 0
        detector.add_snapshot(ate=0.1)
        detector.add_snapshot(ate=0.2)
        assert detector.snapshot_count == 2

    def test_reset(self):
        """reset이 스냅샷을 초기화합니다."""
        detector = DriftDetector()
        detector.add_snapshot(ate=0.1)
        detector.reset()
        assert detector.snapshot_count == 0


# ──────────────────────────────────────────────
# Alerter 테스트
# ──────────────────────────────────────────────
class TestAlerter:
    """알림 발송기 테스트."""

    def test_log_alert(self):
        """로그 알림이 히스토리에 기록됩니다."""
        alerter = Alerter(log_alerts=True)
        alert = Alert(
            level=AlertLevel.WARNING,
            title="테스트 알림",
            message="테스트 메시지",
        )
        alerter.send(alert)

        assert len(alerter.history) == 1
        assert alerter.history[0].title == "테스트 알림"

    def test_multiple_alerts(self):
        """여러 알림이 히스토리에 누적됩니다."""
        alerter = Alerter(log_alerts=False)
        for i in range(3):
            alerter.send(Alert(
                level=AlertLevel.INFO,
                title=f"알림 {i}",
                message=f"메시지 {i}",
            ))
        assert len(alerter.history) == 3

    def test_clear_history(self):
        """히스토리 초기화가 동작합니다."""
        alerter = Alerter(log_alerts=False)
        alerter.send(Alert(level=AlertLevel.INFO, title="t", message="m"))
        alerter.clear_history()
        assert len(alerter.history) == 0

    def test_alert_levels(self):
        """AlertLevel enum 값이 올바릅니다."""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"

    def test_alert_timestamp(self):
        """Alert의 timestamp가 자동 생성됩니다."""
        alert = Alert(level=AlertLevel.INFO, title="t", message="m")
        assert alert.timestamp != ""
        assert "T" in alert.timestamp  # ISO format
