# -*- coding: utf-8 -*-
"""인과 드리프트 모니터 — CI(X→Y) 시간 변화 추적.

에이전트 결정의 인과적 영향력이 시간에 따라 변화하는지 감시합니다.
DI(Drift Index) 임계값 초과 시 DampingController에 경고를 보내
보수적 업데이트 모드로 전환합니다.

고도화 리서치(v2.1) 기반:
- CI(X→Y) 시계열 추적
- 구조적 변화(Structural break) 감지
- DampingController 연동
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional

from engine.audit.schemas import AuditResult, AuditVerdict

logger = logging.getLogger("whylab.audit.drift_monitor")


class CausalDriftMonitor:
    """인과적 드리프트 지수(DI) 모니터링.

    최근 감사 결과의 판결 변동성과 ATE 변화율을 추적하여
    환경의 안정성을 실시간 평가합니다.
    """

    def __init__(
        self,
        drift_threshold: float = 0.3,
        window_size: int = 10,
        break_sensitivity: float = 2.0,
    ) -> None:
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.break_sensitivity = break_sensitivity
        self._audit_history: List[AuditResult] = []

    def record(self, result: AuditResult) -> float:
        """감사 결과를 기록하고 현재 DI를 반환합니다."""
        self._audit_history.append(result)
        di = self.compute_drift_index()
        if di > self.drift_threshold:
            logger.warning(
                "🚨 Drift Alert: DI=%.3f > threshold=%.3f (history=%d)",
                di, self.drift_threshold, len(self._audit_history),
            )
        return di

    def compute_drift_index(
        self,
        recent_audits: Optional[List[AuditResult]] = None,
        window_days: int = 30,
    ) -> float:
        """최근 감사 결과의 드리프트 지수를 계산합니다.

        DI = (판결 변동률 × 0.4) + (ATE 변동 계수 × 0.3) + (신뢰도 하락률 × 0.3)

        Returns:
            드리프트 지수 (0~1, 높을수록 불안정)
        """
        audits = recent_audits or self._audit_history
        if len(audits) < 3:
            return 0.0

        recent = audits[-self.window_size:]

        # 1. 판결 변동률 (verdict 전환 빈도)
        verdict_changes = 0
        for i in range(1, len(recent)):
            if recent[i].verdict != recent[i - 1].verdict:
                verdict_changes += 1
        verdict_volatility = verdict_changes / max(len(recent) - 1, 1)

        # 2. ATE 변동 계수 (Coefficient of Variation)
        ates = [r.ate for r in recent if r.ate != 0]
        if len(ates) >= 2:
            ate_mean = statistics.mean(ates)
            ate_std = statistics.stdev(ates)
            ate_cv = ate_std / abs(ate_mean) if abs(ate_mean) > 1e-10 else 0
            ate_volatility = min(ate_cv, 1.0)
        else:
            ate_volatility = 0.0

        # 3. 신뢰도 하락률
        confidences = [r.confidence for r in recent]
        if len(confidences) >= 2:
            first_half = statistics.mean(confidences[:len(confidences) // 2])
            second_half = statistics.mean(confidences[len(confidences) // 2:])
            conf_decline = max(0, first_half - second_half)
        else:
            conf_decline = 0.0

        di = (
            verdict_volatility * 0.4
            + ate_volatility * 0.3
            + conf_decline * 0.3
        )

        return round(min(di, 1.0), 4)

    def detect_structural_break(self) -> bool:
        """환경의 구조적 변화를 감지합니다.

        최근 ATE가 이전 평균에서 break_sensitivity × σ 이상
        벗어나면 구조적 변화로 판단합니다.
        """
        if len(self._audit_history) < 6:
            return False

        mid = len(self._audit_history) // 2
        old_ates = [r.ate for r in self._audit_history[:mid]]
        new_ates = [r.ate for r in self._audit_history[mid:]]

        if not old_ates or not new_ates:
            return False

        old_mean = statistics.mean(old_ates)
        old_std = statistics.stdev(old_ates) if len(old_ates) > 1 else 1.0
        new_mean = statistics.mean(new_ates)

        deviation = abs(new_mean - old_mean) / max(old_std, 1e-10)
        is_break = deviation > self.break_sensitivity

        if is_break:
            logger.warning(
                "🔴 Structural Break 감지: old_mean=%.4f, new_mean=%.4f, "
                "deviation=%.2fσ",
                old_mean, new_mean, deviation,
            )

        return is_break

    def get_status(self) -> Dict[str, Any]:
        """현재 모니터링 상태를 반환합니다."""
        di = self.compute_drift_index()
        return {
            "drift_index": di,
            "is_drifting": di > self.drift_threshold,
            "structural_break": self.detect_structural_break(),
            "history_size": len(self._audit_history),
            "recent_verdicts": [
                r.verdict.value for r in self._audit_history[-5:]
            ],
        }
