# -*- coding: utf-8 -*-
"""
Adaptive Audit Layer — Condition-Aware C2 Sensitivity Gate
===========================================================
Extends the base AgentAuditLayer with adaptive threshold calibration.

Key insight: Fixed E-value/RV thresholds cause over-rejection on stable
baselines (GPT-4o-mini) while providing appropriate filtering on unstable
ones (Gemini). The Adaptive layer solves this by:

1. WARMUP phase: Observe baseline behavior without intervention
2. CALIBRATE: Set thresholds proportional to observed instability
3. MONITOR: Apply calibrated thresholds with ongoing adaptation

This implements the "Condition-Aware Activation Principle" (C4):
each component activates proportionally to baseline instability.

Usage:
    from experiments.adaptive_audit_layer import AdaptiveAuditLayer
    audit = AdaptiveAuditLayer(config, warmup_epochs=3)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from experiments.audit_layer import (
    AgentAuditLayer,
    AuditDecision,
    DriftMonitor,
    SensitivityGate,
    DampingController,
)


@dataclass
class CalibrationState:
    """Tracks baseline statistics during warmup and ongoing monitoring."""
    warmup_scores: List[float] = field(default_factory=list)
    warmup_complete: bool = False
    baseline_mean: float = 0.0
    baseline_var: float = 0.0
    baseline_instability: float = 0.0  # combined osc + reg signal
    calibrated_e_thresh: float = 2.0
    calibrated_rv_thresh: float = 0.1
    # Running statistics (post-warmup)
    ema_variance: float = 0.0
    n_observations: int = 0


class AdaptiveSensitivityGate(SensitivityGate):
    """C2 with baseline-adaptive thresholds.

    Instead of fixed thresholds that work well for some models but
    over-reject for others, this gate auto-calibrates based on the
    observed baseline variance during a warmup period.

    Threshold scaling logic:
        - High baseline variance (unstable agent) → LOWER thresholds
          = more aggressive filtering (needed to prevent oscillation)
        - Low baseline variance (stable agent) → HIGHER thresholds
          = less filtering (avoids unnecessary rejection)

    This implements the empirical observation from E7v2:
        "benefit is proportional to baseline instability"
    as a concrete mechanism.
    """

    def __init__(
        self,
        base_e_thresh: float = 2.0,
        base_rv_thresh: float = 0.1,
        sensitivity_scaling: float = 10.0,
        min_e_thresh: float = 1.2,
        max_e_thresh: float = 5.0,
        min_rv_thresh: float = 0.02,
        max_rv_thresh: float = 0.5,
    ):
        super().__init__(e_thresh=base_e_thresh, rv_thresh=base_rv_thresh)
        self.base_e_thresh = base_e_thresh
        self.base_rv_thresh = base_rv_thresh
        self.sensitivity_scaling = sensitivity_scaling
        self.min_e_thresh = min_e_thresh
        self.max_e_thresh = max_e_thresh
        self.min_rv_thresh = min_rv_thresh
        self.max_rv_thresh = max_rv_thresh

    def calibrate(self, baseline_variance: float):
        """Calibrate thresholds based on observed baseline variance.

        Intuition: When variance is low, the agent is already stable,
        so we need a higher bar (higher threshold) to justify rejection.
        When variance is high, the agent is unstable, so we lower the
        bar to filter more aggressively.

        Formula:
            e_thresh = base_e / (1 + scaling * var)
            - var → 0: e_thresh → base_e (high bar, minimal filtering)
            - var → ∞: e_thresh → min_e (low bar, aggressive filtering)
        """
        scale = 1.0 + self.sensitivity_scaling * baseline_variance

        self.e_thresh = np.clip(
            self.base_e_thresh / scale,
            self.min_e_thresh,
            self.max_e_thresh,
        )
        self.rv_thresh = np.clip(
            self.base_rv_thresh / scale,
            self.min_rv_thresh,
            self.max_rv_thresh,
        )

    def get_calibration_info(self) -> Dict[str, float]:
        return {
            "e_thresh": self.e_thresh,
            "rv_thresh": self.rv_thresh,
            "base_e_thresh": self.base_e_thresh,
            "base_rv_thresh": self.base_rv_thresh,
        }


class AdaptiveAuditLayer(AgentAuditLayer):
    """Audit layer with condition-aware adaptive thresholds.

    Extends AgentAuditLayer with:
    1. Warmup phase to observe baseline behavior
    2. Automatic threshold calibration
    3. Ongoing variance tracking for threshold adjustment
    """

    def __init__(
        self,
        config: dict,
        warmup_epochs: int = 3,
        sensitivity_scaling: float = 10.0,
        ema_alpha: float = 0.1,
    ):
        # Initialize base components
        self.enable_c1 = config.get("c1", False)
        self.enable_c2 = config.get("c2", False)
        self.enable_c3 = config.get("c3", False)

        self.c1 = DriftMonitor(
            window=config.get("c1_window", 20),
            agreement_threshold=config.get("c1_agreement_threshold", 0.6),
        ) if self.enable_c1 else None

        # Use adaptive C2 instead of fixed
        self.c2 = AdaptiveSensitivityGate(
            base_e_thresh=config.get("c2_e_thresh", 2.0),
            base_rv_thresh=config.get("c2_rv_thresh", 0.1),
            sensitivity_scaling=sensitivity_scaling,
        ) if self.enable_c2 else None

        self.c3 = DampingController(
            epsilon_floor=config.get("c3_epsilon_floor", 0.01),
            ceiling=config.get("c3_ceiling", 0.8),
        ) if self.enable_c3 else None

        # Adaptive-specific state
        self.warmup_epochs = warmup_epochs
        self.ema_alpha = ema_alpha
        self.calibration = CalibrationState()
        self._epoch_count = 0

    def observe_warmup(self, score: float):
        """Record a score during warmup (no filtering applied)."""
        self.calibration.warmup_scores.append(score)

    def complete_warmup(self):
        """Finalize warmup and calibrate thresholds."""
        scores = self.calibration.warmup_scores
        if len(scores) < 2:
            # Not enough data — use base thresholds
            self.calibration.warmup_complete = True
            return

        self.calibration.baseline_mean = np.mean(scores)
        self.calibration.baseline_var = np.var(scores, ddof=1)
        self.calibration.ema_variance = self.calibration.baseline_var

        # Compute instability metric: variance + score drops
        drops = sum(1 for i in range(1, len(scores))
                    if scores[i] < scores[i-1] - 0.1)
        self.calibration.baseline_instability = (
            self.calibration.baseline_var * 100 + drops
        )

        # Calibrate C2
        if self.c2 is not None and isinstance(self.c2, AdaptiveSensitivityGate):
            self.c2.calibrate(self.calibration.baseline_var)
            info = self.c2.get_calibration_info()
            self.calibration.calibrated_e_thresh = info["e_thresh"]
            self.calibration.calibrated_rv_thresh = info["rv_thresh"]

        self.calibration.warmup_complete = True

    def evaluate_update(
        self,
        cheap_score: float,
        full_pass: bool,
        scores_before: list[float],
        scores_after: list[float],
        update_magnitude: float,
    ) -> AuditDecision:
        """Run audit with adaptive thresholds.

        During warmup: always accept (observe only).
        After warmup: apply calibrated gates.
        """
        self._epoch_count += 1
        self.calibration.n_observations += 1

        # Warmup phase: observe without intervention
        if not self.calibration.warmup_complete:
            self.observe_warmup(cheap_score)
            if self._epoch_count >= self.warmup_epochs:
                self.complete_warmup()
            return AuditDecision(
                accept=True,
                details={"phase": "warmup", "epoch": self._epoch_count},
            )

        # Update running variance estimate (EMA)
        if self.calibration.n_observations > self.warmup_epochs:
            score_dev = (cheap_score - self.calibration.baseline_mean) ** 2
            self.calibration.ema_variance = (
                (1 - self.ema_alpha) * self.calibration.ema_variance
                + self.ema_alpha * score_dev
            )

            # Re-calibrate periodically (every 5 observations)
            if self.calibration.n_observations % 5 == 0:
                if self.c2 is not None and isinstance(self.c2, AdaptiveSensitivityGate):
                    self.c2.calibrate(self.calibration.ema_variance)

        # Delegate to base evaluation with calibrated thresholds
        decision = super().evaluate_update(
            cheap_score=cheap_score,
            full_pass=full_pass,
            scores_before=scores_before,
            scores_after=scores_after,
            update_magnitude=update_magnitude,
        )

        # Annotate with adaptive info
        decision.details["phase"] = "active"
        decision.details["calibrated_e_thresh"] = (
            self.c2.e_thresh if self.c2 else None
        )
        decision.details["calibrated_rv_thresh"] = (
            self.c2.rv_thresh if self.c2 else None
        )
        decision.details["ema_variance"] = self.calibration.ema_variance

        return decision

    def get_calibration_summary(self) -> dict:
        """Return calibration state for logging."""
        return {
            "warmup_complete": self.calibration.warmup_complete,
            "baseline_var": self.calibration.baseline_var,
            "baseline_instability": self.calibration.baseline_instability,
            "calibrated_e_thresh": self.calibration.calibrated_e_thresh,
            "calibrated_rv_thresh": self.calibration.calibrated_rv_thresh,
            "current_ema_var": self.calibration.ema_variance,
            "n_observations": self.calibration.n_observations,
        }

    def get_config_label(self) -> str:
        """Human-readable config label."""
        parts = []
        if self.enable_c1:
            parts.append("C1")
        if self.enable_c2:
            parts.append("C2-adaptive")
        if self.enable_c3:
            parts.append("C3")
        return "+".join(parts) if parts else "none"
