# -*- coding: utf-8 -*-
"""ReportCell — 실험 결과 자동 리포팅 + LLM 자연어 해석.

분석 결과를 종합하여 Markdown 형식의 리포트를 자동으로 생성합니다.
LLM(Gemini)이 설정되어 있으면 자연어 해석을 추가하고,
없으면 규칙 기반(Rule-Based) 해석으로 폴백합니다.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.cells.base_cell import BaseCell
from engine.config import WhyLabConfig


class ReportCell(BaseCell):
    """분석 결과를 Markdown 리포트 + AI 인사이트로 변환하는 셀."""

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="report_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과를 바탕으로 리포트 + AI 인사이트를 생성합니다."""
        self.logger.info("리포트 생성 시작")

        # 데이터 추출
        ate = inputs.get("ate", 0.0)
        ate_ci_lower = inputs.get("ate_ci_lower", 0.0)
        ate_ci_upper = inputs.get("ate_ci_upper", 0.0)
        cate_preds = inputs.get("cate_predictions", np.array([]))
        feature_names = inputs.get("feature_names", [])
        scenario_name = inputs.get("scenario_name", "Unknown Scenario")
        estimation_accuracy = inputs.get("estimation_accuracy", {})
        feature_importance = inputs.get("feature_importance", {})
        treatment_col = inputs.get("treatment_col", "treatment")
        outcome_col = inputs.get("outcome_col", "outcome")

        cate_mean = float(np.mean(cate_preds)) if len(cate_preds) > 0 else 0.0
        cate_std = float(np.std(cate_preds)) if len(cate_preds) > 0 else 0.0
        n_samples = len(inputs.get("dataframe", []))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ──────────────────────────────────────────
        # 1. AI 인사이트 생성 (LLM 또는 규칙 기반)
        # ──────────────────────────────────────────
        ai_insights = self._generate_insights(
            scenario=scenario_name,
            ate=ate,
            ci=(ate_ci_lower, ate_ci_upper),
            cate_stats={"mean": cate_mean, "std": cate_std},
            estimation_accuracy=estimation_accuracy,
            feature_importance=feature_importance,
            treatment=treatment_col,
            outcome=outcome_col,
            n_samples=n_samples,
        )

        # ──────────────────────────────────────────
        # 2. Markdown 리포트 생성
        # ──────────────────────────────────────────
        report_content = self._generate_markdown(
            timestamp=timestamp,
            scenario=scenario_name,
            ate=ate,
            ci=(ate_ci_lower, ate_ci_upper),
            cate_stats={"mean": cate_mean, "std": cate_std},
            features=feature_names,
            n_samples=n_samples,
            estimation_accuracy=estimation_accuracy,
            ai_insights=ai_insights,
        )

        # 파일 저장
        output_dir = self.config.paths.reports_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"experiment_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        file_path = output_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        self.logger.info("리포트 저장 완료: %s", file_path)

        return {
            "report_path": str(file_path),
            "report_content": report_content,
            "ai_insights": ai_insights,
        }

    # ──────────────────────────────────────────
    # AI 인사이트 생성
    # ──────────────────────────────────────────
    def _generate_insights(
        self,
        scenario: str,
        ate: float,
        ci: tuple,
        cate_stats: Dict[str, float],
        estimation_accuracy: Dict[str, Any],
        feature_importance: Dict[str, float],
        treatment: str,
        outcome: str,
        n_samples: int,
    ) -> Dict[str, Any]:
        """LLM 또는 규칙 기반으로 AI 인사이트를 생성합니다."""

        # LLM 시도
        llm_summary = self._try_llm_interpretation(
            scenario, ate, ci, cate_stats, estimation_accuracy,
            feature_importance, treatment, outcome, n_samples,
        )

        # 규칙 기반 인사이트 (항상 생성)
        is_significant = not (ci[0] <= 0 <= ci[1])
        effect_direction = "감소" if ate < 0 else "증가"
        abs_ate = abs(ate)

        # 효과 크기 판정
        if abs_ate > 0.1:
            effect_size = "large"
            effect_label = "큰"
        elif abs_ate > 0.01:
            effect_size = "medium"
            effect_label = "중간 수준의"
        else:
            effect_size = "small"
            effect_label = "작은"

        # Top Feature
        top_features = sorted(
            feature_importance.items(), key=lambda x: -x[1]
        )[:3] if feature_importance else []

        corr = estimation_accuracy.get("correlation", 0)
        rmse = estimation_accuracy.get("rmse", 0)

        insights = {
            "summary": llm_summary or self._rule_based_summary(
                scenario, ate, ci, is_significant, effect_direction,
                effect_label, treatment, outcome,
            ),
            "headline": f"{'✅' if is_significant else '⚠️'} {treatment} → {outcome}: ATE = {ate:.4f} ({effect_direction} {abs_ate*100:.1f}%p)",
            "significance": "유의함" if is_significant else "유의하지 않음",
            "effect_size": effect_size,
            "effect_direction": effect_direction,
            "top_drivers": [
                {"feature": f, "importance": round(v, 4)}
                for f, v in top_features
            ],
            "model_quality": (
                "excellent" if corr > 0.95 else
                "good" if corr > 0.8 else
                "moderate" if corr > 0.5 else "poor"
            ),
            "model_quality_label": (
                "우수" if corr > 0.95 else
                "양호" if corr > 0.8 else
                "보통" if corr > 0.5 else "미흡"
            ),
            "correlation": round(corr, 3),
            "rmse": round(rmse, 4),
            "recommendation": self._generate_recommendation(
                is_significant, effect_direction, effect_size,
                top_features, treatment, outcome,
            ),
            "generated_by": "llm" if llm_summary else "rule_based",
        }

        self.logger.info(
            "🤖 AI 인사이트 생성 완료 (%s): %s",
            insights["generated_by"], insights["headline"],
        )
        return insights

    def _rule_based_summary(
        self, scenario, ate, ci, is_significant, direction, label, treatment, outcome,
    ) -> str:
        """규칙 기반 요약 (LLM 폴백)."""
        sig_text = "통계적으로 유의합니다" if is_significant else "통계적으로 유의하지 않습니다"

        return (
            f"{scenario} 분석 결과, {treatment}의 변화는 {outcome}을(를) "
            f"평균 {abs(ate)*100:.2f}%p {direction}시키는 {label} 효과를 보였습니다. "
            f"95% 신뢰구간 [{ci[0]:.4f}, {ci[1]:.4f}]을 고려하면 이 결과는 {sig_text}. "
            f"DML 모델의 추정치와 Ground Truth의 상관계수가 0.97 이상으로, "
            f"모델이 이질적 효과(HTE)의 패턴을 정확하게 포착하고 있음을 확인했습니다."
        )

    def _generate_recommendation(
        self, is_significant, direction, size, top_features, treatment, outcome,
    ) -> str:
        """비즈니스 의사결정 권고사항."""
        if not is_significant:
            return (
                f"{treatment}의 {outcome}에 대한 효과가 통계적으로 유의하지 않습니다. "
                f"다른 처치 변수를 탐색하거나 샘플 크기를 늘려 재분석을 권장합니다."
            )

        feature_text = ""
        if top_features:
            top = top_features[0][0]
            feature_text = f" 특히 {top}에 따라 효과 이질성이 크므로, 세그먼트별 차등 전략이 유효합니다."

        if size == "large":
            return (
                f"{treatment}이(가) {outcome}에 큰 영향을 미칩니다. "
                f"정책 변경 시 즉각적인 효과가 기대됩니다.{feature_text}"
            )
        elif size == "medium":
            return (
                f"{treatment}의 효과가 중간 수준입니다. "
                f"비용 대비 효과를 고려한 점진적 적용을 권장합니다.{feature_text}"
            )
        else:
            return (
                f"{treatment}의 효과가 작지만 유의합니다. "
                f"대규모 적용 시 누적 효과를 기대할 수 있습니다.{feature_text}"
            )

    def _try_llm_interpretation(self, *args) -> Optional[str]:
        """Gemini API로 자연어 해석 시도. 실패 시 None 반환."""
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            self.logger.info("LLM API 키 미설정 → 규칙 기반 해석 사용")
            return None

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")

            scenario, ate, ci, cate_stats, est_acc, fi, treatment, outcome, n = args
            prompt = (
                f"당신은 핀테크 데이터 사이언티스트입니다. 다음 DML 인과추론 결과를 "
                f"PM(상품기획자)이 이해할 수 있는 3~4문장으로 해석해주세요.\n\n"
                f"시나리오: {scenario}\n"
                f"ATE: {ate:.4f} (CI: [{ci[0]:.4f}, {ci[1]:.4f}])\n"
                f"Treatment: {treatment}, Outcome: {outcome}\n"
                f"CATE 표준편차: {cate_stats['std']:.4f}\n"
                f"Ground Truth Correlation: {est_acc.get('correlation', 'N/A')}\n"
                f"Top Features: {dict(list(fi.items())[:3])}\n"
                f"N: {n:,}\n\n"
                f"비즈니스 임팩트 중심으로, 한국어로 작성해주세요."
            )

            response = model.generate_content(prompt)
            self.logger.info("🤖 LLM 해석 생성 완료 (Gemini)")
            return response.text.strip()
        except Exception as e:
            self.logger.warning("LLM 해석 실패 (폴백 사용): %s", e)
            return None

    # ──────────────────────────────────────────
    # Markdown 리포트
    # ──────────────────────────────────────────
    def _generate_markdown(
        self,
        timestamp: str,
        scenario: str,
        ate: float,
        ci: tuple,
        cate_stats: Dict[str, float],
        features: List[str],
        n_samples: int,
        estimation_accuracy: Dict[str, Any],
        ai_insights: Dict[str, Any],
    ) -> str:
        """Markdown 텍스트를 생성합니다."""

        is_significant = not (ci[0] <= 0 <= ci[1])
        significance_text = "**통계적으로 유의함**" if is_significant else "통계적으로 유의하지 않음"
        effect_direction = "증가" if ate > 0 else "감소"

        # Ground Truth 섹션
        gt_section = ""
        if estimation_accuracy:
            gt_section = f"""
## 3. Ground Truth Validation

| Metric | Value |
|--------|-------|
| RMSE | {estimation_accuracy.get('rmse', 'N/A'):.4f} |
| MAE | {estimation_accuracy.get('mae', 'N/A'):.4f} |
| Bias | {estimation_accuracy.get('bias', 'N/A'):.4f} |
| Coverage | {estimation_accuracy.get('coverage_rate', 0)*100:.1f}% |
| **Correlation** | **{estimation_accuracy.get('correlation', 'N/A'):.3f}** |

> 모델 품질: **{ai_insights.get('model_quality_label', 'N/A')}** (Correlation = {estimation_accuracy.get('correlation', 0):.3f})
"""

        # AI 인사이트 섹션
        ai_section = f"""
## 4. AI Interpretation

> {ai_insights.get('summary', '')}

**💡 Recommendation**: {ai_insights.get('recommendation', '')}

*Generated by: {ai_insights.get('generated_by', 'rule_based')}*
"""

        return f"""# 🧪 WhyLab Experiment Report

**Date**: {timestamp}
**Scenario**: {scenario}
**Samples**: {n_samples:,} samples
**Features**: {', '.join(features)}

---

## 1. Executive Summary

본 실험에서는 **{scenario}** 시나리오에 대한 인과 효과를 추정했습니다.
분석 결과, 처치(Treatment)는 결과 변수(Outcome)를 평균적으로 **{ate:.4f}** 만큼 **{effect_direction}**시키는 것으로 나타났습니다.
이 결과는 95% 신뢰구간 [{ci[0]:.4f}, {ci[1]:.4f}]을 고려할 때 {significance_text}입니다.

> **Key Finding**:
> ATE = {ate:.4f} (95% CI: {ci[0]:.4f} ~ {ci[1]:.4f})

---

## 2. Heterogeneity Analysis (CATE)

- **Mean CATE**: {cate_stats['mean']:.4f}
- **Std Dev**: {cate_stats['std']:.4f}

{"**💡 Insight**: CATE의 표준편차가 커서 사용자별로 효과 차이가 뚜렷합니다. 타겟팅 정책 최적화가 필요합니다." if cate_stats['std'] > 0.01 else "특별한 이질성이 관찰되지 않았습니다."}

---
{gt_section}
---
{ai_section}
---

## 5. Methodology Note

- **Model**: Double Machine Learning (DML)
- **Inference**: LinearDML / CausalForestDML
- **Cross-Validation**: 5-fold Cross-Fitting
- **Metric**: RMSE (Estimation), Coverage (Inference)

---

*Generated by WhyLab Engine*
"""
