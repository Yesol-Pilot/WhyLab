# -*- coding: utf-8 -*-
"""DaV 정량 평가 파이프라인.

DaV(Debate-as-Verification) 프로토콜의 판결 품질을 정량적으로 평가합니다.

평가 지표:
  1. 판결 정확도 (Accuracy): Ground Truth 대비 VERIFIED/REFUTED 일치 비율
  2. 판결 신뢰도 보정 (Calibration): 신뢰도 vs 실제 정확도 정렬
  3. 증거 커버리지 (Evidence Coverage): 수집된 증거 유형 다양성
  4. 교차 심문 품질 (Cross-Exam Quality): 찬반 균형

사용법:
    python scripts/evaluate_dav.py
    python scripts/evaluate_dav.py --scenarios 20
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.agents.dav_protocol import DaVProtocol, DaVVerdict
from engine.config import WhyLabConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("dav_eval")


# ──────────────────────────────────────────────
# 평가 시나리오 정의
# ──────────────────────────────────────────────

@dataclass
class EvalScenario:
    """DaV 평가 시나리오."""
    name: str
    description: str
    ground_truth: str  # "VERIFIED" | "REFUTED" | "INCONCLUSIVE"
    context: Dict[str, Any]


def generate_eval_scenarios(n_scenarios: int = 10) -> List[EvalScenario]:
    """다양한 평가 시나리오를 생성합니다.

    참(VERIFIED) / 거짓(REFUTED) 시나리오를 혼합합니다.
    """
    rng = np.random.RandomState(42)
    scenarios = []

    for i in range(n_scenarios):
        is_true_effect = i % 2 == 0  # 짝수: 참, 홀수: 거짓

        if is_true_effect:
            # 참 효과: 강한 ATE, 좁은 CI, 높은 통계적 유의성
            ate = rng.uniform(0.3, 0.8)
            ci_lower = ate - rng.uniform(0.05, 0.15)
            ci_upper = ate + rng.uniform(0.05, 0.15)
            e_val = float(rng.uniform(3, 10))
            placebo_passed = True
            ground_truth = "VERIFIED"
        else:
            # 거짓 효과: 약한 ATE, 넓은 CI, 낮은 유의성
            ate = rng.uniform(-0.1, 0.1)
            ci_lower = ate - rng.uniform(0.2, 0.5)
            ci_upper = ate + rng.uniform(0.2, 0.5)
            e_val = float(rng.uniform(1.0, 1.5))
            placebo_passed = rng.random() < 0.2
            ground_truth = "REFUTED"

        # DaVProtocol이 기대하는 context 키 구조:
        #   - ate: dict {point_estimate, ci_lower, ci_upper}
        #   - sensitivity: dict {e_value: {point}}
        #   - refutation: dict {placebo: {passed}}
        #   - meta_learners: dict {learner_name: {ate}}
        #   - dag_edges: list[tuple]
        context = {
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            # _construct_claim이 읽는 키
            "ate": {
                "point_estimate": float(ate),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
            },
            # _collect_evidence: 증거 3 (E-value)
            "sensitivity": {
                "e_value": {"point": e_val},
            },
            # _collect_evidence: 증거 4 (위약 검정)
            "refutation": {
                "placebo": {"passed": placebo_passed},
            },
            # _collect_evidence: 증거 2 (메타러너 합의)
            "meta_learners": {
                "S-Learner": {"ate": float(ate + rng.uniform(-0.05, 0.05))},
                "T-Learner": {"ate": float(ate + rng.uniform(-0.05, 0.05))},
                "X-Learner": {"ate": float(ate + rng.uniform(-0.05, 0.05))},
            },
            # _collect_evidence: 증거 6 (인과 그래프)
            "dag_edges": [("treatment", "outcome")] if is_true_effect else [],
            "feature_names": [f"x{j}" for j in range(5)],
        }

        scenarios.append(EvalScenario(
            name=f"scenario_{i+1:02d}",
            description=f"{'참 효과' if is_true_effect else '거짓 효과'} 시나리오",
            ground_truth=ground_truth,
            context=context,
        ))

    return scenarios


# ──────────────────────────────────────────────
# 평가 메트릭
# ──────────────────────────────────────────────

def evaluate_verdicts(
    scenarios: List[EvalScenario],
    verdicts: List[DaVVerdict],
) -> Dict[str, Any]:
    """DaV 판결 결과를 정량적으로 평가합니다."""

    correct = 0
    total = len(scenarios)
    confidences = []
    correct_confidences = []
    wrong_confidences = []
    evidence_counts = []
    cross_exam_balances = []

    for scenario, verdict in zip(scenarios, verdicts):
        gt = scenario.ground_truth
        pred = verdict.verdict

        # 1. 정확도
        is_correct = (
            (pred == "VERIFIED" and gt == "VERIFIED") or
            (pred == "REFUTED" and gt == "REFUTED") or
            (pred == "INCONCLUSIVE" and gt == "INCONCLUSIVE")
        )
        if is_correct:
            correct += 1
            correct_confidences.append(verdict.confidence)
        else:
            wrong_confidences.append(verdict.confidence)

        confidences.append(verdict.confidence)

        # 2. 증거 커버리지
        evidence_sources = set(e.source for e in verdict.evidence_chain)
        evidence_counts.append(len(evidence_sources))

        # 3. 교차 심문 균형
        if verdict.cross_examination:
            pro_count = sum(
                1 for r in verdict.cross_examination
                if r.agent in ("Advocate", "Growth Hacker")
            )
            con_count = sum(
                1 for r in verdict.cross_examination
                if r.agent in ("Critic", "Risk Manager")
            )
            balance = min(pro_count, con_count) / max(pro_count, con_count, 1)
            cross_exam_balances.append(balance)

    accuracy = correct / max(total, 1)

    # 보정 지표: 정답의 평균 신뢰도 vs 오답의 평균 신뢰도
    avg_correct_conf = np.mean(correct_confidences) if correct_confidences else 0.0
    avg_wrong_conf = np.mean(wrong_confidences) if wrong_confidences else 0.0
    calibration_gap = avg_correct_conf - avg_wrong_conf  # 양수일수록 잘 보정됨

    return {
        "accuracy": float(accuracy),
        "correct_count": correct,
        "total_count": total,
        "avg_confidence": float(np.mean(confidences)),
        "calibration": {
            "avg_correct_confidence": float(avg_correct_conf),
            "avg_wrong_confidence": float(avg_wrong_conf),
            "calibration_gap": float(calibration_gap),
        },
        "evidence_coverage": {
            "avg_source_types": float(np.mean(evidence_counts)),
            "min_source_types": int(min(evidence_counts)) if evidence_counts else 0,
            "max_source_types": int(max(evidence_counts)) if evidence_counts else 0,
        },
        "cross_exam_balance": {
            "avg_balance": float(np.mean(cross_exam_balances)) if cross_exam_balances else 0.0,
        },
    }


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="DaV 정량 평가")
    parser.add_argument("--scenarios", type=int, default=10, help="평가 시나리오 수")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DaV 정량 평가 시작 (%d 시나리오)", args.scenarios)
    logger.info("=" * 60)

    # 시나리오 생성
    scenarios = generate_eval_scenarios(args.scenarios)

    # DaV 프로토콜 실행
    dav = DaVProtocol()
    verdicts = []

    for i, scenario in enumerate(scenarios):
        logger.info("  [%d/%d] %s: %s", i + 1, len(scenarios), scenario.name, scenario.description)
        verdict = dav.verify(scenario.context)
        verdicts.append(verdict)
        logger.info(
            "    -> %s (conf=%.2f, GT=%s)",
            verdict.verdict, verdict.confidence, scenario.ground_truth,
        )

    # 평가
    metrics = evaluate_verdicts(scenarios, verdicts)

    # 결과 저장
    output_dir = PROJECT_ROOT / "paper" / "reports" / "dav_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = {
        "meta": {
            "n_scenarios": args.scenarios,
            "timestamp": timestamp,
        },
        "metrics": metrics,
        "details": [
            {
                "scenario": s.name,
                "ground_truth": s.ground_truth,
                "predicted": v.verdict,
                "confidence": v.confidence,
                "evidence_count": len(v.evidence_chain),
                "correct": (
                    (v.verdict == s.ground_truth) or
                    (v.verdict in ("VERIFIED", "REFUTED") and s.ground_truth == v.verdict)
                ),
            }
            for s, v in zip(scenarios, verdicts)
        ],
    }

    json_path = output_dir / f"dav_eval_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # latest
    latest_path = output_dir / "dav_eval_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("결과 저장: %s", json_path)

    # 요약 출력
    print("\n" + "=" * 60)
    print("DaV 정량 평가 결과")
    print("=" * 60)
    print(f"  정확도: {metrics['accuracy']:.1%} ({metrics['correct_count']}/{metrics['total_count']})")
    print(f"  평균 신뢰도: {metrics['avg_confidence']:.3f}")
    print(f"  보정 격차: {metrics['calibration']['calibration_gap']:.3f}")
    print(f"  증거 커버리지: 평균 {metrics['evidence_coverage']['avg_source_types']:.1f} 유형")
    print(f"  교차심문 균형: {metrics['cross_exam_balance']['avg_balance']:.2f}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
