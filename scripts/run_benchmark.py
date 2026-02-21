# -*- coding: utf-8 -*-
"""벤치마크 실행 스크립트 — DragonNet/TARNet 포함.

6종 벤치마크에서 메타러너 + DragonNet + TARNet + LinearDML을 비교합니다.
"""
import sys
import os
import logging
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from engine.data.benchmark_data import BENCHMARK_REGISTRY
from engine.cells.meta_learner_cell import (
    SLearner, TLearner, XLearner, DRLearner, RLearner,
)

# DeepCATECell 선택적 임포트
try:
    from engine.cells.deep_cate_cell import DeepCATECell, DeepCATEConfig
    HAS_DEEP = True
except ImportError:
    HAS_DEEP = False
    print("WARNING: DeepCATECell not found -- DragonNet/TARNet skipped")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def sqrt_pehe(tau_hat, tau_true):
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


def ate_bias(tau_hat, tau_true):
    return float(np.abs(np.mean(tau_hat) - np.mean(tau_true)))


def run_benchmark():
    # 소규모 n으로 빠른 실행 (Criteo는 5000으로 제한)
    dataset_configs = {
        "ihdp": {"n_reps": 5},
        "acic": {"n_reps": 3},
        "jobs": {"n_reps": 5},
        "twins": {"n_reps": 3},
        "criteo": {"n_reps": 2},
        "lalonde": {"n_reps": 3},
    }

    # 메타러너 레지스트리
    LEARNERS = {
        "S-Learner": SLearner,
        "T-Learner": TLearner,
        "X-Learner": XLearner,
        "DR-Learner": DRLearner,
        "R-Learner": RLearner,
    }

    # 딥러닝 아키텍처
    DEEP_ARCHS = {}
    if HAS_DEEP:
        DEEP_ARCHS = {"DragonNet": "dragonnet", "TARNet": "tarnet"}

    all_results = {}
    start = time.time()

    for ds_name, ds_cfg in dataset_configs.items():
        if ds_name not in BENCHMARK_REGISTRY:
            continue

        loader = BENCHMARK_REGISTRY[ds_name]()
        n_reps = ds_cfg["n_reps"]

        print(f"\n{'='*60}")
        print(f"📊 {ds_name.upper()} (반복 {n_reps}회)")
        print(f"{'='*60}")

        metrics = {name: {"pehe": [], "ate_bias": []}
                   for name in list(LEARNERS) + list(DEEP_ARCHS)}

        for rep in range(n_reps):
            data = loader.load(seed=42 + rep)

            # 일반 메타러너
            for name, Cls in LEARNERS.items():
                try:
                    from engine.config import WhyLabConfig
                    learner = Cls(config=WhyLabConfig())
                    learner.fit(data.X, data.T, data.Y)
                    tau = learner.predict_cate(data.X)
                    metrics[name]["pehe"].append(sqrt_pehe(tau, data.tau_true))
                    metrics[name]["ate_bias"].append(ate_bias(tau, data.tau_true))
                except Exception as e:
                    metrics[name]["pehe"].append(float("nan"))
                    metrics[name]["ate_bias"].append(float("nan"))

            # 딥러닝 CATE
            for name, arch in DEEP_ARCHS.items():
                try:
                    deep_cfg = DeepCATEConfig(
                        architecture=arch,
                        shared_dims=(64, 32),
                        head_dims=(32,),
                        epochs=100,
                        batch_size=min(64, len(data.X)),
                        use_gpu=True,
                    )
                    cell = DeepCATECell(deep_config=deep_cfg)
                    cell.fit(data.X, data.T, data.Y)
                    tau = cell.predict_cate(data.X)
                    metrics[name]["pehe"].append(sqrt_pehe(tau, data.tau_true))
                    metrics[name]["ate_bias"].append(ate_bias(tau, data.tau_true))
                except Exception as e:
                    print(f"  ⚠️ {name} (rep={rep}) 실패: {e}")
                    metrics[name]["pehe"].append(float("nan"))
                    metrics[name]["ate_bias"].append(float("nan"))

            print(f"  ✅ Rep {rep+1}/{n_reps}")

        ds_results = {}
        for name, m in metrics.items():
            pa = np.array(m["pehe"])
            ba = np.array(m["ate_bias"])
            ds_results[name] = {
                "pehe_mean": float(np.nanmean(pa)),
                "pehe_std": float(np.nanstd(pa)),
                "ate_bias_mean": float(np.nanmean(ba)),
                "ate_bias_std": float(np.nanstd(ba)),
            }
            print(f"  {name:14s}: √PEHE={np.nanmean(pa):.4f}±{np.nanstd(pa):.4f}  "
                  f"ATE Bias={np.nanmean(ba):.4f}±{np.nanstd(ba):.4f}")

        all_results[ds_name] = ds_results

    elapsed = time.time() - start
    print(f"\n⏱️ 총 소요 시간: {elapsed:.1f}초")

    # 마크다운 테이블 생성
    print("\n\n## 벤치마크 비교표\n")
    ds_names = list(all_results.keys())
    header = "| Method |"
    sep = "|---|"
    for ds in ds_names:
        header += f" {ds.upper()} √PEHE |"
        sep += "---|"

    print(header)
    print(sep)

    ordered = list(LEARNERS) + list(DEEP_ARCHS)
    for method in ordered:
        row = f"| {method} |"
        for ds in ds_names:
            if ds in all_results and method in all_results[ds]:
                r = all_results[ds][method]
                row += f" {r['pehe_mean']:.4f}±{r['pehe_std']:.4f} |"
            else:
                row += " — |"
        print(row)

    # 결과 저장
    report_dir = ROOT / "paper" / "reports" / "benchmarks"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "benchmark_dragonnet.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# WhyLab 벤치마크 — DragonNet/TARNet 포함\n\n")
        f.write(f"실행 시간: {elapsed:.1f}초\n\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for method in ordered:
            row = f"| {method} |"
            for ds in ds_names:
                if ds in all_results and method in all_results[ds]:
                    r = all_results[ds][method]
                    row += f" {r['pehe_mean']:.4f}±{r['pehe_std']:.4f} |"
                else:
                    row += " — |"
            f.write(row + "\n")

    print(f"\n📄 결과 저장: {report_path}")


if __name__ == "__main__":
    run_benchmark()
