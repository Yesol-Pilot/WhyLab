# -*- coding: utf-8 -*-
"""
Phase 1-2: E7v2 GPT-4o-mini Adverse Effect Diagnosis
=====================================================
Root-cause analysis of why the audit layer WORSENS performance
on GPT-4o-mini while improving Gemini 2.0 Flash.

Core hypothesis: Fixed C2 thresholds over-reject when baseline
is already stable (low variance), causing unnecessary performance loss.

Usage:
    python -m experiments.e7v2_diagnosis
"""
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_e7v2_results():
    """Load both Gemini and GPT-4o-mini E7v2 results."""
    gemini_path = RESULTS_DIR / "e7v2_gemini-2.0-flash_results.csv"
    gpt_path = RESULTS_DIR / "e7v2_gpt-4o-mini_results.csv"

    dfs = {}
    if gemini_path.exists():
        dfs["gemini"] = pd.read_csv(gemini_path)
    if gpt_path.exists():
        dfs["gpt4o-mini"] = pd.read_csv(gpt_path)
    return dfs


def compute_model_statistics(df: pd.DataFrame, model_name: str) -> dict:
    """Compute per-model baseline and audit statistics."""
    baseline = df[df["mode"] == "baseline"]
    audited = df[df["mode"] == "audit"]

    bl_stats = {
        "model": model_name,
        "bl_mean_acc": baseline["mean_accuracy"].mean(),
        "bl_std_acc": baseline["mean_accuracy"].std(),
        "bl_variance": baseline["mean_accuracy"].var(),
        "bl_mean_osc": baseline["oscillations"].mean(),
        "bl_mean_reg": baseline["regressions"].mean(),
        "bl_instability": baseline["regressions"].mean() + baseline["oscillations"].mean(),
    }

    au_stats = {
        "au_mean_acc": audited["mean_accuracy"].mean(),
        "au_std_acc": audited["mean_accuracy"].std(),
        "au_mean_osc": audited["oscillations"].mean(),
        "au_mean_reg": audited["regressions"].mean(),
        "au_mean_rej": audited["rejections"].mean(),
    }

    # Deltas
    deltas = {
        "delta_acc": au_stats["au_mean_acc"] - bl_stats["bl_mean_acc"],
        "delta_osc": au_stats["au_mean_osc"] - bl_stats["bl_mean_osc"],
        "delta_reg": au_stats["au_mean_reg"] - bl_stats["bl_mean_reg"],
    }

    # Bootstrap CI for regression delta
    n_boot = 10000
    bl_reg = baseline["regressions"].values
    au_reg = audited["regressions"].values
    boot_deltas = []
    for _ in range(n_boot):
        sb = np.random.choice(bl_reg, len(bl_reg), replace=True)
        sa = np.random.choice(au_reg, len(au_reg), replace=True)
        boot_deltas.append(sa.mean() - sb.mean())
    ci = np.percentile(boot_deltas, [2.5, 97.5])
    p_value = np.mean([d >= 0 for d in boot_deltas])  # one-sided: audit worse?

    ci_stats = {
        "delta_reg_ci_lo": ci[0],
        "delta_reg_ci_hi": ci[1],
        "p_audit_worse": p_value,
    }

    return {**bl_stats, **au_stats, **deltas, **ci_stats}


def analyze_rejection_patterns(df: pd.DataFrame, model_name: str):
    """Analyze when audit rejects and whether those rejections help."""
    audited = df[df["mode"] == "audit"]
    baseline = df[df["mode"] == "baseline"]

    print(f"\n{'='*60}")
    print(f"  REJECTION PATTERN ANALYSIS: {model_name}")
    print(f"{'='*60}")

    for _, row in audited.iterrows():
        seed = row["seed"]
        bl_row = baseline[baseline["seed"] == seed].iloc[0]

        rej_label = "⚠️ OVER" if row["rejections"] > 10 else "  OK"
        worse = "🔴" if row["regressions"] > bl_row["regressions"] else "🟢"

        print(f"  seed={seed}: rej={int(row['rejections']):2d} {rej_label} | "
              f"reg: {int(bl_row['regressions'])}→{int(row['regressions'])} {worse} | "
              f"osc: {int(bl_row['oscillations'])}→{int(row['oscillations'])} | "
              f"acc: {bl_row['mean_accuracy']:.3f}→{row['mean_accuracy']:.3f}")


def compute_condition_aware_metrics(all_stats: list[dict]):
    """Test the condition-aware activation hypothesis:
    Does audit improvement correlate with baseline instability?"""

    print(f"\n{'='*60}")
    print("  CONDITION-AWARE ACTIVATION ANALYSIS")
    print(f"{'='*60}")

    for s in all_stats:
        benefit = -(s["delta_reg"])  # positive = audit reduces regressions
        instability = s["bl_instability"]
        print(f"  {s['model']:>12}: instability={instability:.1f}, "
              f"reg_reduction={benefit:+.1f}, "
              f"acc_cost={s['delta_acc']:+.3f}, "
              f"osc_change={s['delta_osc']:+.1f}")

    # The key insight:
    instabilities = [s["bl_instability"] for s in all_stats]
    benefits = [-s["delta_reg"] for s in all_stats]

    if len(instabilities) >= 2:
        corr = np.corrcoef(instabilities, benefits)[0, 1]
        print(f"\n  Correlation(instability, reg_reduction) = {corr:.3f}")
        if corr > 0.5:
            print("  ✅ CONFIRMED: Audit benefit is proportional to baseline instability")
        elif corr < -0.5:
            print("  ⚠️ INVERTED: Audit is counterproductive on unstable baselines")
        else:
            print("  ⚪ WEAK: No clear relationship (need more data points)")


def recommend_adaptive_thresholds(all_stats: list[dict]):
    """Derive recommended adaptive threshold logic."""
    print(f"\n{'='*60}")
    print("  ADAPTIVE THRESHOLD RECOMMENDATION")
    print(f"{'='*60}")

    for s in all_stats:
        bl_var = s["bl_variance"]
        # Proposed: scale E-value threshold inversely with variance
        # High variance → low threshold (more filtering)
        # Low variance → high threshold (less filtering)
        proposed_e = max(1.2, 2.0 / (1 + 10 * bl_var))
        proposed_rv = max(0.02, 0.1 / (1 + 10 * bl_var))

        print(f"  {s['model']:>12}: bl_var={bl_var:.6f}")
        print(f"    Current:  E_thresh=1.5, RV_thresh=0.05")
        print(f"    Proposed: E_thresh={proposed_e:.3f}, RV_thresh={proposed_rv:.4f}")
        print(f"    Rejections expected: {'↓ (fewer, fixes over-rejection)' if bl_var < 0.001 else '~ (similar)'}")


def main():
    dfs = load_e7v2_results()

    if not dfs:
        print("[ERROR] No E7v2 result files found.")
        return

    all_stats = []
    for model_name, df in dfs.items():
        stats = compute_model_statistics(df, model_name)
        all_stats.append(stats)

        print(f"\n{'='*60}")
        print(f"  {model_name.upper()} — SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"  Baseline: acc={stats['bl_mean_acc']:.3f}±{stats['bl_std_acc']:.3f}, "
              f"var={stats['bl_variance']:.6f}")
        print(f"            osc={stats['bl_mean_osc']:.1f}, reg={stats['bl_mean_reg']:.1f}, "
              f"instability={stats['bl_instability']:.1f}")
        print(f"  Audited:  acc={stats['au_mean_acc']:.3f}±{stats['au_std_acc']:.3f}")
        print(f"            osc={stats['au_mean_osc']:.1f}, reg={stats['au_mean_reg']:.1f}, "
              f"rej={stats['au_mean_rej']:.1f}")
        print(f"  Delta:    acc={stats['delta_acc']:+.3f}, osc={stats['delta_osc']:+.1f}, "
              f"reg={stats['delta_reg']:+.1f}")
        print(f"  CI(reg):  [{stats['delta_reg_ci_lo']:+.2f}, {stats['delta_reg_ci_hi']:+.2f}], "
              f"P(audit worse)={stats['p_audit_worse']:.3f}")

        analyze_rejection_patterns(df, model_name)

    compute_condition_aware_metrics(all_stats)
    recommend_adaptive_thresholds(all_stats)

    # Save aggregated diagnostics
    out = pd.DataFrame(all_stats)
    out_path = RESULTS_DIR / "e7v2_diagnosis.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
