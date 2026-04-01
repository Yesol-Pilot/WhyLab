# -*- coding: utf-8 -*-
"""
E7v2 3-Way Comparison: Baseline vs Fixed vs Adaptive
=====================================================
Analyzes the complete E7v2 results with all three conditions.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def analyze_model(path: Path, model_name: str):
    """3-way comparison for one model."""
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return None

    df = pd.read_csv(path)
    modes = df["mode"].unique()
    
    print(f"\n{'='*70}")
    print(f"  {model_name.upper()} - 3-WAY COMPARISON (N={len(df)} rows, modes={list(modes)})")
    print(f"{'='*70}")

    results = {}
    for mode in sorted(modes):
        sub = df[df["mode"] == mode]
        stats = {
            "n": len(sub),
            "acc_mean": sub["mean_accuracy"].mean(),
            "acc_std": sub["mean_accuracy"].std(),
            "osc_mean": sub["oscillations"].mean(),
            "reg_mean": sub["regressions"].mean(),
            "rej_mean": sub["rejections"].mean(),
        }
        results[mode] = stats

    # Print table
    header = f"  {'Metric':<22}"
    for mode in sorted(results.keys()):
        header += f" {mode:<18}"
    print(header)
    print("  " + "-" * (22 + 18 * len(results)))

    for metric, label in [
        ("acc_mean", "Mean Accuracy"),
        ("acc_std", "Acc Std Dev"),
        ("osc_mean", "Oscillations"),
        ("reg_mean", "Regressions"),
        ("rej_mean", "Rejections"),
    ]:
        row = f"  {label:<22}"
        for mode in sorted(results.keys()):
            row += f" {results[mode][metric]:<18.4f}"
        print(row)

    # Deltas vs baseline
    if "baseline" in results:
        bl = results["baseline"]
        print(f"\n  --- Deltas vs Baseline ---")
        for mode in sorted(results.keys()):
            if mode == "baseline":
                continue
            s = results[mode]
            d_acc = s["acc_mean"] - bl["acc_mean"]
            d_osc = s["osc_mean"] - bl["osc_mean"]
            d_reg = s["reg_mean"] - bl["reg_mean"]
            print(f"  {mode:<20}: d_acc={d_acc:+.4f}, d_osc={d_osc:+.1f}, d_reg={d_reg:+.2f}")

    # Bootstrap CI for regression delta
    if "baseline" in results:
        bl_df = df[df["mode"] == "baseline"]
        print(f"\n  --- Bootstrap CI (10,000 iterations) ---")
        for mode in sorted(results.keys()):
            if mode == "baseline":
                continue
            au_df = df[df["mode"] == mode]
            bl_reg = bl_df["regressions"].values
            au_reg = au_df["regressions"].values
            
            n_boot = 10000
            deltas = []
            for _ in range(n_boot):
                sb = np.random.choice(bl_reg, len(bl_reg), replace=True)
                sa = np.random.choice(au_reg, len(au_reg), replace=True)
                deltas.append(sa.mean() - sb.mean())
            
            ci = np.percentile(deltas, [2.5, 97.5])
            p_worse = np.mean([d > 0 for d in deltas])
            p_better = np.mean([d < 0 for d in deltas])
            
            sig = "[SIG]" if ci[1] < 0 or ci[0] > 0 else "[NS]"
            print(f"  {mode:<20}: d_reg CI=[{ci[0]:+.2f}, {ci[1]:+.2f}], "
                  f"P(better)={p_better:.3f}, P(worse)={p_worse:.3f} {sig}")

    # Key question: Does adaptive fix the accuracy cost?
    if "audit" in results and "audit_adaptive" in results:
        fixed_cost = results["audit"]["acc_mean"] - results["baseline"]["acc_mean"]
        adapt_cost = results["audit_adaptive"]["acc_mean"] - results["baseline"]["acc_mean"]
        
        print(f"\n  --- KEY QUESTION: Does adaptive fix the accuracy cost? ---")
        print(f"  Fixed audit accuracy cost:    {fixed_cost:+.4f}")
        print(f"  Adaptive audit accuracy cost: {adapt_cost:+.4f}")
        if abs(adapt_cost) < abs(fixed_cost):
            improvement = (1 - abs(adapt_cost) / abs(fixed_cost)) * 100
            print(f"  [OK] Adaptive reduces accuracy cost by {improvement:.0f}%")
        else:
            print(f"  [WARN] Adaptive does NOT reduce accuracy cost")
        
        # Regression comparison
        fixed_reg = results["audit"]["reg_mean"] - results["baseline"]["reg_mean"]
        adapt_reg = results["audit_adaptive"]["reg_mean"] - results["baseline"]["reg_mean"]
        print(f"  Fixed audit d_reg:    {fixed_reg:+.2f}")
        print(f"  Adaptive audit d_reg: {adapt_reg:+.2f}")

    return results


def main():
    print("=" * 70)
    print("  E7v2 ADAPTIVE C2 - COMPREHENSIVE ANALYSIS")
    print("=" * 70)

    all_results = {}
    
    for model_tag, model_name in [
        ("gemini-2.0-flash", "Gemini 2.0 Flash"),
        ("gpt-4o-mini", "GPT-4o-mini"),
    ]:
        path = RESULTS_DIR / f"e7v2_{model_tag}_results.csv"
        r = analyze_model(path, model_name)
        if r:
            all_results[model_name] = r

    # Cross-model summary
    if len(all_results) >= 2:
        print(f"\n{'='*70}")
        print("  CROSS-MODEL SUMMARY")
        print(f"{'='*70}")
        for model, modes in all_results.items():
            if "baseline" in modes and "audit_adaptive" in modes:
                bl = modes["baseline"]
                ad = modes["audit_adaptive"]
                print(f"  {model}: baseline_var={bl['acc_std']**2:.6f}, "
                      f"d_acc(adaptive)={ad['acc_mean']-bl['acc_mean']:+.4f}, "
                      f"d_reg(adaptive)={ad['reg_mean']-bl['reg_mean']:+.2f}")


if __name__ == "__main__":
    main()
