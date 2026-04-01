# -*- coding: utf-8 -*-
"""
Cross-Environment Condition-Aware Activation Analysis
======================================================
Unifies results from E5, E6, and E7v2 to validate the core principle:
"Audit benefit is proportional to baseline instability."

This generates data for the key paper figure showing the correlation
between baseline instability and WhyLab improvement across ALL
experimental conditions.

Usage:
    python -m experiments.condition_aware_analysis
"""
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def extract_e5_datapoints() -> list[dict]:
    """Extract (instability, benefit) from E5 SWE-bench results."""
    path = RESULTS_DIR / "e5_summary.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    baseline = df[df["ablation"] == "none"].iloc[0]
    c2_cal = df[df["ablation"] == "C2_calibrated"].iloc[0]

    # SWE-bench: very low instability (zero regressions in baseline)
    instability = baseline["mean_oscillation"]  # 0.0401
    osc_benefit = baseline["mean_oscillation"] - c2_cal["mean_oscillation"]
    reg_benefit = baseline["mean_regressions"] - c2_cal["mean_regressions"]

    return [{
        "env": "E5 (SWE-bench)",
        "condition": "Gemini 2.0 Flash, static",
        "baseline_instability": instability,
        "osc_reduction": osc_benefit,
        "reg_reduction": reg_benefit,
        "acc_cost": c2_cal["pass_rate"] - baseline["pass_rate"],
        "marker": "E5",
    }]


def extract_e6_datapoints() -> list[dict]:
    """Extract (instability, benefit) from E6 non-stationary results."""
    path = RESULTS_DIR / "e6_summary.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    points = []

    for lr in [0.1, 0.5]:
        for h_rate in [0.0, 0.3, 0.5]:
            subset = df[(df["lr_base"] == lr) & (df["h_rate"] == h_rate)]
            if subset.empty:
                continue

            baseline_row = subset[subset["ablation"] == "none"]
            full_row = subset[subset["ablation"] == "full"]

            if baseline_row.empty or full_row.empty:
                continue

            bl = baseline_row.iloc[0]
            fu = full_row.iloc[0]

            instability = bl["osc_index"] + bl["final_energy"] / 100
            osc_benefit = bl["osc_index"] - fu["osc_index"]
            energy_benefit = bl["final_energy"] - fu["final_energy"]
            regret_benefit = bl["regret"] - fu["regret"]

            points.append({
                "env": f"E6 (η={lr}, h={h_rate})",
                "condition": f"Non-stationary, lr={lr}, halluc={h_rate}",
                "baseline_instability": instability,
                "osc_reduction": osc_benefit,
                "reg_reduction": 0,  # E6 doesn't track regressions directly
                "energy_reduction": energy_benefit,
                "regret_reduction": regret_benefit,
                "acc_cost": 0,
                "marker": "E6",
            })

    return points


def extract_e7v2_datapoints() -> list[dict]:
    """Extract (instability, benefit) from E7v2 adversarial results."""
    points = []

    for model_tag, model_name in [
        ("gemini-2.0-flash", "Gemini 2.0 Flash"),
        ("gpt-4o-mini", "GPT-4o-mini"),
    ]:
        path = RESULTS_DIR / f"e7v2_{model_tag}_results.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path)
        baseline = df[df["mode"] == "baseline"]
        audited = df[df["mode"] == "audit"]

        bl_instability = baseline["regressions"].mean() + baseline["oscillations"].mean()
        osc_benefit = baseline["oscillations"].mean() - audited["oscillations"].mean()
        reg_benefit = baseline["regressions"].mean() - audited["regressions"].mean()
        acc_cost = audited["mean_accuracy"].mean() - baseline["mean_accuracy"].mean()

        points.append({
            "env": f"E7v2 ({model_name})",
            "condition": f"Adversarial fact-tracking, {model_name}",
            "baseline_instability": bl_instability,
            "osc_reduction": osc_benefit,
            "reg_reduction": reg_benefit,
            "acc_cost": acc_cost,
            "marker": "E7v2",
        })

    return points


def compute_correlation_analysis(points: list[dict]):
    """Compute and display the condition-aware correlation."""
    df = pd.DataFrame(points)

    print("=" * 70)
    print("  CONDITION-AWARE ACTIVATION: CROSS-ENVIRONMENT ANALYSIS")
    print("=" * 70)
    print()

    # Display all data points
    print(f"  {'Environment':<35} {'Instab.':<10} {'ΔOsc':<10} {'ΔReg':<10} {'ΔAcc':<10}")
    print("  " + "-" * 65)
    for _, row in df.iterrows():
        print(f"  {row['env']:<35} {row['baseline_instability']:<10.3f} "
              f"{row.get('osc_reduction', 0):<+10.3f} "
              f"{row.get('reg_reduction', 0):<+10.3f} "
              f"{row.get('acc_cost', 0):<+10.4f}")

    print()

    # Correlation analysis
    valid = df[df["baseline_instability"].notna() & df["osc_reduction"].notna()]
    if len(valid) >= 3:
        instab = valid["baseline_instability"].values
        osc_red = valid["osc_reduction"].values

        corr = np.corrcoef(instab, osc_red)[0, 1]
        print(f"  Spearman correlation(instability, osc_reduction):")

        from scipy import stats as scipy_stats
        try:
            rho, p = scipy_stats.spearmanr(instab, osc_red)
            print(f"    ρ = {rho:.3f}, p = {p:.4f}")
        except Exception:
            print(f"    Pearson r = {corr:.3f} (scipy not available for Spearman)")

        # Linear regression
        if len(instab) >= 3:
            slope, intercept = np.polyfit(instab, osc_red, 1)
            print(f"    Linear fit: osc_reduction = {slope:.3f} * instability + {intercept:.3f}")
            print(f"    Interpretation: {slope:.1f}x more oscillation reduction per unit instability")

    print()

    # Key finding summary
    print("  KEY FINDING:")
    if len(valid) >= 2:
        low_instab = valid[valid["baseline_instability"] < valid["baseline_instability"].median()]
        high_instab = valid[valid["baseline_instability"] >= valid["baseline_instability"].median()]

        if not low_instab.empty and not high_instab.empty:
            lo_benefit = low_instab["osc_reduction"].mean()
            hi_benefit = high_instab["osc_reduction"].mean()
            print(f"    Low instability group:  mean osc_reduction = {lo_benefit:+.3f}")
            print(f"    High instability group: mean osc_reduction = {hi_benefit:+.3f}")
            if hi_benefit > lo_benefit:
                print("    ✅ Audit benefit increases with baseline instability")
            else:
                print("    ⚠️ Audit benefit does NOT increase with instability")

    return df


def generate_figure_data(df: pd.DataFrame):
    """Save data for the paper figure."""
    out_path = RESULTS_DIR / "condition_aware_figure_data.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Figure data saved: {out_path}")
    print("  → Use this for Figure: 'Condition-Aware Activation'")
    print("    X: baseline_instability, Y: osc_reduction or reg_reduction")
    print("    Color by: marker (E5/E6/E7v2)")


def main():
    all_points = []
    all_points.extend(extract_e5_datapoints())
    all_points.extend(extract_e6_datapoints())
    all_points.extend(extract_e7v2_datapoints())

    if not all_points:
        print("[ERROR] No results found. Run experiments first.")
        return

    print(f"\n  Collected {len(all_points)} data points across all experiments\n")

    df = compute_correlation_analysis(all_points)
    generate_figure_data(df)


if __name__ == "__main__":
    main()
