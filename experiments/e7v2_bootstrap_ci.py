"""E7v2 Bootstrap CI Analysis — regression/oscillation 차이의 95% CI 계산."""
import csv
import numpy as np

def load_results(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def bootstrap_ci(baseline_vals, audit_vals, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap CI for (audit - baseline) difference."""
    rng = np.random.RandomState(seed)
    bl = np.array(baseline_vals, dtype=float)
    au = np.array(audit_vals, dtype=float)
    n = len(bl)
    diffs = []
    for _ in range(n_boot):
        idx_bl = rng.randint(0, n, n)
        idx_au = rng.randint(0, n, n)
        diffs.append(np.mean(au[idx_au]) - np.mean(bl[idx_bl]))
    diffs = np.array(diffs)
    alpha = 1 - ci
    lo = np.percentile(diffs, 100 * alpha / 2)
    hi = np.percentile(diffs, 100 * (1 - alpha / 2))
    observed = np.mean(au) - np.mean(bl)
    # p-value: proportion of bootstrap samples where diff >= 0 (for neg diffs)
    if observed < 0:
        p = np.mean(diffs >= 0)
    else:
        p = np.mean(diffs <= 0)
    return observed, lo, hi, p

def analyze(path, model_name):
    rows = load_results(path)
    bl = [r for r in rows if r["mode"] == "baseline"]
    au = [r for r in rows if r["mode"] == "audit"]

    print(f"\n{'='*60}")
    print(f"  {model_name} - Bootstrap 95% CI (10,000 resamples)")
    print(f"{'='*60}")

    for metric in ["mean_accuracy", "oscillations", "regressions"]:
        bl_vals = [float(r[metric]) for r in bl]
        au_vals = [float(r[metric]) for r in au]
        obs, lo, hi, p = bootstrap_ci(bl_vals, au_vals)
        bl_mean = np.mean(bl_vals)
        au_mean = np.mean(au_vals)

        if metric == "regressions" and bl_mean > 0:
            pct = obs / bl_mean * 100
            pct_str = f" ({pct:+.1f}%)"
        elif metric == "oscillations" and bl_mean > 0:
            pct = obs / bl_mean * 100
            pct_str = f" ({pct:+.1f}%)"
        else:
            pct_str = ""

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"\n  {metric}:")
        print(f"    Baseline: {bl_mean:.3f}  |  WhyLab: {au_mean:.3f}")
        print(f"    Δ = {obs:+.3f}{pct_str}")
        print(f"    95% CI: [{lo:+.3f}, {hi:+.3f}]")
        print(f"    p = {p:.4f}  {sig}")

# Gemini
analyze(
    r"d:\00.test\PAPER\WhyLab\experiments\results\e7v2_gemini-2.0-flash_results.csv",
    "Gemini 2.0 Flash"
)
# GPT
analyze(
    r"d:\00.test\PAPER\WhyLab\experiments\results\e7v2_gpt-4o-mini_results.csv",
    "GPT-4o-mini"
)
