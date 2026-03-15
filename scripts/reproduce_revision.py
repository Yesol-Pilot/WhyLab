"""
WhyLab Revision Reproduction Script
=====================================
One-click script to reproduce all revision-related experiments
and generate paper artifacts (tables, figures).

Usage:
    python scripts/reproduce_revision.py          # all stages
    python scripts/reproduce_revision.py --stage p0   # P0 only
    python scripts/reproduce_revision.py --stage p1   # P1 only

Output:
    experiments/results/e2_refutation*.csv
    experiments/results/e3a_ablation*.csv
    paper/tables/*.tex
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
PAPER_TABLES = ROOT / "paper" / "tables"


def run_script(name, script_path):
    """Run a Python script and report success/failure."""
    print(f"\n{'='*60}")
    print(f"  [{name}] {script_path.name}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"  ✅ PASS ({elapsed:.1f}s)")
        # Show last 5 lines of output
        lines = result.stdout.strip().split("\n")
        for line in lines[-5:]:
            print(f"    {line}")
    else:
        print(f"  ❌ FAIL (exit {result.returncode}, {elapsed:.1f}s)")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-10:]:
                print(f"    {line}")

    return result.returncode == 0


def stage_p0():
    """P0: Core revision experiments (W1 + W2)."""
    print("\n" + "=" * 60)
    print("  STAGE P0: Core Revision Experiments")
    print("=" * 60)

    ok = True
    # W1: Refutation
    ok &= run_script("P0-W1", EXPERIMENTS / "e2_refutation.py")
    # W2: Ablation
    ok &= run_script("P0-W2", EXPERIMENTS / "e3a_ablation.py")
    return ok


def stage_p1():
    """P1: Evidence packaging (W4 + W3)."""
    print("\n" + "=" * 60)
    print("  STAGE P1: Evidence Packaging")
    print("=" * 60)

    ok = True
    # W4: Aggregate stats → LaTeX tables
    ok &= run_script("P1-W4", EXPERIMENTS / "aggregate_stats.py")
    # W3: Invariance check
    ok &= run_script("P1-W3", EXPERIMENTS / "invariance_check.py")
    return ok


def verify_outputs():
    """Check that expected output files exist."""
    print("\n" + "=" * 60)
    print("  OUTPUT VERIFICATION")
    print("=" * 60)

    expected = [
        EXPERIMENTS / "results" / "e2_refutation.csv",
        EXPERIMENTS / "results" / "e2_refutation_summary.csv",
        EXPERIMENTS / "results" / "e3a_ablation_t1.csv",
        EXPERIMENTS / "results" / "e3a_ablation_t2.csv",
        EXPERIMENTS / "results" / "e3a_ablation_t1_summary.csv",
        EXPERIMENTS / "results" / "e3a_ablation_t2_summary.csv",
        EXPERIMENTS / "results" / "invariance_check.csv",
        PAPER_TABLES / "e2_refutation.tex",
        PAPER_TABLES / "e3a_ablation.tex",
        PAPER_TABLES / "invariance.tex",
    ]

    all_ok = True
    for f in expected:
        exists = f.exists()
        size = f.stat().st_size if exists else 0
        status = f"✅ {size:,} bytes" if exists else "❌ MISSING"
        print(f"  {f.relative_to(ROOT)}  {status}")
        all_ok &= exists

    return all_ok


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WhyLab revision reproduction")
    parser.add_argument("--stage", choices=["p0", "p1", "all"], default="all")
    args = parser.parse_args()

    t_start = time.time()
    ok = True

    if args.stage in ("p0", "all"):
        ok &= stage_p0()
    if args.stage in ("p1", "all"):
        ok &= stage_p1()

    ok &= verify_outputs()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    if ok:
        print(f"  🎉 ALL PASSED ({elapsed:.1f}s)")
    else:
        print(f"  ⚠️  SOME FAILURES ({elapsed:.1f}s)")
    print(f"{'='*60}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
