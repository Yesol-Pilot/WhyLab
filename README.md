# WhyLab: Causal Audit Framework for Stable Agent Self-Improvement

> Anonymous code for NeurIPS 2026 submission

Self-improving AI agents lack runtime safeguards that prevent evaluation drift, fragile outcome acceptance, and unbounded parameter updates from compounding into catastrophic policy degradation. **WhyLab** introduces a causal audit framework comprising three complementary defenses:

## Key Contributions

| ID | Contribution | Method |
|:---|:---|:---|
| **C1** | Drift Detection | Information-theoretic divergence monitoring across evaluation streams |
| **C2** | Sensitivity Filtering | E-value × Robustness Value dual-threshold filter for fragile outcomes |
| **C3** | Lyapunov Damping | Observable energy proxy with EMA-smoothed adaptive step-size control |

## Abstract

We address the problem of maintaining stability in self-improving AI agents that iteratively update their strategies based on evaluation feedback. We formalize three failure modes—evaluation drift, fragile outcome acceptance, and unbounded parameter updates—and propose a lightweight audit layer that wraps any base estimator. The framework combines information-theoretic drift detection (C1), sensitivity-aware effect filtering using E-values and robustness values (C2), and Lyapunov-bounded adaptive damping with an observable energy proxy (C3). Experiments on synthetic environments demonstrate that C1 improves within-horizon detection reliability, C2 substantially reduces fragile acceptance rates, and C3 achieves the lowest violation frequency with strong proxy–state alignment.

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib

# Run E1: Drift Detection (40 seeds, ~60s)
python experiments/e1_drift_detection.py

# Run E2: Sensitivity Filtering (40 seeds, ~30s)
python experiments/e2_sensitivity_filter.py

# Run E3a: Lyapunov Stability (20 seeds × 4 step sizes, ~45s)
python experiments/e3a_stationary.py

# Run E3b: Heavy-Tail Stress Test (40 seeds, ~90s)
python experiments/e3b_heavy_tail.py
```

## Repository Structure

```
WhyLab/
├── paper/                      # LaTeX source + compiled PDF
│   ├── main.tex
│   ├── main.pdf
│   ├── references.bib
│   └── neurips_2025.sty
├── experiments/                # Experiment scripts
│   ├── e1_drift_detection.py   # E1: Drift detection (C1)
│   ├── e1_censoring.py         # E1: Censoring analysis
│   ├── e1_figures.py           # E1: KM curve generation
│   ├── e2_sensitivity_filter.py # E2: Sensitivity filter (C2)
│   ├── e2_figures.py           # E2: Pareto frontier
│   ├── e3a_stationary.py       # E3a: Stationary stability (C3)
│   ├── e3a_figures.py          # E3a: Proxy trajectory plots
│   ├── e3b_heavy_tail.py       # E3b: Heavy-tail stress test
│   ├── config.yaml             # Shared hyperparameters
│   ├── figures/                # Generated figures (PDF + PNG)
│   └── results/                # Raw experiment outputs (CSV)
└── README.md
```

## Reproducing Results

| Script | Output | Paper Reference |
|:---|:---|:---|
| `e1_drift_detection.py` | `results/e1_metrics.csv` | Table 1 (E1 detection rates) |
| `e1_figures.py` | `figures/e1_km.pdf` | Figure 1 (KM curves) |
| `e2_sensitivity_filter.py` | `results/e2_metrics.csv` | Table 2 (E2 filtering) |
| `e2_figures.py` | `figures/e2_filtering.pdf` | Figure 2 (Pareto frontier) |
| `e3a_stationary.py` | `results/e3a_stationary_metrics.csv` | Table 3 (E3a stability) |
| `e3b_heavy_tail.py` | `results/e3b_full_metrics.csv` | Table A1 (E3b stress test) |

All experiments use fixed random seeds for reproducibility. Results were generated on Python 3.11 with NumPy 1.26 and SciPy 1.12.

## License

MIT License — see [LICENSE](LICENSE) for details.
