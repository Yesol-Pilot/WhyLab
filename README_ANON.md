# WhyLab: A Causal Audit Framework for Stable Agent Self-Improvement

> Autonomous code for NeurIPS 2026 Submission

This repository contains the official implementation of **WhyLab**, a causal audit framework designed to safeguard self-improving AI agents against evaluation drift, fragile outcomes, and unbounded parameter updates.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all experiments sequentially
python experiments/e1_drift_detection.py
python experiments/e2_sensitivity_filter.py
python experiments/e3a_stationary.py
python experiments/e3b_heavy_tail.py

# 3. Non-stationary Agent Environment (E6)
python experiments/e6_nonstationary_agent.py
```

## 📁 Repository Structure

```text
WhyLab_Anonymous/
├── README_ANON.md           # This file
├── requirements.txt         # Minimal deps list
├── experiments/             # Core scripts (E1-E3)
│   ├── config.yaml          # Hyperparameters
│   ├── e1_drift_detection.py
│   ├── e2_sensitivity_filter.py
│   ├── e3a_stationary.py
│   ├── e3b_heavy_tail.py
│   └── e6_nonstationary_agent.py
└── paper/                   # Paper LaTeX source
    ├── main.tex
    ├── references.bib
    └── main.pdf             # Compiled PDF
```
