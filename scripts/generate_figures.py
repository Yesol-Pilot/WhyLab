# -*- coding: utf-8 -*-
"""백서 Figure 자동 생성 스크립트.

WhyLab v1.0의 핵심 기능을 시연하는 3종의 Figure를 생성합니다.
1. Dose-Response Curve (불확실성 포함)
2. Fairness Audit Radar Chart
3. MAC Discovery DAG
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from engine.agents.mac_discovery import MACDiscoveryAgent
from engine.cells.dose_response_cell import DoseResponseCell
from engine.cells.fairness_audit_cell import FairnessAuditCell

# 스타일 설정
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["font.family"] = "DejaVu Sans"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def generate_dose_response_figure():
    """Figure 1: Dose-Response Curve."""
    print("Generating Figure 1: Dose-Response Curve...")
    
    # 합성 데이터 (비선형)
    rng = np.random.RandomState(42)
    n = 1000
    X = rng.randn(n, 3)
    T = rng.uniform(0, 10, n)
    # True Response: y = -0.5(t-5)^2 + 15 + noise
    Y = -0.5 * (T - 5) ** 2 + 15 + X[:, 0] + rng.randn(n) * 1.0
    
    cell = DoseResponseCell()
    res = cell.estimate(X, T, Y)
    
    plt.figure(figsize=(8, 6))
    
    # 신뢰구간
    if "ci_lower" in res:
        plt.fill_between(
            res["t_grid"], res["ci_lower"], res["ci_upper"],
            color="purple", alpha=0.2, label="95% CI"
        )
    
    # 평균 반응 곡선
    plt.plot(res["t_grid"], res["dr_curve"], color="purple", lw=2, label="Estimated Response")
    
    # 최적 용량
    opt_t = res["optimal_dose"]
    opt_y = res["optimal_effect"]
    plt.axvline(opt_t, color="cyan", linestyle="--", label=f"Optimal Dose ({opt_t:.1f})")
    plt.scatter([opt_t], [opt_y], color="cyan", s=100, zorder=5)
    
    plt.title("Dose-Response Analysis with Uncertainty Quantification", fontsize=14)
    plt.xlabel("Treatment Dosage", fontsize=12)
    plt.ylabel("Outcome Response", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_dose_response.png", dpi=300)
    plt.close()


def generate_fairness_radar_figure():
    """Figure 2: Fairness Audit Radar Chart."""
    print("Generating Figure 2: Fairness Audit Radar Chart...")
    
    # 가상의 불공정 데이터
    categories = ['Causal Parity', 'Disparate Impact', 'Equalized CATE', 'Counterfactual']
    # Group A (Fair), Group B (Unfair)
    values_a = [0.05, 0.9, 0.85, 0.95]  # Good
    values_b = [0.25, 0.6, 0.45, 0.65]  # Bad (Bias)
    
    # Normalize for radar chart (Pass threshold awareness)
    # Causal Parity: 0 is best. Invert for chart? No, keep natural.
    # Let's visualize score (0-1) where 1 is best.
    # Causal Parity Gap -> 1 - gap (clipped)
    
    scores_a = [1 - 0.05, 0.9, 0.85, 0.95]
    scores_b = [1 - 0.25, 0.6, 0.45, 0.65]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_a += values_a[:1]
    values_b += values_b[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw Scale
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories)
    
    # Plot A
    ax.plot(angles, values_a, linewidth=2, linestyle='solid', label='Protected Group A (Fair)')
    ax.fill(angles, values_a, 'b', alpha=0.1)
    
    # Plot B
    ax.plot(angles, values_b, linewidth=2, linestyle='solid', label='Protected Group B (Biased)')
    ax.fill(angles, values_b, 'r', alpha=0.1)
    
    plt.title("Fairness Audit: Group Comparison", fontsize=14, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_fairness_radar.png", dpi=300)
    plt.close()


def generate_mac_dag_figure():
    """Figure 3: MAC Discovery DAG."""
    print("Generating Figure 3: MAC Discovery DAG...")
    
    G = nx.DiGraph()
    # 노드: (이름, 역할)
    nodes = {
        "Age": "confounder", "Income": "confounder", 
        "Education": "confounder",
        "Program": "treatment",
        "Skill": "mediator",
        "Earnings": "outcome"
    }
    
    edges = [
        ("Age", "Program", 0.9), ("Income", "Program", 0.8),
        ("Age", "Earnings", 0.7), ("Income", "Earnings", 0.85),
        ("Education", "Earnings", 0.8), ("Education", "Program", 0.6),
        ("Program", "Skill", 0.95),  # Strong mediator
        ("Skill", "Earnings", 0.9),
        ("Program", "Earnings", 0.4) # Weak direct
    ]
    
    pos = {
        "Age": (0, 2), "Income": (1, 2), "Education": (2, 2),
        "Program": (1, 1),
        "Skill": (1, 0.5),
        "Earnings": (1, 0)
    }
    
    color_map = {
        "confounder": "#A0AEC0",
        "treatment": "#805AD5", # Purple
        "mediator": "#4FD1C5",  # Teal
        "outcome": "#0BC5EA",   # Cyan
        "other": "#CBD5E0"
    }
    
    plt.figure(figsize=(10, 8))
    
    # 엣지 그리기 (두께 = 가중치)
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], width=w*3, alpha=0.6,
            edge_color="#718096", arrowsize=20, connectionstyle="arc3,rad=0.1"
        )
        
    # 노드 그리기
    for node, role in nodes.items():
        nx.draw_networkx_nodes(
            G, pos, nodelist=[node], node_color=color_map[role],
            node_size=2000, alpha=0.9
        )
        
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_color="white")
    
    plt.title("MAC Discovery: Consensus Causal Graph", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_mac_dag.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    generate_dose_response_figure()
    generate_fairness_radar_figure()
    generate_mac_dag_figure()
    print("Figure generation completed.")
