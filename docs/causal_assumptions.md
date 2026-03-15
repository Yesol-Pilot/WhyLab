# Causal Assumptions & Failure Modes — WhyLab Framework

> **Purpose**: NeurIPS Appendix 1-page reference for reviewers.
> Covers C2 (Sensitivity Filter), C3 (Lyapunov Stability), and Experiment assumptions.

---

## C2: Sensitivity / Causal Filter Assumptions

| ID | Assumption | Formal Statement | Violation Failure Mode | Mitigation / Refutation |
|:---|:-----------|:-----------------|:-----------------------|:------------------------|
| C2-A1 | **Ignorability (conditional)** | $(Y(1),Y(0)) \perp T \mid X$ | False positive causal attribution → unnecessary ζ suppression | E-value reports "minimum confounding strength to nullify" (VanderWeele & Ding 2017); Refutation: Random Common Cause test |
| C2-A2 | **Overlap (positivity)** | $0 < P(T=1\mid X) < 1$ for all $X$ | Extreme propensity → unstable ATE; extrapolation bias | OLS on balanced DGP (E2 design); flagged when propensity ∈ {0,1} |
| C2-A3 | **No measurement error** | Observed $X, T, Y$ are exact | Attenuated ATE → biased E-value/RV | SE-based Welch correction; acknowledged as limitation |
| C2-A4 | **No unmeasured confounding** | No $U$ s.t. $U \to T$ and $U \to Y$ | Spurious ATE accepted as causal → fragile rollout | (i) E-value ≥ threshold ensures "confounder must be ≥ E-strong", (ii) RV_q ≥ threshold ensures "partial R² of confounder must be ≥ RV", (iii) Placebo/Subset refutation tests |
| C2-A5 | **Stationarity of effect** | ATE does not change within observation window | Time-varying effect → biased estimate | Observation window padding; E1 drift detection triggers re-audit |

### Refutation Strategy (DoWhy-aligned, §4.2 Appendix)

| Refuter | Mechanism | Pass Criterion |
|:--------|:----------|:---------------|
| **Random Common Cause** | Add random covariate $Z \sim \mathcal{N}(0,1)$ to OLS | $|\Delta\text{ATE}| / |\text{ATE}_\text{orig}| < 0.15$ |
| **Placebo Treatment** | Replace $T$ with $T_\text{rand} \sim \text{Bernoulli}(0.5)$ | $|\text{ATE}_\text{placebo}| < 0.5 \times |\text{ATE}_\text{orig}|$ |
| **Data Subset** | Re-estimate on 80% random subset (×5) | $\text{CV}(\text{ATE}_\text{subset}) < 0.5$ |

---

## C3: Lyapunov Stability Assumptions

| ID | Assumption | Formal Statement | Violation Failure Mode | Mitigation |
|:---|:-----------|:-----------------|:-----------------------|:-----------|
| C3-A1 | **Quadratic energy landscape** | $V(\theta) = \frac{1}{2}\|\theta - \theta^*\|^2$ is valid proxy | Non-convex landscape → local minima; ζ_max guarantees don't hold globally | Reward-based proxy $V_t \approx \frac{1}{2}(R_{\max} - R_t)^2$; EMA smoothing |
| C3-A2 | **Bounded gradient noise** | $\mathbb{E}[\|\hat{g}_t\|^2] < \infty$ | Heavy-tail noise → ζ_max → 0 → learning deadlock | ε_floor = min_zeta (default 0.01) ensures minimum learning rate |
| C3-A3 | **θ* existence & accessibility** | Optimal policy θ* exists and is reachable via gradient descent | No stable optimum → energy never converges | Ceiling C = max_zeta (default 0.8) prevents overshooting; tracked via `is_converging()` |

### Design Trade-offs (Ablation targets, E3a)

| Component | Purpose | Removal Effect (E3a prediction) |
|:----------|:--------|:-------------------------------|
| EMA smoothing | Reduces ζ oscillation under noisy gradients | Higher variance in ζ → more violation events |
| ε_floor (0.01) | Prevents learning deadlock under heavy-tail noise | Without: learning stops entirely at high hallucination rates |
| Ceiling C (0.8) | Prevents overshooting even when signal is strong | Without: rare large ζ spikes → divergence at h_rate=0 |
| β (EMA coeff) | Controls memory/reactivity trade-off | β too high → slow adaptation; β too low → noisy ζ |

---

## Experiment Assumptions

| ID | Assumption | Impact if Violated | Status |
|:---|:-----------|:-------------------|:-------|
| EX-A1 | **Synthetic DGP is representative** | Results may not generalize to real agent pipelines | Acknowledged; E4 (HumanEval benchmark) provides partial real-world validation |
| EX-A2 | **Hallucination noise is Gaussian/mixture** | Real LLM hallucinations may be structured, not i.i.d. | Heavy-tail (E3b) tests Pareto-distributed noise |
| EX-A3 | **OLS is sufficient estimator for E2** | Complex treatment mechanisms may need DML/AIPW | Phase 2 upgrade path documented; current OLS provides interpretable baseline |
| EX-A4 | **40 seeds provide stable estimates** | May miss rare failure modes | Bootstrap CI reported; tail-risk analysis in E3b |

---

## Cross-references

- C2 assumptions connect to E2 experimental validation (§4.2 in paper)
- C3 assumptions connect to E3a/E3b experimental validation (§4.3 in paper)
- Refutation results: `experiments/results/e2_refutation.csv`
- Ablation results: `experiments/results/e3a_ablation.csv`
