# -*- coding: utf-8 -*-
"""E10: Simple Baselines Comparison.

Addresses reviewer concern: "lack of comparison against simple alternatives."
Compares WhyLab audit components against 5 simple baselines on the E6
non-stationary environment (d=10, T=600, drift at 200/400, h_rate=0.3,
lr=0.5 aggressive).

Baselines:
  B1  Rollback-on-regression   — revert to previous theta if reward drops
  B2  Best-of-3                — generate 3 candidate updates, pick highest reward
  B3  Reward-variance rejection — reject updates when reward variance > threshold
  B4  Gradient clipping + momentum — clip gradient norm, apply momentum
  B5  Cosine step-size decay   — cosine annealing learning rate schedule

WhyLab conditions:
  No audit, C2 only, C3 only, Full (C1+C2+C3)

Metrics: final_energy, oscillation_index, regression_count, regret
"""
import numpy as np
import json
import os
import time
from dataclasses import dataclass
from typing import Tuple, List

# ── Import E6 components ──────────────────────────────────────
from e6_nonstationary_agent import (
    E6Config, NonStationaryEnv,
    C1DriftDetector, C2SensitivityFilter, C3LyapunovDamper,
)


# ── Experiment config ─────────────────────────────────────────

EXPERIMENT_CFG = dict(
    d=10,
    T=600,
    drift_points=(200, 400),
    h_rate=0.3,
    lr_base=0.5,        # aggressive
    noise_std=1.0,
    drift_magnitude=3.0,
)
N_SEEDS = 20


# ── Metric helpers ────────────────────────────────────────────

def compute_metrics(rewards: List[float], energies: List[float],
                    theta: np.ndarray, env: NonStationaryEnv,
                    T: int) -> dict:
    """Compute the four target metrics."""
    target_final = env.get_target(T - 1)
    final_energy = float(0.5 * np.sum((theta - target_final) ** 2))

    # Oscillation index: fraction of sign-changes in 10-step reward trend
    oscillations = 0
    prev_improving = None
    for i in range(len(rewards)):
        if i > 10:
            improving = rewards[i] > np.mean(rewards[i - 10:i])
        else:
            improving = True
        if prev_improving is not None and improving != prev_improving:
            oscillations += 1
        prev_improving = improving
    oscillation_index = float(oscillations / T)

    # Regression count: number of steps where energy increased
    regression_count = 0
    for i in range(1, len(energies)):
        if energies[i] > energies[i - 1] + 1e-8:
            regression_count += 1

    # Regret: cumulative negative reward (higher = worse)
    regret = float(sum(-r for r in rewards))

    return dict(
        final_energy=round(final_energy, 4),
        oscillation_index=round(oscillation_index, 4),
        regression_count=int(regression_count),
        regret=round(regret, 2),
    )


# ── WhyLab runners (ablation variants) ───────────────────────

def run_whylab(cfg: E6Config, ablation: str) -> dict:
    """Run E6 agent with WhyLab audit ablation."""
    rng = np.random.default_rng(cfg.seed)
    env = NonStationaryEnv(cfg, rng)

    theta = rng.standard_normal(cfg.d) * 0.5

    use_c1 = "C1" in ablation or ablation == "full"
    use_c2 = "C2" in ablation or ablation == "full"
    use_c3 = "C3" in ablation or ablation == "full"

    c1 = C1DriftDetector(cfg.c1_window, cfg.c1_threshold) if use_c1 else None
    c2 = C2SensitivityFilter(cfg.c2_threshold, cfg.c2_rv_threshold) if use_c2 else None
    c3 = C3LyapunovDamper(cfg.beta_ema, cfg.floor, cfg.ceiling) if use_c3 else None

    rewards, energies = [], []

    for t in range(cfg.T):
        target = env.get_target(t)
        energies.append(float(0.5 * np.sum((theta - target) ** 2)))

        reward, grad = env.observe(theta, t)
        rewards.append(reward)

        drift_alert = False
        if c1 is not None:
            drift_alert = c1.update(reward)

        if c2 is not None:
            theta_cand = theta + cfg.lr_base * grad
            reward_after, _ = env.observe(theta_cand, t)
            if not c2.should_accept(reward, reward_after, np.linalg.norm(grad)):
                continue  # reject update

        if c3 is not None:
            zeta = c3.compute_zeta(grad, drift_alert)
        else:
            zeta = cfg.lr_base

        effective_lr = zeta if use_c3 else cfg.lr_base
        theta = theta + effective_lr * grad

    return compute_metrics(rewards, energies, theta, env, cfg.T)


# ── Baseline 1: Rollback on regression ───────────────────────

def run_b1_rollback(cfg: E6Config) -> dict:
    """If reward drops compared to previous step, revert to previous theta."""
    rng = np.random.default_rng(cfg.seed)
    env = NonStationaryEnv(cfg, rng)

    theta = rng.standard_normal(cfg.d) * 0.5
    rewards, energies = [], []
    prev_reward = None

    for t in range(cfg.T):
        target = env.get_target(t)
        energies.append(float(0.5 * np.sum((theta - target) ** 2)))

        reward, grad = env.observe(theta, t)
        rewards.append(reward)

        theta_new = theta + cfg.lr_base * grad

        if prev_reward is not None and reward < prev_reward:
            # Rollback: keep current theta, don't update
            pass
        else:
            theta = theta_new

        prev_reward = reward

    return compute_metrics(rewards, energies, theta, env, cfg.T)


# ── Baseline 2: Best-of-3 ────────────────────────────────────

def run_b2_best_of_3(cfg: E6Config) -> dict:
    """Generate 3 independent updates, pick the one with highest reward."""
    rng = np.random.default_rng(cfg.seed)
    env = NonStationaryEnv(cfg, rng)

    theta = rng.standard_normal(cfg.d) * 0.5
    rewards, energies = [], []

    for t in range(cfg.T):
        target = env.get_target(t)
        energies.append(float(0.5 * np.sum((theta - target) ** 2)))

        reward, grad = env.observe(theta, t)
        rewards.append(reward)

        # Generate 3 candidate updates (each with a fresh observation)
        candidates = []
        for _ in range(3):
            _, g = env.observe(theta, t)
            theta_cand = theta + cfg.lr_base * g
            r_cand, _ = env.observe(theta_cand, t)
            candidates.append((r_cand, theta_cand))

        # Pick best
        best = max(candidates, key=lambda x: x[0])
        theta = best[1]

    return compute_metrics(rewards, energies, theta, env, cfg.T)


# ── Baseline 3: Reward-variance rejection ────────────────────

def run_b3_variance_rejection(cfg: E6Config) -> dict:
    """Reject updates when recent reward variance exceeds a threshold."""
    rng = np.random.default_rng(cfg.seed)
    env = NonStationaryEnv(cfg, rng)

    theta = rng.standard_normal(cfg.d) * 0.5
    rewards, energies = [], []
    variance_threshold = 5.0  # tuned for this environment

    for t in range(cfg.T):
        target = env.get_target(t)
        energies.append(float(0.5 * np.sum((theta - target) ** 2)))

        reward, grad = env.observe(theta, t)
        rewards.append(reward)

        # Check recent reward variance
        if len(rewards) >= 10:
            recent_var = np.var(rewards[-10:])
            if recent_var > variance_threshold:
                continue  # skip update — too volatile

        theta = theta + cfg.lr_base * grad

    return compute_metrics(rewards, energies, theta, env, cfg.T)


# ── Baseline 4: Gradient clipping + momentum ─────────────────

def run_b4_clip_momentum(cfg: E6Config) -> dict:
    """Clip gradient norm to max value, use momentum."""
    rng = np.random.default_rng(cfg.seed)
    env = NonStationaryEnv(cfg, rng)

    theta = rng.standard_normal(cfg.d) * 0.5
    rewards, energies = [], []
    momentum = np.zeros(cfg.d)
    beta_m = 0.9
    max_grad_norm = 3.0

    for t in range(cfg.T):
        target = env.get_target(t)
        energies.append(float(0.5 * np.sum((theta - target) ** 2)))

        reward, grad = env.observe(theta, t)
        rewards.append(reward)

        # Clip gradient norm
        g_norm = np.linalg.norm(grad)
        if g_norm > max_grad_norm:
            grad = grad * (max_grad_norm / g_norm)

        # Momentum
        momentum = beta_m * momentum + (1 - beta_m) * grad
        theta = theta + cfg.lr_base * momentum

    return compute_metrics(rewards, energies, theta, env, cfg.T)


# ── Baseline 5: Cosine step-size decay ───────────────────────

def run_b5_cosine_decay(cfg: E6Config) -> dict:
    """Cosine annealing learning rate schedule."""
    rng = np.random.default_rng(cfg.seed)
    env = NonStationaryEnv(cfg, rng)

    theta = rng.standard_normal(cfg.d) * 0.5
    rewards, energies = [], []
    lr_max = cfg.lr_base
    lr_min = 0.01

    for t in range(cfg.T):
        target = env.get_target(t)
        energies.append(float(0.5 * np.sum((theta - target) ** 2)))

        reward, grad = env.observe(theta, t)
        rewards.append(reward)

        # Cosine annealing
        lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / cfg.T))
        theta = theta + lr_t * grad

    return compute_metrics(rewards, energies, theta, env, cfg.T)


# ── Registry ──────────────────────────────────────────────────

METHODS = {
    # WhyLab ablations
    "No audit":       lambda cfg: run_whylab(cfg, "none"),
    "C2 only":        lambda cfg: run_whylab(cfg, "C2_only"),
    "C3 only":        lambda cfg: run_whylab(cfg, "C3_only"),
    "Full WhyLab":    lambda cfg: run_whylab(cfg, "full"),
    # Simple baselines
    "B1: Rollback":   run_b1_rollback,
    "B2: Best-of-3":  run_b2_best_of_3,
    "B3: Var-reject": run_b3_variance_rejection,
    "B4: Clip+Mom":   run_b4_clip_momentum,
    "B5: Cos-decay":  run_b5_cosine_decay,
}


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("E10: Simple Baselines Comparison")
    print(f"  d={EXPERIMENT_CFG['d']}, T={EXPERIMENT_CFG['T']}, "
          f"h_rate={EXPERIMENT_CFG['h_rate']}, lr={EXPERIMENT_CFG['lr_base']}")
    print(f"  Seeds: {N_SEEDS}  |  Methods: {len(METHODS)}")
    print("=" * 65)

    all_results = {}
    t0 = time.time()

    for method_name, runner in METHODS.items():
        seed_results = []
        for seed in range(N_SEEDS):
            cfg = E6Config(
                seed=seed,
                d=EXPERIMENT_CFG["d"],
                T=EXPERIMENT_CFG["T"],
                drift_points=EXPERIMENT_CFG["drift_points"],
                h_rate=EXPERIMENT_CFG["h_rate"],
                lr_base=EXPERIMENT_CFG["lr_base"],
                noise_std=EXPERIMENT_CFG["noise_std"],
                drift_magnitude=EXPERIMENT_CFG["drift_magnitude"],
            )
            metrics = runner(cfg)
            metrics["seed"] = seed
            seed_results.append(metrics)

        # Aggregate across seeds
        keys = ["final_energy", "oscillation_index", "regression_count", "regret"]
        agg = {}
        for k in keys:
            vals = [r[k] for r in seed_results]
            agg[k + "_mean"] = round(float(np.mean(vals)), 4)
            agg[k + "_std"] = round(float(np.std(vals)), 4)
            agg[k + "_median"] = round(float(np.median(vals)), 4)
        agg["per_seed"] = seed_results

        all_results[method_name] = agg
        elapsed = time.time() - t0
        print(f"  {method_name:22s} | energy={agg['final_energy_mean']:8.2f} "
              f"osc={agg['oscillation_index_mean']:.3f}  "
              f"reg={agg['regression_count_mean']:6.1f}  "
              f"regret={agg['regret_mean']:9.1f}  ({elapsed:.1f}s)")

    # ── Save results ──────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e10_simple_baselines.json")

    output = {
        "experiment": "E10: Simple Baselines Comparison",
        "config": EXPERIMENT_CFG,
        "n_seeds": N_SEEDS,
        "methods": list(METHODS.keys()),
        "results": all_results,
    }
    # Strip per-seed data for cleaner top-level JSON, keep it nested
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")

    # ── Summary table ─────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Method':22s} | {'Energy':>10s} | {'Osc-Idx':>9s} | {'Regressions':>12s} | {'Regret':>12s}")
    print("-" * 90)
    for method_name in METHODS:
        r = all_results[method_name]
        print(f"{method_name:22s} | "
              f"{r['final_energy_mean']:7.2f}+/-{r['final_energy_std']:<5.2f} | "
              f"{r['oscillation_index_mean']:.3f}+/-{r['oscillation_index_std']:.3f} | "
              f"{r['regression_count_mean']:5.1f}+/-{r['regression_count_std']:<5.1f} | "
              f"{r['regret_mean']:8.1f}+/-{r['regret_std']:<6.1f}")
    print("=" * 90)

    total = time.time() - t0
    print(f"\nTotal time: {total:.1f}s")


if __name__ == "__main__":
    main()
