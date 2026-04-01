# -*- coding: utf-8 -*-
"""
E7v2: Adversarial Fact-Tracking Benchmark
==========================================
Addresses E7's null-result by creating genuine oscillation pressure:
  1. 20 facts (hierarchical, with derived relationships)
  2. 5 drift events (including reversals and cascading changes)
  3. Multi-hop questions that require inference
  4. Stale context injection (past facts mixed with current)
  5. Uses paper's AgentAuditLayer (C1/C2/C3), not SimpleAuditLayer

Usage:
    python -m experiments.e7v2_adversarial_benchmark            # full
    python -m experiments.e7v2_adversarial_benchmark --pilot     # quick test
"""
import os
import json
import re
import time
import csv
import copy
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Fuzzy Evaluation ────────────────────────────────────────────
def normalize(s) -> str:
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9% ]', '', s)
    return s.strip()


def fuzzy_match(expected: str, actual: str) -> bool:
    ne, na = normalize(expected), normalize(actual)
    if not ne:
        return False
    if ne in na:
        return True
    if '%' in expected:
        num = expected.replace('%', '').strip()
        if num in na:
            return True
    return False


# ── Adversarial Dynamic Environment ────────────────────────────
class AdversarialQAEnv:
    """20 facts, 5 drifts (including reversals), multi-hop questions."""

    INITIAL_FACTS = {
        # Leadership
        "CEO": "Alice Johnson",
        "CTO": "David Chen",
        "CFO": "Bob Smith",
        "VP_Engineering": "Emily Davis",
        "VP_Sales": "Frank Miller",
        # Reporting
        "CTO_reports_to": "CEO",
        "CFO_reports_to": "CEO",
        "VP_Eng_reports_to": "CTO",
        "VP_Sales_reports_to": "CEO",
        # Company info
        "Headquarters": "New York",
        "Branch_Office": "San Francisco",
        "Main_Product": "Cloud Services",
        "Revenue_Model": "Subscription",
        "Tax_Rate": "15%",
        "Employee_Count": "500",
        # Financial
        "Last_Quarter_Revenue": "$50M",
        "Operating_Margin": "20%",
        "Currency": "USD",
        "Fiscal_Year_End": "December",
        "Founded": "2015",
    }

    # Drift schedule: epoch -> changes
    DRIFT_SCHEDULE = {
        4: {
            "CEO": "Charlie Park",  # Leadership change
            "Tax_Rate": "20%",
        },
        7: {
            "CEO": "Alice Johnson",       # REVERT CEO (contradiction!)
            "Tax_Rate": "22%",            # Different from both original and drift-1
            "Branch_Office": "Austin",    # Relocation
        },
        10: {
            "CTO": "Grace Lee",           # Cascading: new CTO
            "VP_Eng_reports_to": "CTO",   # Still reports to CTO (implicit)
            "Employee_Count": "750",
        },
        14: {
            "Headquarters": "London",
            "Currency": "GBP",            # Correlated multi-fact change
            "Main_Product": "AI Infrastructure",
        },
        17: {
            "CEO": "Henry Wilson",        # 3rd CEO change
            "CFO": "Irene Zhang",
            "Operating_Margin": "18%",
        },
    }

    def __init__(self, stale_noise_ratio: float = 0.2, seed: int = 42):
        self.facts = copy.deepcopy(self.INITIAL_FACTS)
        self.epoch = 0
        self.drift_events = []
        self.stale_facts: list[tuple[str, str]] = []  # (key, old_value)
        self.stale_noise_ratio = stale_noise_ratio
        self.rng = random.Random(seed)

    def evolve(self):
        self.epoch += 1
        changes = self.DRIFT_SCHEDULE.get(self.epoch)
        if changes:
            for k, v in changes.items():
                old = self.facts.get(k)
                if old and old != v:
                    self.stale_facts.append((k, old))
                self.facts[k] = v
            self.drift_events.append(self.epoch)

    def get_knowledge_context(self) -> str:
        """Current facts + stale noise (shuffled)."""
        lines = [f"- {k}: {v}" for k, v in self.facts.items()]

        # Inject stale facts as noise
        n_stale = max(1, int(len(self.stale_facts) * self.stale_noise_ratio))
        if self.stale_facts:
            sampled = self.rng.sample(
                self.stale_facts, min(n_stale, len(self.stale_facts))
            )
            for k, v in sampled:
                lines.append(f"- [Previous Record] {k}: {v}")

        self.rng.shuffle(lines)
        return "Company Knowledge Base (may contain historical records):\n" + "\n".join(lines)

    def get_questions(self) -> list[dict]:
        """Direct + multi-hop questions."""
        qs = []
        # Direct questions
        for key in ["CEO", "CTO", "CFO", "Headquarters", "Tax_Rate",
                     "Main_Product", "Currency", "Employee_Count",
                     "Operating_Margin", "Revenue_Model"]:
            qs.append({
                "key": key,
                "question": f"What is the current {key.replace('_', ' ')}?",
                "answer": self.facts[key],
                "type": "direct",
            })

        # Multi-hop questions (derived from relationships)
        cto_name = self.facts["CTO"]
        ceo_name = self.facts["CEO"]
        qs.append({
            "key": "CTO_boss_name",
            "question": "Who does the CTO report to? Give the person's name.",
            "answer": ceo_name,  # CTO reports to CEO -> CEO's name
            "type": "multi_hop",
        })
        qs.append({
            "key": "VP_Eng_boss_boss",
            "question": "Who is the VP of Engineering's boss's boss?",
            "answer": ceo_name,  # VP_Eng -> CTO -> CEO
            "type": "multi_hop",
        })
        qs.append({
            "key": "HQ_currency",
            "question": "What currency is used at the headquarters?",
            "answer": self.facts["Currency"],
            "type": "multi_hop",
        })

        return qs

    def evaluate(self, answers: dict) -> tuple[float, dict]:
        """Evaluate answers against current facts."""
        questions = self.get_questions()
        correct = 0
        detail = {}
        for q in questions:
            actual = answers.get(q["key"], "")
            match = fuzzy_match(q["answer"], actual)
            if match:
                correct += 1
            detail[q["key"]] = {
                "expected": q["answer"], "actual": actual,
                "correct": match, "type": q["type"],
            }
        score = correct / len(questions) if questions else 0.0
        return score, detail


# ── LLM Agent (supports Gemini + OpenAI) ───────────────────────
class LLMAgent:
    """Agent that supports both Gemini and OpenAI APIs."""

    def __init__(self, provider: str = "gemini", model: str = "gemini-2.0-flash"):
        self.provider = provider
        self.model_name = model
        self.system_rules = (
            "You are an executive assistant AI. Given a company knowledge base, "
            "answer questions about the company. CRITICAL: Use ONLY the current "
            "values from the database. Ignore any entries marked as [Previous Record]. "
            "Output answers as JSON: {\"key\": \"value\", ...}. No extra text."
        )
        self.prompt_history = [self.system_rules]
        self._init_client()

    def _init_client(self):
        if self.provider == "gemini":
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model_name)
        elif self.provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def answer_questions(self, questions: list[dict], context: str) -> dict:
        keys = [q["key"] for q in questions]
        q_texts = [f'"{q["question"]}"' for q in questions]
        prompt = (
            f"{self.system_rules}\n\n"
            f"{context}\n\n"
            f"Answer these questions (use these exact keys: {keys}):\n"
            + "\n".join(f"- {q['key']}: {q['question']}" for q in questions)
            + f"\n\nOutput ONLY valid JSON with keys: {keys}"
        )
        return self._call_llm(prompt, temperature=0.0, max_tokens=400)

    def propose_improvement(self, failed_items: dict) -> str:
        if not failed_items:
            return self.system_rules
        fail_str = "\n".join(
            f"  {k}: expected '{v['expected']}', got '{v['actual']}' (type={v.get('type','direct')})"
            for k, v in failed_items.items()
        )
        meta = (
            f"Your agent failed on:\n{fail_str}\n\n"
            f"Current rules:\n'{self.system_rules}'\n\n"
            f"Rewrite the rules to fix these errors. Key issues:\n"
            f"1. Ignore [Previous Record] entries\n"
            f"2. For multi-hop questions, follow reporting chain\n"
            f"3. Use ONLY current database values\n"
            f"Max 100 words. Output ONLY the new rules text."
        )
        result = self._call_llm_text(meta, temperature=0.7, max_tokens=200)
        return result or self.system_rules

    def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> dict:
        text = self._call_llm_text(prompt, temperature, max_tokens)
        try:
            match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return {}

    def _call_llm_text(self, prompt: str, temperature: float, max_tokens: int) -> str:
        max_retries = 6
        for attempt in range(max_retries):
            try:
                if self.provider == "gemini":
                    resp = self.client.generate_content(
                        prompt,
                        generation_config={
                            "temperature": temperature,
                            "max_output_tokens": max_tokens,
                        }
                    )
                    return resp.text.strip()
                elif self.provider == "openai":
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return resp.choices[0].message.content.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "Resource exhausted" in err_str:
                    wait = min(10 * (2 ** attempt), 60)  # 10, 20, 40, 60, 60, 60
                    print(f"    [Rate limit] Retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                print(f"    [LLM Error] {e}")
                return ""
        print(f"    [LLM Error] Max retries exceeded")
        return ""


# ── Benchmark Runner ────────────────────────────────────────────
@dataclass
class E7v2Result:
    seed: int
    mode: str
    provider: str
    model: str
    mean_accuracy: float
    final_accuracy: float
    oscillations: int
    regressions: int
    rejections: int
    recoveries: int = 0  # fail->pass transitions


def run_e7v2(
    n_epochs: int = 20,
    n_seeds: int = 10,
    use_audit: bool = True,
    audit_mode: str = "fixed",
    provider: str = "gemini",
    model: str = "gemini-2.0-flash",
    stale_noise: float = 0.2,
    warmup_epochs: int = 3,
    sensitivity_scaling: float = 10.0,
) -> list[E7v2Result]:
    """Run E7v2 adversarial benchmark.

    Args:
        audit_mode: 'fixed' (original), 'adaptive' (auto-calibrated), or ignored if use_audit=False
        warmup_epochs: Number of warmup epochs for adaptive mode
        sensitivity_scaling: How strongly variance affects threshold (adaptive mode)
    """
    # Lazy imports
    from experiments.audit_layer import AgentAuditLayer
    from experiments.adaptive_audit_layer import AdaptiveAuditLayer

    if not use_audit:
        label = "baseline"
    elif audit_mode == "adaptive":
        label = "audit_adaptive"
    else:
        label = "audit"
    results = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        env = AdversarialQAEnv(stale_noise_ratio=stale_noise, seed=seed)
        agent = LLMAgent(provider=provider, model=model)

        # Audit config matching paper's E7 settings
        audit_cfg = {
            "c1": True,
            "c2": True,
            "c3": True,
            "c1_window": 5,
            "c1_agreement_threshold": 0.4,
            "c2_e_thresh": 1.5,
            "c2_rv_thresh": 0.05,
            "c3_epsilon_floor": 0.01,
            "c3_ceiling": 0.8,
        }

        audit = None
        if use_audit:
            if audit_mode == "adaptive":
                audit = AdaptiveAuditLayer(
                    audit_cfg,
                    warmup_epochs=warmup_epochs,
                    sensitivity_scaling=sensitivity_scaling,
                )
            else:
                audit = AgentAuditLayer(audit_cfg)

        accs = []
        score_history = []  # for audit layer's before/after windows
        rejections = 0
        oscillations = 0
        regressions = 0
        recoveries = 0
        prev_acc = None

        for epoch in range(n_epochs):
            env.evolve()
            questions = env.get_questions()
            context = env.get_knowledge_context()
            answers = agent.answer_questions(questions, context)
            acc, detail = env.evaluate(answers)
            accs.append(acc)
            score_history.append(acc)

            # Track oscillation, regression, recovery
            if prev_acc is not None:
                if acc < prev_acc - 0.1:
                    regressions += 1
                if prev_acc < 0.5 and acc >= 0.7:
                    recoveries += 1
                if len(accs) >= 3:
                    a, b, c = accs[-3], accs[-2], accs[-1]
                    if (a > b + 0.05 and c > b + 0.05) or \
                       (a < b - 0.05 and c < b - 0.05):
                        oscillations += 1

            # Reflexion: propose improvement on failures
            failed = {k: v for k, v in detail.items() if not v["correct"]}
            if failed:
                proposed = agent.propose_improvement(failed)
                if audit and proposed != agent.system_rules:
                    # Compute magnitude
                    old_len = len(agent.system_rules)
                    new_len = len(proposed)
                    magnitude = abs(new_len - old_len) / max(old_len, 1)

                    # Build before/after score windows for audit layer
                    window = 5
                    scores_before = score_history[max(0, len(score_history)-window-1):-1] \
                        if len(score_history) > 1 else [acc]
                    scores_after = score_history[-window:]

                    decision = audit.evaluate_update(
                        cheap_score=acc,
                        full_pass=(acc >= 0.7),
                        scores_before=scores_before,
                        scores_after=scores_after,
                        update_magnitude=magnitude,
                    )
                    if decision.accept:
                        agent.system_rules = proposed
                    else:
                        rejections += 1
                else:
                    agent.system_rules = proposed

            prev_acc = acc
            time.sleep(4)  # Rate limiting (4s for Gemini free tier: 15 RPM)

        result = E7v2Result(
            seed=seed, mode=label, provider=provider, model=model,
            mean_accuracy=round(float(np.mean(accs)), 4),
            final_accuracy=round(float(accs[-1]), 4),
            oscillations=oscillations,
            regressions=regressions,
            rejections=rejections,
            recoveries=recoveries,
        )
        results.append(result)
        print(f"  [seed={seed}] {label}: acc={result.mean_accuracy:.3f}, "
              f"osc={oscillations}, reg={regressions}, rej={rejections}", flush=True)

    return results


def save_results(results: list[E7v2Result], filename: str):
    out = RESULTS_DIR / filename
    with open(out, "w", newline="", encoding="utf-8") as f:
        fields = ["seed", "mode", "provider", "model", "mean_accuracy",
                  "final_accuracy", "oscillations", "regressions",
                  "rejections", "recoveries"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "seed": r.seed, "mode": r.mode,
                "provider": r.provider, "model": r.model,
                "mean_accuracy": r.mean_accuracy,
                "final_accuracy": r.final_accuracy,
                "oscillations": r.oscillations,
                "regressions": r.regressions,
                "rejections": r.rejections,
                "recoveries": r.recoveries,
            })
    print(f"Saved: {out}")


def print_summary(baseline: list[E7v2Result], audited: list[E7v2Result]):
    bl_acc = np.mean([r.mean_accuracy for r in baseline])
    au_acc = np.mean([r.mean_accuracy for r in audited])
    bl_osc = np.mean([r.oscillations for r in baseline])
    au_osc = np.mean([r.oscillations for r in audited])
    bl_reg = np.mean([r.regressions for r in baseline])
    au_reg = np.mean([r.regressions for r in audited])
    bl_rej = np.mean([r.rejections for r in baseline])
    au_rej = np.mean([r.rejections for r in audited])

    print(f"\n{'Metric':<20} {'Baseline':<12} {'WhyLab':<12} {'Delta':<12}")
    print("-" * 56)
    print(f"{'Mean Accuracy':<20} {bl_acc:<12.3f} {au_acc:<12.3f} {au_acc-bl_acc:<+12.3f}")
    print(f"{'Oscillations':<20} {bl_osc:<12.1f} {au_osc:<12.1f} {au_osc-bl_osc:<+12.1f}")
    print(f"{'Regressions':<20} {bl_reg:<12.1f} {au_reg:<12.1f} {au_reg-bl_reg:<+12.1f}")
    print(f"{'Rejections':<20} {bl_rej:<12.1f} {au_rej:<12.1f} {au_rej-bl_rej:<+12.1f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="E7v2: Adversarial Fact-Tracking")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openai"])
    parser.add_argument("--model", default=None,
                        help="Model name (default: auto by provider)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--stale-noise", type=float, default=0.2)
    parser.add_argument("--pilot", action="store_true",
                        help="Quick test: 3 seeds, 5 epochs")
    parser.add_argument("--adaptive", action="store_true",
                        help="Also run adaptive threshold mode")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup epochs for adaptive mode")
    parser.add_argument("--sensitivity-scaling", type=float, default=10.0,
                        help="Sensitivity scaling factor for adaptive thresholds")
    args = parser.parse_args()

    if args.pilot:
        args.seeds = 3
        args.epochs = 5

    if args.model is None:
        args.model = {
            "gemini": "gemini-2.0-flash",
            "openai": "gpt-4o-mini",
        }.get(args.provider, args.provider)

    provider_tag = args.provider
    model_tag = args.model.replace("/", "_")

    print("=" * 60, flush=True)
    print(f"E7v2: Adversarial Fact-Tracking ({args.provider} / {args.model})", flush=True)
    print(f"  Epochs={args.epochs}, Seeds={args.seeds}, Noise={args.stale_noise}", flush=True)
    print("=" * 60, flush=True)

    print("\n--- Baseline (No Audit) ---")
    baseline = run_e7v2(
        n_epochs=args.epochs, n_seeds=args.seeds, use_audit=False,
        provider=args.provider, model=args.model,
        stale_noise=args.stale_noise,
    )

    print("\n--- WhyLab Fixed (With Audit) ---")
    audited = run_e7v2(
        n_epochs=args.epochs, n_seeds=args.seeds, use_audit=True,
        audit_mode="fixed",
        provider=args.provider, model=args.model,
        stale_noise=args.stale_noise,
    )

    all_results = baseline + audited

    # Adaptive mode (optional)
    if args.adaptive:
        print("\n--- WhyLab Adaptive (Auto-calibrated) ---")
        adaptive = run_e7v2(
            n_epochs=args.epochs, n_seeds=args.seeds, use_audit=True,
            audit_mode="adaptive",
            provider=args.provider, model=args.model,
            stale_noise=args.stale_noise,
            warmup_epochs=args.warmup,
            sensitivity_scaling=args.sensitivity_scaling,
        )
        all_results += adaptive
        print("\n=== Fixed Audit Summary ===")

    save_results(
        all_results,
        f"e7v2_{model_tag}_results.csv"
    )
    print_summary(baseline, audited)

    if args.adaptive:
        print("\n=== Adaptive Audit Summary ===")
        print_summary(baseline, adaptive)


if __name__ == "__main__":
    main()

