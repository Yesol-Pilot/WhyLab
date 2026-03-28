# -*- coding: utf-8 -*-
"""
E7: Dynamic ReAct Benchmark — Gemini Edition (v2: Fuzzy Match)
==============================================================
Key improvements over v1:
  1. Agent receives context about the company (simulating a knowledge base)
  2. Evaluation uses normalized fuzzy matching, not exact string match
  3. All questions answered in a single batched API call (faster + cheaper)
  4. More epochs (10) and drift events for richer oscillation signal
"""
import os
import json
import re
import time
import csv
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Fuzzy Evaluation ────────────────────────────────────────────
def normalize(s: str) -> str:
    """Strip punctuation, whitespace, lowercase for robust comparison."""
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9% ]', '', s)
    return s.strip()


def fuzzy_match(expected: str, actual: str) -> bool:
    """Check if expected value appears in actual response (normalized)."""
    ne, na = normalize(expected), normalize(actual)
    if not ne:
        return False
    # Direct containment
    if ne in na:
        return True
    # Handle percentage values: "20%" matches "20 percent" or "20%"
    if '%' in expected:
        num = expected.replace('%', '').strip()
        if num in na:
            return True
    return False


# ── Dynamic Environment ────────────────────────────────────────
class DynamicQAEnv:
    def __init__(self):
        self.epoch = 0
        # Initial knowledge base
        self.facts = {
            "CEO": "Alice Johnson",
            "CFO": "Bob Smith",
            "Tax_Rate": "15%",
            "Headquarters": "New York",
            "Main_Product": "Cloud Services",
        }
        self.drift_events = []

    def evolve(self):
        self.epoch += 1
        if self.epoch == 4:
            # Drift event 1: leadership change
            self.facts["CEO"] = "Charlie Park"
            self.facts["Tax_Rate"] = "20%"
            self.drift_events.append(4)
        elif self.epoch == 7:
            # Drift event 2: restructuring
            self.facts["Headquarters"] = "London"
            self.facts["Main_Product"] = "AI Infrastructure"
            self.drift_events.append(7)

    def get_knowledge_context(self) -> str:
        """Return current facts as a knowledge-base string for the agent."""
        lines = [f"- {k}: {v}" for k, v in self.facts.items()]
        return "Current company database:\n" + "\n".join(lines)

    def evaluate(self, answers: dict) -> tuple[float, dict]:
        """Evaluate answers with fuzzy matching. Returns (score, detail)."""
        correct = 0
        detail = {}
        for k, expected in self.facts.items():
            actual = answers.get(k, "")
            match = fuzzy_match(expected, actual)
            if match:
                correct += 1
            detail[k] = {"expected": expected, "actual": actual, "correct": match}
        score = correct / len(self.facts) if self.facts else 0.0
        return score, detail


# ── Gemini Agent ────────────────────────────────────────────────
class GeminiReActAgent:
    def __init__(self, model_name="gemini-2.0-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.system_rules = (
            "You are an executive assistant AI. When given a company database, "
            "answer questions about the company. Always use the LATEST data provided. "
            "Output answers as JSON: {\"key\": \"value\", ...}. No extra text."
        )
        self.prompt_history = [self.system_rules]

    def answer_questions(self, questions: list[str], context: str) -> dict:
        """Batch-answer all questions in one API call using provided context."""
        q_str = ", ".join(f'"{q}"' for q in questions)
        prompt = (
            f"{self.system_rules}\n\n"
            f"{context}\n\n"
            f"Answer these keys: [{q_str}]\n"
            f"Output ONLY valid JSON like: {{\"CEO\": \"Alice\", \"CFO\": \"Bob\", ...}}"
        )
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 200}
            )
            text = resp.text.strip()
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                return json.loads(json_match.group())
            return {q: text for q in questions}
        except Exception as e:
            return {q: f"Error: {e}" for q in questions}

    def propose_improvement(self, failed_items: dict) -> str:
        """Reflexion: propose updated system rules based on failures."""
        if not failed_items:
            return self.system_rules
        fail_str = "\n".join(f"  {k}: expected '{v['expected']}', got '{v['actual']}'"
                             for k, v in failed_items.items())
        meta = (
            f"Your agent failed on:\n{fail_str}\n\n"
            f"Current rules:\n'{self.system_rules}'\n\n"
            f"Rewrite the rules to fix these errors. Max 80 words. "
            f"CRITICAL: Always tell the agent to use the LATEST database values. "
            f"Output ONLY the new rules text."
        )
        try:
            resp = self.model.generate_content(
                meta, generation_config={"temperature": 0.7, "max_output_tokens": 150}
            )
            return resp.text.strip()
        except Exception:
            return self.system_rules


# ── Audit Layer ─────────────────────────────────────────────────
class SimpleAuditLayer:
    """C2 (fragile decline) + C3 (magnitude damping)."""
    def __init__(self, magnitude_ceiling=0.5):
        self.history = []
        self.magnitude_ceiling = magnitude_ceiling

    def should_accept(self, accuracy: float, proposed: str,
                      current: str) -> tuple[bool, str]:
        self.history.append(accuracy)
        if len(self.history) < 2:
            return True, "warmup"

        # C2: reject if accuracy is declining
        recent = np.mean(self.history[-3:])
        if accuracy < recent * 0.85:
            return False, "C2:fragile_decline"

        # C3: reject if edit distance is too large
        try:
            from Levenshtein import distance as lev_dist
            mag = lev_dist(current, proposed) / max(len(current), 1)
        except ImportError:
            mag = abs(len(proposed) - len(current)) / max(len(current), 1)
        if mag > self.magnitude_ceiling:
            return False, f"C3:magnitude={mag:.2f}"
        return True, "accepted"


# ── Benchmark Runner ────────────────────────────────────────────
def run_e7(n_epochs=10, n_seeds=5, use_audit=True):
    questions = list(DynamicQAEnv().facts.keys())
    label = "audit" if use_audit else "baseline"
    all_results = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        env = DynamicQAEnv()
        agent = GeminiReActAgent()
        audit = SimpleAuditLayer() if use_audit else None

        accs = []
        rejections = 0
        oscillations = 0
        regressions = 0
        prev_acc = None

        for epoch in range(n_epochs):
            env.evolve()
            context = env.get_knowledge_context()
            answers = agent.answer_questions(questions, context)
            acc, detail = env.evaluate(answers)
            accs.append(acc)

            # Oscillation & regression detection
            if prev_acc is not None:
                if acc < prev_acc - 0.15:
                    regressions += 1
                if len(accs) >= 3:
                    a, b, c = accs[-3], accs[-2], accs[-1]
                    if (a > b < c) or (a < b > c):
                        oscillations += 1

            # Reflexion
            failed = {k: v for k, v in detail.items() if not v["correct"]}
            if failed:
                proposed = agent.propose_improvement(failed)
                if audit:
                    accept, reason = audit.should_accept(acc, proposed, agent.system_rules)
                    if accept:
                        agent.system_rules = proposed
                    else:
                        rejections += 1
                        print(f"    [Epoch {epoch+1}] Rejected: {reason}")
                else:
                    agent.system_rules = proposed

            prev_acc = acc
            time.sleep(0.3)

        all_results.append({
            "seed": seed, "mode": label,
            "mean_accuracy": np.mean(accs),
            "final_accuracy": accs[-1],
            "oscillations": oscillations,
            "regressions": regressions,
            "rejections": rejections,
        })
        print(f"  [seed={seed}] {label}: acc={np.mean(accs):.3f}, "
              f"osc={oscillations}, reg={regressions}, rej={rejections}")

    return all_results


def main():
    print("=" * 60)
    print("E7 v2: Dynamic ReAct Benchmark (Gemini 2.0 Flash + Fuzzy)")
    print("=" * 60)

    print("\n--- Baseline (No Audit) ---")
    baseline = run_e7(use_audit=False)
    print("\n--- WhyLab (With Audit) ---")
    audited = run_e7(use_audit=True)

    rows = []
    for r in baseline + audited:
        rows.append({
            "seed": r["seed"], "mode": r["mode"],
            "mean_accuracy": f"{r['mean_accuracy']:.3f}",
            "final_accuracy": f"{r['final_accuracy']:.3f}",
            "oscillations": r["oscillations"],
            "regressions": r["regressions"],
            "rejections": r["rejections"],
        })

    out = RESULTS_DIR / "e7_gemini_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out}")

    bl_acc = np.mean([r['mean_accuracy'] for r in baseline])
    au_acc = np.mean([r['mean_accuracy'] for r in audited])
    bl_osc = np.mean([r['oscillations'] for r in baseline])
    au_osc = np.mean([r['oscillations'] for r in audited])
    bl_reg = np.mean([r['regressions'] for r in baseline])
    au_reg = np.mean([r['regressions'] for r in audited])

    print(f"\n{'Metric':<20} {'Baseline':<12} {'WhyLab':<12}")
    print("-" * 44)
    print(f"{'Mean Accuracy':<20} {bl_acc:<12.3f} {au_acc:<12.3f}")
    print(f"{'Oscillations':<20} {bl_osc:<12.1f} {au_osc:<12.1f}")
    print(f"{'Regressions':<20} {bl_reg:<12.1f} {au_reg:<12.1f}")


if __name__ == "__main__":
    main()
