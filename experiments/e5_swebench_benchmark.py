"""
SWE-bench Lite: Execution-based Benchmarking Script
===================================================
Runs WhyLab causal audit on SWE-bench Lite utilizing the official
Docker execution environment (eval_mode='docker'). Replaces the deprecated
string-match heuristics.
"""
import os
import argparse
import pandas as pd
from swebench_loader import load_swebench_lite
from llm_client import CachedLLMClient
from audit_layer import AgentAuditLayer
from swebench_reflexion import run_swe_reflexion_episode

def run_evaluation(num_problems=10, seed=42, use_docker=True):
    print("=" * 60)
    print("WhyLab SWE-bench Execution Benchmark Runner")
    print(f"Mode: {'DOCKER' if use_docker else 'LOCAL REPO'}")
    print("=" * 60)
    
    problems = load_swebench_lite(subset=num_problems)
    
    llm = CachedLLMClient(
        model="gemini-2.0-flash",
        temperature=0.7,
        max_tokens=4096
    )
    
    audit_cfg = {
        "c1": True, "c2": True, "c3": True,
        "c1_window": 5, "c1_agreement_threshold": 0.4,
        "c2_e_thresh": 1.5, "c2_rv_thresh": 0.05,
        "c3_epsilon_floor": 0.01, "c3_ceiling": 0.8
    }
    audit = AgentAuditLayer(audit_cfg)
    
    eval_mode = "docker" if use_docker else "lightweight"
    
    results = []
    for problem in problems:
        print(f"Executing {problem.instance_id}...")
        try:
            episode = run_swe_reflexion_episode(
                problem=problem,
                llm=llm,
                max_attempts=7,
                audit=audit,
                seed=seed,
                eval_mode=eval_mode
            )
            results.append({
                "instance_id": episode.instance_id,
                "final_passed": episode.final_passed,
                "oscillation_index": episode.oscillation_index,
                "regression_count": episode.regression_count,
            })
            print(f" -> Passed: {episode.final_passed}, Regressions: {episode.regression_count}")
        except Exception as e:
            print(f" -> Execution Failed: {e}")
            
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/e5_docker_execution_results.csv", index=False)
    print("Saved -> results/e5_docker_execution_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=int, default=10)
    parser.add_argument("--no-docker", action="store_true")
    args = parser.parse_args()
    
    run_evaluation(num_problems=args.problems, use_docker=not args.no_docker)
