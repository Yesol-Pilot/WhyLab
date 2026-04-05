# Docker SWE-bench Execution Experiment Design
## WhyLab: Ground-Truth Regression Detection via Full Test Execution

**Purpose**: Replace proxy-metric evaluation with Docker-based test execution to
produce ground-truth pass/fail signals, enabling detection of real pass-to-fail
regressions in the Reflexion self-improvement loop.

**Motivation**: All 8 NeurIPS reviewers identified the same critical flaw: the
proxy metric (file-overlap heuristic) records zero regressions across all
conditions including baseline. This means the metric structurally cannot detect
pass-to-fail transitions. Docker execution is the only path to credible results.

---

## 1. Problem Selection (N=75 problems)

### 1.1 Selection Strategy

Three tiers, totaling 75 problems:

| Tier | Count | Source | Rationale |
|------|-------|--------|-----------|
| A: Oscillation-prone | 45 | All 54 from e5_metrics.csv with oscillation_count > 0 minus 9 that fail Docker setup (see 1.3) | Highest prior probability of exhibiting pass-to-fail under real execution |
| B: High-attempt non-oscillating | 15 | Non-oscillating problems with mean attempts >= 4 under baseline | Problems where the agent struggles but proxy never flips; Docker may reveal hidden regressions |
| C: Easy controls | 15 | Non-oscillating, pass on attempt 1 under all seeds | Sanity check; should show ~0 regressions under Docker too |

### 1.2 Why 75, Not 50 or 100

- 50 problems x 3 seeds = 150 episodes per condition = 300 total. Marginal for
  McNemar's test on binary regression events.
- 75 problems x 3 seeds = 225 episodes per condition = 450 total. With expected
  regression rate of 10-20% under baseline (based on SWE-bench literature), this
  gives approximately 22-45 regression events per condition -- sufficient for
  chi-squared and Fisher's exact tests.
- 100 problems x 3 seeds = 300 episodes per condition = 600 total. Better power
  but pushes runtime and cost beyond budget.

### 1.3 Docker Feasibility Pre-filter

Before the main experiment, run a setup-only pass on all 54 oscillation-prone
problems (no LLM calls):

```
For each instance_id:
  1. Pull/build the SWE-bench Docker image for the repo+version
  2. Apply the GOLD patch + test patch
  3. Run the test suite
  4. Verify: gold patch passes, empty patch fails
```

Any instance that fails this pre-filter (image build failure, test timeout > 300s,
flaky gold patch) is excluded. Expect to lose ~9 instances based on known
SWE-bench Docker issues (astropy build failures, large scipy images).

---

## 2. Experimental Conditions

### 2.1 Conditions (2)

| Condition | Audit Config | Description |
|-----------|-------------|-------------|
| **baseline** | `audit=None` | Standard Reflexion loop, no safety gate |
| **whylab_c2** | `c1=False, c2=True, c3=False` | C2 sensitivity filter only (the paper's primary claim) |

Only two conditions, not the full 7-ablation sweep from E5. Rationale:
- The reviewer criticism is about ground truth, not ablation granularity.
- 2 conditions halves runtime and cost.
- If C2 shows a regression reduction, the paper's core claim is validated.
- Full ablation sweep can be run later on the same infrastructure.

### 2.2 Why Not Include C1+C2+C3 (full)?

The full_calibrated condition showed identical oscillation reduction to C2_calibrated
in the proxy experiment (70 vs 70 oscillations). Adding it provides no additional
statistical information for the paper's main claim. If reviewers request it in
revision, it can be added as a third condition without redesigning the experiment.

---

## 3. Reflexion Protocol

### 3.1 Episode Structure

Each episode: 1 problem x 1 seed x 1 condition.

```
for attempt in range(7):            # max 7 attempts (matching E5)
    patch = LLM_solve(problem, memory, seed)
    result = docker_test(problem, patch)   # <-- THE KEY CHANGE
    
    if audit and attempt > 0:
        decision = audit.evaluate_update(...)
        if not decision.accept:
            revert patch, skip memory update
    
    record(attempt, patch, result.passed, result.tests_passed, result.tests_total)
    
    if result.passed:
        break  # early exit on success
    
    reflection = LLM_reflect(problem, patch, result)
    update memory
```

### 3.2 Docker Test Function

Replace `_evaluate_lightweight` with `_evaluate_docker`. The existing
`_evaluate_docker` in `swebench_loader.py` (lines 384-448) uses the official
`swebench.harness.run_evaluation` CLI. This needs modifications:

**Current issues with the existing Docker function:**

1. **Single-instance overhead**: It calls `run_evaluation` per patch, which
   rebuilds the Docker image each time. For 7 attempts on the same problem,
   the image should be built once and reused.

2. **Output parsing**: Only checks for "RESOLVED" in stdout. Need to also parse
   per-test pass/fail for the `tests_total` and `tests_passed` fields.

3. **No container reuse**: Each call creates a fresh container. We need a
   persistent container per problem that we `docker exec` into.

**Proposed Docker evaluation architecture:**

```
Per problem (once):
  1. Build/pull SWE-bench instance image: swebench.harness.docker_build
  2. Create container from image, stopped
  3. Apply test patch (gold test additions) inside container

Per attempt (7x per episode):
  1. Start container from clean snapshot (docker commit + docker run)
  2. Copy generated patch into container
  3. git apply patch inside container
  4. Run test suite: pytest with SWE-bench test specification
  5. Capture exit code + stdout/stderr
  6. Parse: PASSED/FAILED per test, total count, pass count
  7. Stop container (do NOT remove -- reuse base for next attempt)
```

**Alternative (simpler, recommended for v1):**

Use the official `swebench` Python API directly instead of the CLI:

```python
from swebench.harness.test_spec import make_test_spec
from swebench.harness.docker_utils import (
    build_instance_image,
    run_instance_in_container,
)

# Build once per problem
test_spec = make_test_spec(instance)
build_instance_image(test_spec)

# Per attempt: run in container with generated patch
result = run_instance_in_container(test_spec, generated_patch, timeout=300)
# result contains: resolved (bool), test_output (str)
```

This delegates container lifecycle management to the official harness, which
handles image caching, test spec parsing, and cleanup.

### 3.3 Test Outcome Extraction

The SWE-bench harness produces a structured log. Ground truth determination:

```
RESOLVED   = all fail-to-pass tests now pass AND no pass-to-fail tests fail
FAILED     = at least one fail-to-pass test still fails OR a pass-to-fail test broke

For our regression metric:
  pass_to_fail_tests: tests that passed on base_commit but fail after patch
  This is EXACTLY what we need -- Docker gives us real regressions at the
  individual test level, not just problem level.
```

Key insight: SWE-bench test specs distinguish "fail_to_pass" tests (the bug
being fixed) from "pass_to_pass" tests (existing tests that must not break).
A patch that fixes the bug but breaks an existing test is NOT resolved -- this
is the regression signal the proxy metric was missing.

---

## 4. Data Flow Architecture

```
                    Linux Server (100.67.221.25)
                    8 CPU, 8GB RAM, Docker
                    
[Problem List]  -->  [Orchestrator]  -->  [LLM Client]  -->  Gemini API
  75 problems        run_docker_e5.py      (hybrid mode)
  3 seeds                |
  2 conditions           |
                         v
                    [Docker Harness]
                    swebench.harness
                         |
                    [Container Pool]
                    1 container at a time
                    (sequential execution)
                         |
                         v
                    [Results DB]
                    results/docker_e5/
                      checkpoints.jsonl    (per-episode append)
                      final_metrics.csv    (aggregated)
                      docker_logs/         (per-instance test output)
```

### 4.1 Orchestrator Design

```python
# Pseudocode for run_docker_e5.py

problems = load_selected_problems(75)
seeds = [42, 43, 44]
conditions = ["baseline", "whylab_c2"]

# Resume from checkpoint
completed = load_checkpoint()

work_items = []
for problem in problems:
    for seed in seeds:
        for condition in conditions:
            key = (problem.instance_id, seed, condition)
            if key not in completed:
                work_items.append(key)

# Sequential execution (1 container at a time due to 8GB RAM)
for instance_id, seed, condition in work_items:
    try:
        episode = run_episode(instance_id, seed, condition, eval_mode="docker")
        save_checkpoint(episode)
    except DockerError as e:
        log_failure(instance_id, seed, condition, e)
        continue  # skip, don't crash
```

### 4.2 Checkpoint/Resume System

Each completed episode is appended as a JSON line to `checkpoints.jsonl`:

```json
{
  "instance_id": "django__django-13265",
  "seed": 42,
  "condition": "baseline",
  "final_passed": false,
  "regression_count": 1,
  "oscillation_count": 2,
  "attempts": [
    {"idx": 0, "passed": false, "tests_passed": 3, "tests_total": 5, "time_ms": 45000},
    {"idx": 1, "passed": true,  "tests_passed": 5, "tests_total": 5, "time_ms": 38000},
    {"idx": 2, "passed": false, "tests_passed": 4, "tests_total": 5, "time_ms": 41000}
  ],
  "timestamp": "2026-03-31T14:00:00Z"
}
```

On restart, the orchestrator reads all lines from `checkpoints.jsonl`, builds
the set of completed (instance_id, seed, condition) tuples, and skips them.

File-level atomicity: each line is written with `flush()` + `os.fsync()` to
survive process crashes. No partial writes because each line is a complete
JSON object.

---

## 5. Cost Estimation

### 5.1 LLM API Costs (Gemini 2.0 Flash)

Gemini 2.0 Flash pricing (as of 2026-Q1):
- Input: $0.10 / 1M tokens
- Output: $0.40 / 1M tokens

Per episode (worst case, 7 attempts):
- Solve calls: 7 x (~2000 input + ~1500 output tokens) = 14K input, 10.5K output
- Reflect calls: 6 x (~1500 input + ~800 output tokens) = 9K input, 4.8K output
- Total per episode: ~23K input, ~15.3K output tokens

Total episodes: 75 problems x 3 seeds x 2 conditions = 450 episodes

But with caching: the baseline condition for seed=42 was already run in E5
(albeit with proxy eval). If prompts are identical, the LLM responses will be
cache hits for the solve/reflect calls. The Docker evaluation is separate from
LLM calls.

Worst-case (no cache hits):
- Input tokens: 450 x 23K = 10.35M tokens -> $1.04
- Output tokens: 450 x 15.3K = 6.89M tokens -> $2.76
- **Total LLM cost: ~$3.80**

Even with zero cache hits, LLM cost is negligible. The real cost is compute time.

### 5.2 Docker Compute Costs

The server is already provisioned (no cloud cost). The cost is wall-clock time
and electricity.

### 5.3 Total Cost: ~$4-10 (LLM) + $0 (compute, self-hosted)

Well within the $100 budget.

---

## 6. Runtime Estimation

### 6.1 Per-Episode Timing

| Phase | Time | Notes |
|-------|------|-------|
| Docker image build (first time per problem) | 2-8 min | Cached after first build; ~30 unique repos |
| Container startup per attempt | 10-30s | Includes git checkout + patch apply |
| Test execution per attempt | 30-120s | Varies by repo; Django tests ~60s, sympy ~90s |
| LLM solve call | 2-5s | Gemini Flash is fast |
| LLM reflect call | 1-3s | Shorter prompt |

Per episode (7 attempts, worst case):
- First attempt (with image build): 8 min + 2.5 min = 10.5 min
- Subsequent attempts (6x): 6 x 2.5 min = 15 min
- Total: ~25.5 min worst case

Per episode (average, 4 attempts, image cached):
- 4 x 2.5 min = 10 min

### 6.2 Total Runtime

**Sequential execution** (1 container at a time, safest for 8GB RAM):

- 450 episodes x 10 min average = 4,500 min = 75 hours

This exceeds the 72-hour budget by ~4%. Mitigation strategies:

1. **Partial parallelism**: Run 2 containers concurrently for small-repo
   problems (Django, pytest). Reduces to ~50 hours. Risky with 8GB RAM.

2. **Smart ordering**: Run image builds first (overnight, no LLM calls), then
   episodes. Image builds for 30 repos: ~2 hours. Episodes without build
   overhead: ~60 hours.

3. **Reduce to 60 problems**: 60 x 3 x 2 = 360 episodes x 10 min = 60 hours.
   Drops 5 from each tier (40 + 10 + 10).

4. **Reduce to 3 attempts instead of 7**: Average episode drops to ~5 min.
   Total: 450 x 5 = 37.5 hours. But reduces regression detection power since
   regressions need multiple attempts. NOT RECOMMENDED.

**Recommended approach**: Strategy 2 (smart ordering) + accept ~75 hours.
If the server is stable, 75 hours finishes within a long weekend. The 72-hour
budget is a soft constraint.

### 6.3 Timeline

```
Day 0 (setup):
  - Install swebench on server (pip install swebench)
  - Docker feasibility pre-filter (2 hours)
  - Finalize problem list
  
Day 1-3 (execution):
  - Image pre-build pass (2 hours)
  - Main experiment run (~75 hours, auto-resuming)
  
Day 4 (analysis):
  - Aggregate results
  - Statistical tests
  - Generate paper figures/tables
```

---

## 7. Statistical Power Analysis

### 7.1 Primary Hypothesis

H1: The baseline Reflexion loop exhibits a higher rate of test-suite regressions
(pass-to-fail transitions) than the WhyLab C2-audited loop.

### 7.2 Expected Effect Size

From SWE-bench literature on self-repair agents (Jimenez et al. 2024, Yang et
al. 2024): unaudited multi-attempt agents exhibit 10-25% pass-to-fail regression
rates when measured by actual test execution.

Our proxy metric found 0% -- but this is exactly the artifact we are fixing.

Conservative estimates:
- Baseline regression rate: p_base = 0.15 (15% of episodes have at least one regression)
- C2 regression rate: p_c2 = 0.05 (5% -- C2 blocks fragile updates)
- Difference: delta = 0.10

### 7.3 Power Calculation

Using McNemar's test (paired design -- same problem, same seed, different condition):

- N pairs = 75 problems x 3 seeds = 225 paired episodes
- Discordant pairs (where one condition regresses, other doesn't):
  Expected: ~0.15 x 225 - 0.05 x 225 = ~22.5 discordant pairs
- McNemar's test with 22 discordant pairs:
  Power at alpha=0.05: approximately 0.80 (adequate)

Using Fisher's exact test (unpaired):
- N_base = 225 episodes, N_c2 = 225 episodes
- Expected regressions: 34 (baseline) vs 11 (C2)
- Power at alpha=0.05: approximately 0.88

Using permutation test (non-parametric):
- 10,000 permutations of condition labels
- Power: approximately 0.85

### 7.4 Secondary Metrics

| Metric | Definition | Expected Pattern |
|--------|-----------|-----------------|
| Regression rate | episodes with >= 1 pass-to-fail / total episodes | baseline > C2 |
| Mean regressions per episode | total pass-to-fail transitions / episodes | baseline > C2 |
| Oscillation index | sign changes / (attempts - 1) | baseline > C2 |
| Final pass rate | episodes ending in pass / total | C2 >= baseline |
| Safe pass rate | pass with 0 regressions / total | C2 > baseline |

### 7.5 What If There Are Still Zero Regressions?

If Docker execution also shows zero regressions in baseline, this would mean:

1. Gemini 2.0 Flash on SWE-bench Lite genuinely does not produce regressions
   (patches either consistently fail or consistently pass).
2. The 7-attempt Reflexion window is too short for regression accumulation.
3. The problem selection does not include "regression-prone" tasks.

Mitigations:
- Tier B problems (high-attempt, non-oscillating) are chosen specifically to
  test scenario 3.
- With Docker, we can also measure **test-level regressions** (individual tests
  that break even if the overall verdict is FAIL). This is a finer signal than
  episode-level regressions.
- Honest reporting of null result + proxy/Docker correlation analysis is still
  a contribution (shows the proxy metric's validity).

---

## 8. Failure Modes and Mitigations

### 8.1 Technical Failures

| Failure Mode | Probability | Impact | Mitigation |
|---|---|---|---|
| Docker image build failure for some repos | High (20%) | Lose problems | Pre-filter pass; have 54 oscillation-prone, need 45 |
| Container OOM (8GB limit) | Medium (10%) | Episode crash | Sequential execution; per-container memory limit of 6GB; checkpoint/resume |
| Server restart mid-run | Medium (15% over 75h) | Lost in-progress episode | Checkpoint after each episode; systemd service with auto-restart |
| Gemini API rate limit | Low (5%) | Slowdown | Exponential backoff already in llm_client.py; 450 episodes over 75h = 6/hour, well under limits |
| Flaky tests (non-deterministic) | Medium (15%) | False regressions | Run each test verdict 2x if borderline; record raw test output for manual audit |
| SWE-bench harness version incompatibility | Medium | Blocked | Pin swebench version; test with 3 problems first |

### 8.2 Scientific Failures

| Failure Mode | Probability | Impact | Mitigation |
|---|---|---|---|
| Zero regressions even with Docker | 25% | Weakens paper | Report honestly; analyze test-level signal; proxy-Docker correlation is still a contribution |
| Too many Docker failures, N < 50 | 15% | Underpowered | Start with 75 problems; buffer for dropouts |
| C2 makes results WORSE (more regressions) | 5% | Contradicts paper | Investigate mechanism; C2 might be rejecting good patches; report transparently |
| All regressions are in Tier C (controls) | 10% | Unexpected | Would suggest regressions correlate with easy problems; interesting finding either way |

---

## 9. Implementation Plan

### Phase 1: Server Setup (Day 0, ~3 hours)

```bash
# On Linux server 100.67.221.25
ssh user@100.67.221.25

# 1. Install SWE-bench
pip install swebench==2.1  # pin version

# 2. Verify Docker
docker run hello-world
docker system info  # confirm 8GB+ available

# 3. Clone WhyLab experiment code
git clone <repo> /home/user/whylab
cd /home/user/whylab/experiments

# 4. Set environment
export GEMINI_API_KEY=<key>

# 5. Download SWE-bench Lite dataset
python -c "from swebench_loader import download_swebench_lite; download_swebench_lite()"
```

### Phase 2: Docker Feasibility Pre-filter (Day 0, ~2 hours)

```bash
python docker_prefilter.py --problems oscillation_prone_54.txt --timeout 300
# Output: feasible_problems.txt (expected: ~45 problems)
```

This script:
1. Iterates over the 54 oscillation-prone instance IDs
2. For each: builds the Docker image, applies gold patch, runs tests
3. Records: build time, test time, pass/fail, any errors
4. Outputs a filtered list of instances where Docker execution works

### Phase 3: Problem List Finalization (Day 0, ~30 min)

Using pre-filter results + e5_metrics.csv:
1. Take all feasible oscillation-prone problems (Tier A, target: 45)
2. Select 15 high-attempt non-oscillating problems (Tier B)
3. Select 15 easy control problems (Tier C)
4. Write final list to `docker_e5_problems.json`

### Phase 4: Main Experiment (Days 1-3, ~75 hours)

```bash
# Run as a tmux/screen session or systemd service
python run_docker_e5.py \
  --problems docker_e5_problems.json \
  --seeds 42,43,44 \
  --conditions baseline,whylab_c2 \
  --max-attempts 7 \
  --docker-timeout 300 \
  --checkpoint-dir results/docker_e5/ \
  --resume
```

The orchestrator:
1. Loads problem list and checkpoint file
2. Generates all work items (75 x 3 x 2 = 450)
3. Skips completed work items
4. For each remaining work item:
   a. Configures audit layer (None for baseline, C2 for whylab_c2)
   b. Runs `run_swe_reflexion_episode()` with `eval_mode="docker"`
   c. Appends result to checkpoint file
   d. Logs progress: `[147/450] django__django-13265 seed=42 baseline: PASS (3 attempts, 0 regressions)`

### Phase 5: Analysis (Day 4, ~4 hours)

```bash
python analyze_docker_e5.py --checkpoint results/docker_e5/checkpoints.jsonl
```

Produces:
1. `final_metrics.csv` -- one row per (instance_id, seed, condition)
2. `summary_table.tex` -- LaTeX table for the paper
3. `regression_comparison.png` -- bar chart: baseline vs C2 regression rates
4. `oscillation_comparison.png` -- oscillation index distributions
5. `proxy_docker_correlation.csv` -- for each episode, proxy result vs Docker result
6. `statistical_tests.txt` -- McNemar, Fisher, permutation test results

---

## 10. Key Metrics to Report in Paper

### 10.1 Primary Result Table

| Metric | Baseline | WhyLab C2 | p-value |
|--------|----------|-----------|---------|
| Regression rate (episode-level) | ? | ? | ? |
| Mean regressions per episode | ? | ? | ? |
| Safe pass rate | ? | ? | ? |

### 10.2 Proxy vs Docker Calibration Table

| | Docker PASS | Docker FAIL |
|---|---|---|
| Proxy PASS | True Positive | **False Positive** |
| Proxy FAIL | False Negative | True Negative |

This table directly addresses the reviewer criticism: it quantifies how often
the proxy metric disagrees with ground truth, and specifically whether the
proxy's zero-regression finding was a true zero or a measurement artifact.

### 10.3 Individual Regression Examples

For the paper's narrative, identify 2-3 specific instances where:
- Baseline shows a clear pass-to-fail regression under Docker
- C2 blocks the regressing update
- Include the actual test names and failure messages

These concrete examples are more convincing than aggregate statistics for
reviewers.

---

## 11. Code Changes Required

### 11.1 Modify `swebench_loader.py`

The existing `_evaluate_docker()` function (lines 384-448) needs:

1. **Use swebench Python API** instead of CLI subprocess call
2. **Parse per-test results** from the harness log, not just "RESOLVED" string
3. **Add container reuse** -- build image once per problem, run tests per attempt
4. **Add timeout handling** per test run (not per entire harness invocation)

### 11.2 New file: `run_docker_e5.py`

Main orchestrator script. Approximately 200 lines. Handles:
- Problem list loading
- Checkpoint/resume
- Sequential execution with progress logging
- Error handling and skip-on-failure
- Final aggregation

### 11.3 New file: `docker_prefilter.py`

Pre-filter script. Approximately 100 lines. Handles:
- Image build verification per problem
- Gold patch test verification
- Timing and feasibility reporting

### 11.4 New file: `analyze_docker_e5.py`

Analysis script. Approximately 150 lines. Handles:
- Checkpoint parsing
- Statistical tests
- Figure generation
- LaTeX table generation
- Proxy-Docker comparison

### 11.5 Modify `e5_swebench_benchmark.py`

Minor changes:
- Add `--seeds` argument (currently only single seed)
- Add `--conditions` argument
- Add `--checkpoint-dir` argument
- Wire up the checkpoint/resume logic

---

## 12. Risk Assessment Summary

**Probability of experiment producing usable results for paper: ~75%**

- 50% chance: Docker reveals real regressions, C2 reduces them -> paper is strongly strengthened (best case)
- 25% chance: Docker reveals real regressions, but C2 effect is not statistically significant -> paper reports honest null on regression reduction, but Docker calibration data is still valuable
- 15% chance: Docker shows zero regressions (same as proxy) -> paper reports proxy was actually correct; honest but does not help the score
- 10% chance: Technical failures reduce N below usable threshold -> partial results, less convincing

**The experiment is worth running regardless of outcome.** Even the worst case
(Docker confirms zero regressions) transforms the paper from "we didn't check"
to "we checked with ground truth and confirmed our proxy metric was reliable."
That alone addresses the reviewer criticism, even if it does not produce the
dramatic regression-reduction result we hope for.

---

## 13. Appendix: SWE-bench Docker Internals

### 13.1 How SWE-bench Docker Works

Each SWE-bench instance has a "test spec" that defines:
- Base Docker image (e.g., `python:3.9-slim`)
- Repository URL and base commit
- Environment setup commands (pip install, etc.)
- Test command (usually `pytest` with specific test files)
- Expected test results: `FAIL_TO_PASS` (tests the fix should make pass) and
  `PASS_TO_PASS` (tests that must remain passing)

The harness:
1. Builds a Docker image with the repo at base_commit + dependencies installed
2. Creates a container from that image
3. Applies the candidate patch inside the container
4. Runs the test command
5. Parses test output to determine RESOLVED/FAILED

### 13.2 Image Caching

SWE-bench Docker images are large (500MB-2GB per repo version). With 30 unique
repos across 75 problems, expect ~20-40GB of Docker images. The 8GB RAM server
needs adequate disk space (recommend 100GB free).

Images are cached by Docker's layer cache. After initial build, subsequent
problems from the same repo+version reuse the cached image. This is why smart
ordering (group problems by repo) reduces total runtime.

### 13.3 Container Lifecycle per Attempt

```
[Base Image] --docker create--> [Container]
     |                               |
     |                          docker cp patch.diff
     |                          docker exec: git apply patch.diff
     |                          docker exec: pytest ...
     |                          docker cp: test_output.log
     |                               |
     |                          docker rm (cleanup)
     |
     +---(reuse for next attempt)---+
```

Each attempt gets a fresh container from the same base image. This ensures
attempt isolation -- a broken patch in attempt 3 does not corrupt attempt 4.
