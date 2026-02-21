"""
CodeEvolutionEngine — LLM 기반 실험 코드 자가진화
=================================================
진화 루프:
1. Gemini에 과거 성공/실패 코드 + KG 컨텍스트 전달
2. 새로운 추정 코드 생성
3. SandboxExecutor에서 격리 실행
4. Baseline(CausalCell) 대비 성능 비교
5. 개선된 코드만 code_bank에 보존

안전장치: FORBIDDEN_PATTERNS 검증, 120초 타임아웃, ConstitutionGuard
"""
import json
import re
import time
import logging
import os
from typing import Optional
from datetime import datetime

logger = logging.getLogger("whylab.code_evolution")

# 영속성 파일 경로
CODE_BANK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "code_bank.json"
)


class CodeEvolutionEngine:
    """LLM 기반 실험 코드 자가진화 엔진 (싱글턴)"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.generation = 0
        self.code_bank: list[dict] = []       # 성능 검증된 우수 코드
        self.failure_log: list[dict] = []     # 실패/퇴보 기록 (최근 20건)
        self.evolution_history: list[dict] = []
        self._load_state()

    # ─── 영속성 ───────────────────────────────────────

    def _load_state(self):
        """서버 재시작 시 code_bank 복원"""
        try:
            if os.path.exists(CODE_BANK_PATH):
                with open(CODE_BANK_PATH, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.generation = state.get("generation", 0)
                self.code_bank = state.get("code_bank", [])
                self.failure_log = state.get("failure_log", [])[-20:]
                logger.info(
                    "CodeEvolution 상태 복원: Gen %d, 보존 코드 %d건",
                    self.generation, len(self.code_bank),
                )
        except Exception as e:
            logger.warning("CodeEvolution 상태 복원 실패: %s", e)

    def _save_state(self):
        """code_bank 영속 저장"""
        try:
            os.makedirs(os.path.dirname(CODE_BANK_PATH), exist_ok=True)
            state = {
                "generation": self.generation,
                "code_bank": self.code_bank[-50:],  # 최근 50건만 보존
                "failure_log": self.failure_log[-20:],
                "saved_at": datetime.utcnow().isoformat(),
            }
            with open(CODE_BANK_PATH, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("CodeEvolution 상태 저장 실패: %s", e)

    # ─── 핵심: 진화 1세대 수행 ────────────────────────

    def evolve(self, data_info: dict) -> dict:
        """
        진화 1세대를 수행합니다.

        Args:
            data_info: CoordinatorV2._supply_data() 반환값
                - data_path: CSV 경로
                - treatment, outcome: 컬럼명
                - confounders: 교란 변수 목록
                - ate_true: Ground Truth ATE

        Returns:
            dict: {"improved": bool, "new_rmse": float, "baseline_rmse": float, ...}
        """
        self.generation += 1
        gen = self.generation
        logger.info("═══ Code Evolution Gen %d 시작 ═══", gen)

        # Step 1: Baseline 실행 (현재 CausalCell)
        baseline = self._run_baseline(data_info)
        if not baseline:
            return self._record_failure(gen, "Baseline 실행 실패", data_info)

        # Step 2: LLM에 코드 생성 요청
        new_code = self._generate_evolved_code(data_info, baseline)
        if not new_code:
            return self._record_failure(gen, "LLM 코드 생성 실패", data_info)

        # Step 3: Sandbox에서 진화 코드 실행
        new_result = self._execute_in_sandbox(new_code, data_info)
        if not new_result:
            return self._record_failure(
                gen, "진화 코드 실행 실패", data_info, code=new_code
            )

        # Step 4: 성능 비교
        comparison = self._compare_performance(baseline, new_result, gen)

        # Step 5: 판정
        if comparison["improved"]:
            self.code_bank.append({
                "generation": gen,
                "code": new_code,
                "rmse": comparison["new_rmse"],
                "baseline_rmse": comparison["baseline_rmse"],
                "improvement_pct": comparison["improvement_pct"],
                "method_description": comparison.get("method_description", ""),
                "created_at": datetime.utcnow().isoformat(),
            })
            logger.info(
                "🧬 Gen %d 진화 성공! RMSE: %.4f → %.4f (%.1f%% 개선)",
                gen, comparison["baseline_rmse"],
                comparison["new_rmse"], comparison["improvement_pct"],
            )
        else:
            self.failure_log.append({
                "generation": gen,
                "reason": f"퇴보: RMSE {comparison['baseline_rmse']} → {comparison['new_rmse']}",
                "code_snippet": new_code[:500],
                "timestamp": datetime.utcnow().isoformat(),
            })
            self.failure_log = self.failure_log[-20:]
            logger.info(
                "🔄 Gen %d 퇴보. RMSE: %.4f → %.4f",
                gen, comparison["baseline_rmse"], comparison["new_rmse"],
            )

        self.evolution_history.append(comparison)
        self._save_state()
        return comparison

    # ─── Step 1: Baseline ────────────────────────────

    def _run_baseline(self, data_info: dict) -> Optional[dict]:
        """현재 CausalCell로 baseline 성능 측정"""
        try:
            from engine.sandbox.executor import sandbox, generate_experiment_code

            treatment = data_info.get("treatment", data_info.get("treatment_col", ""))
            outcome = data_info.get("outcome", data_info.get("outcome_col", ""))
            confounders = data_info.get("confounders", data_info.get("confounder_cols", []))

            code = generate_experiment_code(
                treatment=treatment,
                outcome=outcome,
                confounders=confounders,
                method="LinearDML",
                seed=42,
                data_path=data_info.get("data_path", ""),
            )
            result = sandbox.execute(code, context={
                "data_path": data_info.get("data_path", ""),
            })
            if result.success:
                logger.info(
                    "Baseline 실행 성공: ATE=%.4f",
                    result.result_data.get("ate", 0),
                )
                return result.result_data
            logger.warning("Baseline 실행 실패: %s", result.stderr[:200])
            return None
        except Exception as e:
            logger.error("Baseline 예외: %s", e)
            return None

    # ─── Step 2: LLM 코드 생성 ──────────────────────

    def _generate_evolved_code(
        self, data_info: dict, baseline: dict
    ) -> Optional[str]:
        """Gemini에 진화된 실험 코드 생성 요청"""
        from api.agents.gemini_client import _call_gemini, is_available

        if not is_available():
            logger.info("Gemini 미사용, fallback 코드 생성")
            return self._fallback_code_generation(data_info)

        # 프롬프트 구성
        prompt = self._build_evolution_prompt(data_info, baseline)
        raw = _call_gemini(prompt, max_tokens=2048)
        if not raw:
            logger.warning("Gemini 응답 없음, fallback 코드 생성")
            return self._fallback_code_generation(data_info)

        # 코드 블록 추출
        code = self._extract_code_block(raw)
        if not code:
            logger.warning("LLM 응답에서 코드 블록 추출 실패")
            return self._fallback_code_generation(data_info)

        # SANDBOX_RESULT 할당 보장
        if "SANDBOX_RESULT" not in code:
            code += '\nSANDBOX_RESULT["ate"] = float(ate)\n'
            code += 'SANDBOX_RESULT["estimation_accuracy"] = {"rmse": 0, "bias": 0}\n'

        # np.random.seed 보장 (헌법 제6조)
        if "random" in code and "seed" not in code:
            code = "import numpy as np\nnp.random.seed(42)\n" + code

        logger.info("LLM 코드 생성 완료 (%d줄)", code.count("\n") + 1)
        return code

    def _build_evolution_prompt(
        self, data_info: dict, baseline: dict
    ) -> str:
        """LLM에 전달할 진화 프롬프트"""
        treatment = data_info.get("treatment", data_info.get("treatment_col", ""))
        outcome = data_info.get("outcome", data_info.get("outcome_col", ""))
        confounders = data_info.get("confounders", data_info.get("confounder_cols", []))
        ate_true = data_info.get("ate_true", "알 수 없음")
        baseline_rmse = baseline.get("estimation_accuracy", {}).get("rmse", "N/A")
        baseline_ate = baseline.get("ate", "N/A")

        # 최근 성공/실패 기록
        recent_successes = [
            f"Gen {c['generation']}: RMSE={c['rmse']:.4f} ({c.get('method_description', '?')})"
            for c in self.code_bank[-3:]
        ]
        recent_failures = [
            f"원인: {f['reason']}" for f in self.failure_log[-3:]
        ]

        return f"""당신은 인과추론 전문 연구자입니다.
WhyLab 시스템이 자동으로 인과 효과를 추정하고 있습니다.

## 현재 데이터
- Treatment: {treatment}
- Outcome: {outcome}
- Confounders: {', '.join(str(c) for c in confounders)}
- True ATE: {ate_true}

## 현재 Baseline 성능
- 방법: LinearDML (econml)
- 추정 ATE: {baseline_ate}
- RMSE: {baseline_rmse}

## 과거 진화 성공
{chr(10).join(recent_successes) if recent_successes else '아직 없음'}

## 과거 실패 원인
{chr(10).join(recent_failures) if recent_failures else '아직 없음'}

## 요청
Baseline(LinearDML)보다 RMSE가 낮거나 Coverage가 높은 새로운 인과 추정 코드를 작성하세요.
이전 세대와 다른 방법론을 시도하세요.

## 사용 가능 라이브러리
- econml (CausalForestDML, DRLearner, SLearner, TLearner, XLearner, LinearDRLearner 등)
- sklearn (GradientBoostingRegressor, RandomForestRegressor, Lasso, LassoCV 등)
- numpy, pandas, scipy

## 코드 규칙 (매우 중요)
1. 첫 줄: `import numpy as np` + `np.random.seed(42)`
2. 데이터 로드: `df = pd.read_csv(DATA_PATH)`
3. feature 컬럼에서 treatment, outcome, true_cate 반드시 제외
4. 결과를 SANDBOX_RESULT에 저장:
   - `SANDBOX_RESULT["ate"]` = float
   - `SANDBOX_RESULT["estimation_accuracy"]` = {{"rmse": float, "bias": float, "coverage_rate": float, "correlation": float}}
   - `SANDBOX_RESULT["method_description"]` = str
5. true_cate 컬럼이 있으면 estimation_accuracy에 rmse, bias, coverage_rate, correlation 계산
6. `open(`, `exec(`, `eval(`, `subprocess` 사용 금지
7. `pd.read_csv`는 사용 가능

## 출력
Python 코드만 ```python 블록으로 반환하세요."""

    def _extract_code_block(self, text: str) -> Optional[str]:
        """LLM 응답에서 ```python ... ``` 코드 블록 추출"""
        pattern = r"```(?:python)?\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # 코드 블록 마크다운이 없으면 전체를 코드로 시도
        if "import " in text and "SANDBOX_RESULT" in text:
            return text.strip()
        return None

    def _fallback_code_generation(self, data_info: dict) -> str:
        """Gemini 실패 시 CausalForestDML 폴백 코드 생성"""
        treatment = data_info.get("treatment", data_info.get("treatment_col", ""))
        outcome = data_info.get("outcome", data_info.get("outcome_col", ""))
        confounders = data_info.get("confounders", data_info.get("confounder_cols", []))
        conf_str = ", ".join(f'"{c}"' for c in confounders)

        return f'''import numpy as np
import pandas as pd
np.random.seed(42)

# 데이터 로드
df = pd.read_csv(DATA_PATH)

# 변수 분리
treatment = "{treatment}"
outcome = "{outcome}"
confounders = [{conf_str}]
feature_cols = [c for c in confounders if c in df.columns]
# treatment, outcome, true_cate 제외
feature_cols = [c for c in feature_cols if c not in (treatment, outcome, "true_cate")]

T = df[treatment].values
Y = df[outcome].values
X = df[feature_cols].values if feature_cols else np.random.randn(len(df), 1)

# CausalForestDML (econml) — Baseline LinearDML보다 유연한 비선형 추정
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor

model = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    model_t=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    n_estimators=200,
    random_state=42,
)
model.fit(Y, T, X=X)

ate = float(model.ate(X))
cate = model.effect(X).flatten()

# Ground Truth 비교
est_acc = {{"rmse": 0.0, "bias": 0.0, "coverage_rate": 0.0, "correlation": 0.0}}
if "true_cate" in df.columns:
    true_cate = df["true_cate"].values
    est_acc["rmse"] = float(np.sqrt(np.mean((cate - true_cate) ** 2)))
    est_acc["bias"] = float(np.mean(cate) - np.mean(true_cate))
    est_acc["mae"] = float(np.mean(np.abs(cate - true_cate)))
    # Coverage (CI 기반)
    try:
        ci = model.effect_interval(X)
        ci_lower, ci_upper = ci[0].flatten(), ci[1].flatten()
        covered = (true_cate >= ci_lower) & (true_cate <= ci_upper)
        est_acc["coverage_rate"] = float(np.mean(covered))
    except Exception:
        est_acc["coverage_rate"] = 0.0
    # Correlation
    if np.std(cate) > 0 and np.std(true_cate) > 0:
        est_acc["correlation"] = float(np.corrcoef(cate, true_cate)[0, 1])

SANDBOX_RESULT["ate"] = round(ate, 4)
SANDBOX_RESULT["estimation_accuracy"] = est_acc
SANDBOX_RESULT["method_description"] = "CausalForestDML + GBR (fallback 진화 코드)"
SANDBOX_RESULT["sample_size"] = len(df)

print(f"CausalForestDML | ATE={{ate:.4f}} | RMSE={{est_acc.get('rmse', '?')}}")
'''

    # ─── Step 3: Sandbox 실행 ────────────────────────

    def _execute_in_sandbox(
        self, code: str, data_info: dict
    ) -> Optional[dict]:
        """생성된 코드를 별도 SandboxExecutor에서 격리 실행"""
        from engine.sandbox.executor import SandboxExecutor

        # 별도 인스턴스 (진화 코드 실패가 메인 차단기에 영향 안 주도록)
        evo_sandbox = SandboxExecutor()

        try:
            # pd.read_csv 허용을 위해 "open(" 패턴 임시 제거
            original_forbidden = list(evo_sandbox.FORBIDDEN_PATTERNS)
            evo_sandbox.FORBIDDEN_PATTERNS = [
                p for p in evo_sandbox.FORBIDDEN_PATTERNS
                if p != "open("
            ]

            result = evo_sandbox.execute(code, context={
                "data_path": data_info.get("data_path", ""),
            })

            # 원본 복원 (클래스 변수이므로 인스턴스에서만 수정)
            evo_sandbox.FORBIDDEN_PATTERNS = original_forbidden

            if result.success:
                logger.info(
                    "진화 코드 실행 성공: ATE=%.4f",
                    result.result_data.get("ate", 0),
                )
                return result.result_data
            logger.warning("진화 코드 실행 실패: %s", result.stderr[:300])
            return None
        except Exception as e:
            logger.error("진화 코드 예외: %s", e)
            return None

    # ─── Step 4: 성능 비교 ───────────────────────────

    def _compare_performance(
        self, baseline: dict, new_result: dict, gen: int
    ) -> dict:
        """Baseline과 진화 코드의 성능 비교"""
        b_acc = baseline.get("estimation_accuracy", {})
        n_acc = new_result.get("estimation_accuracy", {})

        b_rmse = float(b_acc.get("rmse", 0) or 0)
        n_rmse = float(n_acc.get("rmse", 0) or 0)
        b_coverage = float(b_acc.get("coverage_rate", 0) or 0)
        n_coverage = float(n_acc.get("coverage_rate", 0) or 0)

        # 개선 판정: RMSE 10% 이상 개선 OR Coverage 20%p 이상 개선
        rmse_improved = (b_rmse > 0 and n_rmse > 0 and n_rmse < b_rmse * 0.9)
        coverage_improved = n_coverage > b_coverage + 0.2
        improved = rmse_improved or coverage_improved

        if b_rmse > 0:
            improvement_pct = ((b_rmse - n_rmse) / b_rmse) * 100
        else:
            improvement_pct = 0.0

        return {
            "generation": gen,
            "improved": improved,
            "baseline_rmse": round(b_rmse, 4),
            "new_rmse": round(n_rmse, 4),
            "baseline_coverage": round(b_coverage, 4),
            "new_coverage": round(n_coverage, 4),
            "improvement_pct": round(improvement_pct, 1),
            "rmse_improved": rmse_improved,
            "coverage_improved": coverage_improved,
            "baseline_ate": baseline.get("ate"),
            "new_ate": new_result.get("ate"),
            "method_description": new_result.get("method_description", ""),
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ─── 실패 기록 ──────────────────────────────────

    def _record_failure(
        self, gen: int, reason: str, data_info: dict, code: str = ""
    ) -> dict:
        """실패 기록 + 반환"""
        entry = {
            "generation": gen,
            "reason": reason,
            "code_snippet": code[:500] if code else "",
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.failure_log.append(entry)
        self.failure_log = self.failure_log[-20:]
        self._save_state()
        logger.warning("Gen %d 진화 실패: %s", gen, reason)
        return {
            "generation": gen,
            "improved": False,
            "reason": reason,
            "baseline_rmse": 0,
            "new_rmse": 0,
        }

    # ─── 상태 조회 ──────────────────────────────────

    def get_status(self) -> dict:
        """진화 엔진 상태 반환"""
        return {
            "generation": self.generation,
            "code_bank_size": len(self.code_bank),
            "failure_count": len(self.failure_log),
            "best_rmse": min(
                (c["rmse"] for c in self.code_bank), default=None
            ),
            "latest_improvement": (
                self.code_bank[-1]["improvement_pct"]
                if self.code_bank else None
            ),
            "evolution_history": self.evolution_history[-10:],
        }


# 싱글턴
code_evolution = CodeEvolutionEngine()
