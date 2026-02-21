"""
ConstitutionGuard — 연구 헌법 런타임 가드레일 (Sprint 29)
=========================================================
연구 헌법(Research Constitution v1.0)을 코드 레벨에서 강제합니다.

하드 인터셉터(Hard Interceptor):
- 제1조: 반증 테스트 2개 이상 통과 검증
- 제4조: 다원적 방법론 교차 검증 강제
- 제5조: 표본 크기 기준 자동 적용
- 제6조: 시드 고정 검증 (SandboxExecutor에서 처리)
- 제12조: 메서드 다양성 보장 (70% 집중 방지)
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger("whylab.constitution")


class AnalysisLevel(Enum):
    """헌법 제5조: 표본 크기에 따른 분석 허용 수준."""
    EXPLORATORY_ONLY = "exploratory"  # n < 500: 인과 주장 금지
    LOW_POWER = "low_power"          # 500 ≤ n < 2000: 경고 부착
    FULL_ANALYSIS = "full"           # n ≥ 2000: 정식 분석 허용


class HypothesisGrade(Enum):
    """헌법 제7조: 가설 품질 등급."""
    S = "superior"      # KG + Gemini + 문헌 3건 이상 → 즉시 실험
    A = "acceptable"    # KG + 문헌 1건 이상 → 실험 후 추가 검증
    B = "baseline"      # 템플릿 기반 → Critic 사전 검토 필수
    F = "fail"          # 근거 없는 추측 → 폐기


@dataclass
class GuardVerdict:
    """가드레일 검증 결과."""
    passed: bool
    violations: list[str]
    warnings: list[str]
    analysis_level: AnalysisLevel
    metadata: dict

    @property
    def can_proceed(self) -> bool:
        """Critical 위반이 없으면 진행 가능."""
        return self.passed

    def summary(self) -> str:
        """검증 요약 문자열."""
        status = "✅ PASS" if self.passed else "🚫 BLOCKED"
        parts = [f"[Constitution Guard] {status}"]
        if self.violations:
            parts.append(f"  위반: {', '.join(self.violations)}")
        if self.warnings:
            parts.append(f"  경고: {', '.join(self.warnings)}")
        parts.append(f"  분석 수준: {self.analysis_level.value}")
        return "\n".join(parts)


class ConstitutionGuard:
    """
    연구 헌법을 런타임에 강제 실행하는 미들웨어.
    
    모든 실험 결과가 Coordinator에게 전달되기 전에
    이 가드를 통과해야 합니다.
    """

    # ── 제5조: 표본 크기 임계값 ──
    SAMPLE_SIZE_MIN = 500
    SAMPLE_SIZE_RECOMMENDED = 2000

    # ── 제1조: 반증 테스트 최소 통과 수 ──
    MIN_REFUTATION_PASSED = 2

    # ── 제4조: 최소 방법론 수 ──
    MIN_METHODS_COUNT = 2

    # ── 제12조: 메서드 집중도 상한 ──
    METHOD_CONCENTRATION_LIMIT = 0.7

    @staticmethod
    def check_sample_size(n: int) -> AnalysisLevel:
        """
        제5조: 표본 크기 기준.
        
        - n < 500: 인과 주장 금지 (탐색적 분석만 허용)
        - 500 ≤ n < 2000: LOW_POWER 경고 부착
        - n ≥ 2000: 정식 분석 허용
        """
        if n < ConstitutionGuard.SAMPLE_SIZE_MIN:
            return AnalysisLevel.EXPLORATORY_ONLY
        elif n < ConstitutionGuard.SAMPLE_SIZE_RECOMMENDED:
            return AnalysisLevel.LOW_POWER
        return AnalysisLevel.FULL_ANALYSIS

    @staticmethod
    def check_multi_method(methods_used: set) -> bool:
        """
        제4조: 다원적 검증 원칙.
        최소 2개 이상의 독립적 방법론이 사용되었는지 확인.
        """
        return len(methods_used) >= ConstitutionGuard.MIN_METHODS_COUNT

    @staticmethod
    def check_refutation(passed_count: int) -> bool:
        """
        제1조: 반증 테스트 최소 2개 통과 여부 확인.
        Placebo, Random Common Cause, Bootstrap 중 2개.
        """
        return passed_count >= ConstitutionGuard.MIN_REFUTATION_PASSED

    @staticmethod
    def check_method_diversity(method_usage: dict) -> tuple[bool, Optional[str]]:
        """
        제12조: 메서드 다양성 보장.
        특정 메서드가 70% 이상 선택되면 경고.
        
        Args:
            method_usage: {"T-Learner": 15, "DML": 3, "PSM": 2}
            
        Returns:
            (다양성 충족 여부, 과집중된 메서드명 or None)
        """
        total = sum(method_usage.values())
        if total == 0:
            return True, None
        
        for method, count in method_usage.items():
            if count / total > ConstitutionGuard.METHOD_CONCENTRATION_LIMIT:
                return False, method
        
        return True, None

    @staticmethod
    def check_experiment_source(source: str) -> bool:
        """
        실험 결과가 실제 엔진에서 나온 것인지 확인.
        시뮬레이션 결과는 경고 태깅.
        """
        return source == "engine"

    @classmethod
    def validate_experiment(
        cls,
        sample_size: int,
        methods_used: set,
        refutation_passed: int,
        experiment_source: str,
        method_usage: Optional[dict] = None,
    ) -> GuardVerdict:
        """
        실험 결과에 대한 종합 헌법 검증.
        
        모든 하드 제약을 한 번에 검사하고 GuardVerdict를 반환합니다.
        """
        violations = []
        warnings = []
        metadata = {}

        # ── 제5조: 표본 크기 ──
        analysis_level = cls.check_sample_size(sample_size)
        metadata["sample_size"] = sample_size
        metadata["analysis_level"] = analysis_level.value

        if analysis_level == AnalysisLevel.EXPLORATORY_ONLY:
            violations.append(
                f"제5조 위반: n={sample_size} < {cls.SAMPLE_SIZE_MIN}. "
                "인과 주장 금지, 탐색적 분석만 허용됩니다."
            )
        elif analysis_level == AnalysisLevel.LOW_POWER:
            warnings.append(
                f"제5조 경고: n={sample_size} < {cls.SAMPLE_SIZE_RECOMMENDED}. "
                "⚠️ LOW POWER 경고가 부착됩니다."
            )

        # ── 제4조: 다원적 검증 ──
        if not cls.check_multi_method(methods_used):
            violations.append(
                f"제4조 위반: {len(methods_used)}개 방법론만 사용됨. "
                f"최소 {cls.MIN_METHODS_COUNT}개 필요합니다."
            )
        metadata["methods_used"] = list(methods_used)

        # ── 제1조: 반증 테스트 ──
        if not cls.check_refutation(refutation_passed):
            violations.append(
                f"제1조 위반: 반증 테스트 {refutation_passed}개 통과. "
                f"최소 {cls.MIN_REFUTATION_PASSED}개 필요합니다. "
                "결과는 '상관관계 수준'으로 강등됩니다."
            )
        metadata["refutation_passed"] = refutation_passed

        # ── 실행 출처 확인 ──
        if not cls.check_experiment_source(experiment_source):
            warnings.append(
                f"⚠️ SIMULATED: 실험 출처가 '{experiment_source}'입니다. "
                "실제 엔진 실행 결과가 아닙니다."
            )
        metadata["experiment_source"] = experiment_source

        # ── 제12조: 메서드 다양성 ──
        if method_usage:
            diverse, concentrated = cls.check_method_diversity(method_usage)
            if not diverse:
                warnings.append(
                    f"제12조 경고: '{concentrated}' 메서드가 70% 이상 선택됨. "
                    "강제 탐색(Exploration)을 권고합니다."
                )
            metadata["method_diversity"] = diverse

        # 종합 판정
        passed = len(violations) == 0
        verdict = GuardVerdict(
            passed=passed,
            violations=violations,
            warnings=warnings,
            analysis_level=analysis_level,
            metadata=metadata,
        )

        # 로깅
        if passed:
            logger.info("헌법 검증 통과: %s", verdict.summary())
        else:
            logger.warning("헌법 검증 실패: %s", verdict.summary())

        return verdict


# 모듈 레벨 싱글턴
guard = ConstitutionGuard()
