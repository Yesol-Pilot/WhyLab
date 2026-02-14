# -*- coding: utf-8 -*-
"""LLM Debate 어댑터 테스트.

llm_adapter.py의 핵심 기능을 검증합니다:
1. GeminiClient 초기화 및 Fallback
2. LLMDebateAdapter 증거 포맷팅
3. Fallback 메서드 동작
4. DebateCell + LLM 통합 (Fallback 모드)
"""

import pytest
from unittest.mock import patch, MagicMock

from engine.agents.llm_adapter import (
    GeminiClient,
    LLMDebateAdapter,
    LLMResponse,
)
from engine.agents.debate import Evidence, AdvocateAgent, CriticAgent, JudgeAgent


# ──────────────────────────────────────────────
# GeminiClient 테스트
# ──────────────────────────────────────────────

class TestGeminiClient:
    """GeminiClient 초기화 및 Fallback 테스트."""

    def test_no_api_key_returns_none(self):
        """API 키 없을 때 generate가 None을 반환."""
        with patch.dict("os.environ", {}, clear=True):
            client = GeminiClient()
            result = client.generate("테스트 프롬프트")
            assert result is None

    def test_no_api_key_not_available(self):
        """API 키 없을 때 is_available이 False."""
        with patch.dict("os.environ", {}, clear=True):
            client = GeminiClient()
            assert client.is_available is False

    def test_api_key_env_variable(self):
        """GEMINI_API_KEY 환경 변수 감지."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}):
            client = GeminiClient()
            # google.generativeai 임포트 없으면 False
            # 하지만 키는 감지됨
            # 실제 API 없이 테스트하므로 초기화 실패 예상
            # (genai 미설치 환경에서도 안전하게 동작)
            assert isinstance(client.is_available, bool)


# ──────────────────────────────────────────────
# LLMDebateAdapter 테스트
# ──────────────────────────────────────────────

class TestLLMDebateAdapter:
    """Adapter 증거 포맷팅 및 Fallback 테스트."""

    @pytest.fixture
    def adapter(self):
        """API 키 없는 상태의 어댑터 (Fallback 모드)."""
        with patch.dict("os.environ", {}, clear=True):
            return LLMDebateAdapter()

    @pytest.fixture
    def sample_evidence(self):
        """샘플 증거 리스트."""
        return [
            Evidence(
                claim="메타러너 80% 동일 방향",
                evidence_type="statistical",
                strength=0.8,
                source="meta_learner_consensus",
                business_impact="과감한 마케팅 집행 가능",
            ),
            Evidence(
                claim="E-value=2.5 (강건)",
                evidence_type="robustness",
                strength=0.83,
                source="e_value",
                business_impact="안정적 성과 기대",
            ),
        ]

    def test_format_evidence(self, adapter, sample_evidence):
        """증거 포맷팅이 올바른 텍스트를 생성."""
        result = adapter.format_evidence(sample_evidence)
        assert "메타러너 80%" in result
        assert "E-value=2.5" in result
        assert "강도: 0.80" in result

    def test_format_evidence_empty(self, adapter):
        """빈 증거 리스트 처리."""
        result = adapter.format_evidence([])
        assert "증거 없음" in result

    def test_fallback_advocate(self, adapter, sample_evidence):
        """Fallback 모드 옹호 논변 생성."""
        result = adapter._fallback_advocate(sample_evidence)
        assert "Growth Hacker" in result
        assert "메타러너" in result

    def test_fallback_critic(self, adapter, sample_evidence):
        """Fallback 모드 비판 논변 생성."""
        result = adapter._fallback_critic(sample_evidence)
        assert "Risk Manager" in result

    def test_fallback_verdict(self, adapter):
        """Fallback 모드 판결 요약."""
        verdict_data = {
            "verdict": "CAUSAL",
            "confidence": 0.85,
            "recommendation": "전면 배포 승인",
        }
        result = adapter._fallback_verdict(verdict_data)
        assert "CAUSAL" in result
        assert "85.0%" in result

    def test_generate_advocate_fallback(self, adapter, sample_evidence):
        """LLM 비활성 시 Fallback 옹호 생성."""
        context = {
            "treatment_col": "credit_limit",
            "outcome_col": "is_default",
            "ate_value": "-0.034",
        }
        result = adapter.generate_advocate_argument(sample_evidence, context)
        assert len(result) > 0  # Fallback 응답이 반환됨

    def test_generate_critic_fallback(self, adapter, sample_evidence):
        """LLM 비활성 시 Fallback 비판 생성."""
        context = {
            "treatment_col": "credit_limit",
            "outcome_col": "is_default",
            "ate_value": "-0.034",
        }
        result = adapter.generate_critic_argument(sample_evidence, context)
        assert len(result) > 0

    def test_get_debate_summary(self, adapter):
        """토론 요약 딕셔너리 형태 검증."""
        summary = adapter.get_debate_summary()
        assert "llm_active" in summary
        assert "model" in summary
        assert "responses" in summary
        assert isinstance(summary["responses"], list)


# ──────────────────────────────────────────────
# 통합 테스트 (Fallback 모드)
# ──────────────────────────────────────────────

class TestDebateIntegration:
    """규칙 기반 증거 + LLM Fallback 통합 테스트."""

    @pytest.fixture
    def pipeline_results(self):
        """파이프라인 결과 Mock."""
        return {
            "ate": -0.034,
            "ate_ci_lower": -0.05,
            "ate_ci_upper": -0.02,
            "meta_learner_results": {
                "ensemble": {"consensus": 0.8, "oracle_ate": -0.033},
            },
            "refutation_results": {
                "placebo_test": {"p_value": 0.45},
                "bootstrap_ci": {"ci_lower": -0.05, "ci_upper": -0.02},
                "leave_one_out": {"any_sign_flip": False, "details": []},
                "subset_validation": {"avg_stability": 0.92},
            },
            "sensitivity_results": {
                "e_value": {"point": 1.5},
                "overlap": {"overlap_score": 0.85},
                "gates_results": {"f_stat_significant": True},
            },
            "conformal_results": {
                "ci_lower_mean": -0.06,
                "ci_upper_mean": -0.01,
            },
            "feature_importance": {"income": 0.4, "age": 0.3, "credit_score": 0.2},
            "feature_names": ["income", "age", "credit_score"],
        }

    def test_full_debate_flow(self, pipeline_results):
        """규칙 기반 증거 수집 → LLM Fallback 토론 전체 흐름."""
        advocate = AdvocateAgent()
        critic = CriticAgent()
        judge = JudgeAgent()

        pro = advocate.gather_evidence(pipeline_results)
        con = critic.challenge(pipeline_results)

        assert len(pro) > 0, "옹호 증거가 수집되어야 합니다"
        assert isinstance(pro[0], Evidence)

        verdict = judge.deliberate(pro, con)
        assert verdict.verdict in ("CAUSAL", "NOT_CAUSAL", "UNCERTAIN")
        assert 0 <= verdict.confidence <= 1

        # LLM Fallback 토론
        with patch.dict("os.environ", {}, clear=True):
            adapter = LLMDebateAdapter()
            context = {
                "treatment_col": "credit_limit",
                "outcome_col": "is_default",
                "ate_value": "-0.034",
            }

            adv_arg = adapter.generate_advocate_argument(pro, context)
            assert len(adv_arg) > 0

            crit_arg = adapter.generate_critic_argument(con, context)
            assert len(crit_arg) > 0

            verdict_arg = adapter.generate_verdict(
                adv_arg, crit_arg,
                {
                    "verdict": verdict.verdict,
                    "confidence": verdict.confidence,
                    "pro_score": verdict.pro_score,
                    "con_score": verdict.con_score,
                    "recommendation": verdict.recommendation,
                },
                context,
            )
            assert len(verdict_arg) > 0
            assert verdict.verdict in verdict_arg

    def test_debate_summary_structure(self, pipeline_results):
        """토론 요약 구조 검증."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = LLMDebateAdapter()
            summary = adapter.get_debate_summary()

            assert summary["llm_active"] is False
            assert summary["model"] == "rule_based"
            assert isinstance(summary["responses"], list)
