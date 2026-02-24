# WhyLab v2.0 — 고도화 설계서 (Gemini 심층 비평 반영)

> **문서 유형**: 아키텍처 고도화 리뷰  
> **작성일**: 2026-02-24  
> **목적**: 초기 설계의 Kitchen Sink 위험 해소, 논문 초점 재조정

---

## 핵심 재조정

초기 설계의 문제: DML, DAG-SHAP, GSC, 제어 이론을 모두 나열 → Kitchen Sink 위험

**재조정된 2축 초점:**
1. **Blame Attribution** — 다중 에이전트의 개별 책임을 MACIE(SCM+Shapley)로 분리
2. **Closed-Loop Damping** — 잘못된 피드백으로 인한 정책 붕괴를 ζ로 제어

---

## 1. Blame Attribution: ECHO/MACIE 프레임워크

- **MACIE**: SCM + Shapley Values → 개별 에이전트의 인과적 기여도 + 창발적 시너지 정량화
- **ECHO**: 계층적 컨텍스트 표현 + 객관적 합의 → 오류 추적 정확도 극대화
- **기존 DML과의 관계**: DML = 거시적 효과 분리, MACIE = 미시적 에이전트 책임 분배

## 2. LLM-as-Judge 환각 방어: ARES 프레임워크

- **문제**: 다수 LLM이 동일 편향 공유 → 환각적 합의(Confabulation consensus)
- **대안**: 다중 에이전트 토론(Multi-agent debate) + ARES 확률 검증
- **WhyLab 연결점**: 기존 DebateCell + JudgeAgent 아키텍처와 자연스럽게 통합 가능
- **목표**: 95% 신뢰 구간 확보

## 3. Sim2Real 갭 → Damping 격상

- **기존**: DampingController = 보조 모듈
- **고도화**: 시스템 코어 엔진으로 격상
- **ζ 계산**: 불확실성(GSC CI 폭, ARES score)에 '반비례'하도록 동적 설정

## 4. Phase 4 논문 전략 재조정

**가제**: "Stabilizing Closed-Loop Self-Improvement in Multi-Agent Systems via Causal Blame Attribution and Adaptive Damping"

**기여점 2축 (Kitchen Sink 방지)**:
1. ARES 기반 교차 검증으로 LLM-as-Judge 오류 전파 해결
2. 동적 ζ로 정책 붕괴 통제 — 실가동 데이터 실증

**타겟**: KDD 2027 Applied Data Science 또는 AAAI 2027
