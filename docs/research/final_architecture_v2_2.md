# WhyLab v2.0 — 최종 아키텍처 설계서 (KDD/AAAI 수락 타겟팅)

> **버전**: v2.2 Final | **작성일**: 2026-02-24  
> **목적**: 심사위원별 공격 방어 전략 + 객관적 성능 지표 기반 아키텍처 확정

---

## 논문 초점 (2축)

1. **Blame Attribution** — MACIE(SCM+Shapley)로 에이전트별 인과적 책임 할당
2. **Closed-Loop Damping** — ARES 신뢰도에 정비례하는 ζ로 정책 붕괴 방어

## 심사위원별 방어 전략

| 심사위원 | 공격 포인트 | 방어 기제 |
|---|---|---|
| KDD ADS | API 할당량 한계, 데이터 희소 | GA4 Lazy Fetching + GSC 노이즈 평활화 |
| AAAI | 다중 에이전트 환각, 책임 모호 | MACIE Shapley + ARES 교차 검증 |

## 핵심 성능 지표 (리서치 기반)

| 모듈 | 지표 | 값 |
|---|---|---|
| MACIE | 평균 절대 속성 정확도 |φᵢ| | 5.07 (σ < 0.05) |
| MACIE | 연산 시간 (CPU) | 0.79초/데이터셋 |
| ARES | 추론 단계 검증 신뢰구간 | ≥ 95% |
| GSC | 부트스트랩 신뢰구간 방식 | Frequentist, 시계열 상관 보존 |
| Damping | ζ 조절 기준 | ARES score에 정비례 |

## AAAI 2027 역산 타임라인

| 마일스톤 | 추정일 |
|---|---|
| Phase 1~3 파이프라인 완비 | 2026-05-10 |
| 실증 데이터 수집 시작 | 2026-05-10 |
| AAAI 2027 초록 등록 | 2026-06-29 |
| AAAI 2027 본문 제출 (11p) | 2026-07-06 |
| 리뷰 결과 통보 | 2026-09-07 |

## 논문 제목 (가제)

"Stabilizing Closed-Loop Self-Improvement in Multi-Agent Systems via Causal Blame Attribution and Adaptive Damping"

## 기여점 집중 (Kitchen Sink 방지)

1. MACIE + ARES → 다중 에이전트 환각 합의 차단 (수학적 증명)
2. 동적 ζ → 정책 붕괴 실증 방어 (실가동 데이터)
3. 오픈소스 벤치마크: No-audit vs CausalImpact vs WhyLab v2.0
