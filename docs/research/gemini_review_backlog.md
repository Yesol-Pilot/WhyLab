# WhyLab v2.0 — Gemini 리뷰 종합 (4차 누적) ✅ 전수 완료

> 4차 리뷰까지의 모든 지적사항 — **9건 전수 해결 (2026-02-25)**

## 🔴 P0 — 논문 수락 필수 (AAAI 2027) ✅ ALL DONE

| # | 지적 | 현황 | 커밋 |
|---|---|---|---|
| R1 | DI 가중치 하드코딩 | ✅ 정보이론 동적 가중치 + 스무딩/캡 | `84c7d2e` |
| R2 | DML 미관측 교란 방어 | ✅ E-value + Partial R² | `f4ee448` |
| R3 | ARES 검증 모듈 | ✅ ARES + CausalFlip + Beta-Binomial CI | `e6b2031` |
| R4 | Who&When 벤치마크 | ✅ 합성 시나리오 + SOTA 비교 실행기 | `70834e1` |
| R5 | ζ 수렴 수학적 증명 | ✅ Lyapunov 안정성 + LyapunovFilter | `e6b2031` |

## 🟡 P1 — 프로덕션 안정성 ✅ ALL DONE

| # | 지적 | 현황 | 커밋 |
|---|---|---|---|
| C1 | 통합 테스트 100% Mock | ✅ 카오스 시뮬레이션 10건 | `70834e1` |
| C2 | Supabase 파티셔닝 미적용 | ✅ 네이티브 파티셔닝 + 롤업 | `8f0611d` |
| C3 | Decision 로그 누락 위험 | ✅ Outbox 패턴 (WAL+DLQ) | `70834e1` |
| C4 | tracing.py 레거시 | ✅ OTel + 동적 샘플링 | `8f0611d` |

> 4차 리뷰까지의 모든 지적사항을 단일 백로그로 통합

## 🔴 P0 — 논문 수락 필수 (AAAI 2027)

| # | 지적 | 현황 | 대응 |
|---|---|---|---|
| R1 | DI 가중치(0.4/0.3) 하드코딩 → 수학적 근거 없음 | 미착수 | Wasserstein/KL-D 또는 Sensitivity Analysis |
| R2 | DML 미관측 교란 변수 방어 없음 | 미착수 | E-value / Gaussian Copula 민감도 분석 |
| R3 | ARES 검증 모듈 누락 | 미착수 | LLM-as-Judge 교차 검증 + CausalFlip 함정 테스트 |
| R4 | Who&When 벤치마크 미검증 | 미착수 | Blame Attribution SOTA 비교 (o1: 15%, Claude: 25% 기준) |
| R5 | ζ 수렴 수학적 증명 없음 | 미착수 | Lyapunov 안정성 정리(Theorem) 한 단락 |

## 🟡 P1 — 프로덕션 안정성

| # | 지적 | 현황 | 대응 |
|---|---|---|---|
| C1 | 통합 테스트 100% Mock | 미착수 | toxiproxy + GA4 503/지연 주입 |
| C2 | Supabase 파티셔닝 미적용 | 미착수 | pg_partman 주간 파티션 + 3개월 아카이빙 |
| C3 | Decision 로그 누락 위험 | 미착수 | Transactional Outbox 패턴 |
| C4 | tracing.py 레거시 | 인지 | OTel 전환 (Phase 4) |
