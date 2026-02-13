# WhyLab 워크스페이스 룰

## 프로젝트 목적: AI 기반 데이터 의사결정

### Why — 이 프로젝트가 존재하는 이유
- 데이터 분석 현장에서 **"상관관계 ≠ 인과관계"**를 구분하지 못해 잘못된 의사결정이 반복됨
- 인과추론은 전문가만 다룰 수 있다는 진입 장벽 존재
- **AI가 이 진입 장벽을 낮추고, 의사결정까지의 과정을 자동화**할 수 있음

### What — 무엇을 하는가
- 합성 데이터 → AI 인과추론 → 민감도 검증 → 시각화 대시보드까지 **E2E 자동 파이프라인**
- 의사결정자가 코드 없이도 인과관계를 이해할 수 있는 **인터랙티브 대시보드**
- AI가 생성한 ATE/CATE 해석을 **자연어로 제공** (`ate.description`)

### How — AI가 어떻게 돕는가
1. **AutoML 모델 선택**: LinearDML, CausalForest, DRLearner 중 최적 모델 자동 선택
2. **이질적 효과 분석**: 세그먼트별 CATE 자동 추정 (소득/연령/신용 구간)
3. **자연어 해석**: ATE 결과를 비전문가도 이해할 수 있는 한 줄 설명으로 변환
4. **민감도 자동 검증**: Placebo Test + Random Common Cause로 결과 신뢰성 자동 검증
5. **의사결정 시뮬레이터**: What-If 분석으로 정책 변경 영향을 사전 예측

---

## 설계 원칙

1. **의사결정 중심**: 모든 기능은 "이걸 보면 어떤 결정을 내릴 수 있는가?"로 평가
2. **AI 자동화 우선**: 수동 분석 단계를 AI로 대체할 수 있으면 반드시 자동화
3. **해석 가능성**: 모든 AI 분석 결과에 자연어 해석(description) 필수 제공
4. **신뢰성 검증**: 결과를 제시할 때 민감도 분석 결과를 함께 표시
5. **재현 가능성**: 동일 입력 → 동일 출력을 보장 (random_seed 고정)

---

## 프로젝트 정체성
- **프로젝트명**: WhyLab — "왜?"에 과학적으로 답하는 AI 인과추론 플랫폼
- **한 줄**: AI가 상관관계와 인과관계를 분리하고, 액션 가능한 인사이트를 자동으로 생성하는 의사결정 지원 플랫폼
- **타깃**: PM, 데이터 분석가, 의사결정자
- **형태**: 학술 논문(White Paper) + 재현 가능 코드 + 인터랙티브 대시보드
- **저장소**: `D:\00.test\PAPER\WhyLab`

---

## 필수 규칙

### 1. 언어
- 모든 코드 주석, 커밋, 문서: **한국어**
- 수학 용어/변수명: 영문 원문 유지 (CATE, DML, Treatment)
- README.md: **영문 + 한국어 병기** (GitHub 글로벌 노출)

### 2. 3-Layer 구조
- `paper/`: 논문 + Jupyter 분석 노트북 (분석가 관점)
- `engine/`: Python 인과추론 엔진 (빌더 관점)
- `dashboard/`: Next.js 인터랙티브 대시보드 (배포 관점)

### 3. Python (`engine/` + `paper/`)
- Python 3.11+, econml + lightgbm + DuckDB
- **모든 함수에 타입 힌트 필수**
- 독스트링: Google 스타일, 한국어 + 영문 수학 용어
- SQL 분석: DuckDB (로컬, 설치 불필요)
- 매직 넘버 금지 → `WhyLabConfig`로 추출

### 4. Next.js 대시보드 (`dashboard/`)
- Next.js 16+ App Router, TypeScript 필수
- 정적 export (`output: "export"`) + Client Component
- React Flow → DAG 시각화 | Recharts → 차트 | Framer Motion → 애니메이션
- 다크모드 필수 (Glassmorphism 디자인 시스템)

### 5. 논문 (`paper/`)
- Jupyter 노트북 (.ipynb)은 **반드시 재현 가능**해야 함
- Figure 번호와 코드 셀이 1:1 매핑
- SQL 쿼리는 DuckDB로 노트북 내에서 실행

### 6. 수학적 엄밀함
- DML 구조 방정식: Y = θ(X)·T + g(X,W) + ε
- CATE 추정값에는 반드시 95% 신뢰구간 포함
- Ground Truth CATE 포함한 합성 데이터로 검증
- AI AutoML: 최소 3개 모델 자동 비교 후 최적 선택

### 7. Git
- 커밋: `<타입>(<범위>): <한국어 설명>`
- 타입: feat, fix, docs, test, style, refactor, chore
- 범위: paper, engine, dashboard, ci, docs

### 8. 보안
- API 키/시크릿 커밋 금지
- 합성 데이터만 사용 (실제 개인정보 절대 금지)
- GitHub Pages 정적 배포만
