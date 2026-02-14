/**
 * WhyLab 프로젝트 종합 지식 베이스 (Knowledge Base)
 *
 * 프로젝트의 아키텍처, 방법론, 사용법, 철학에 대한
 * 모든 지식을 구조화하여 챗봇이 접근할 수 있게 합니다.
 */

/* ──────────────────────────────────────
 * 1. 지식 항목 타입
 * ────────────────────────────────────── */
export interface KnowledgeEntry {
    keywords: string[];        // 매칭 키워드 (소문자)
    question: string;          // 대표 질문
    answer: string;            // 답변 (마크다운)
    category: "architecture" | "methodology" | "usage" | "philosophy" | "feature" | "team" | "comparison";
}

/* ──────────────────────────────────────
 * 2. 전체 지식 베이스
 * ────────────────────────────────────── */
export const KNOWLEDGE_BASE: KnowledgeEntry[] = [
    // ── 프로젝트 개요 ──
    {
        keywords: ["whylab", "프로젝트", "소개", "개요", "뭐야", "뭔가요", "무엇", "what is"],
        question: "WhyLab이 뭔가요?",
        answer: `🧬 **WhyLab**은 인과추론(Causal Inference) 기반 의사결정 지능 엔진입니다.\n\n` +
            `**핵심 가치:** "상관관계가 아닌 인과관계"로 비즈니스 의사결정을 지원합니다.\n\n` +
            `**주요 특징:**\n` +
            `- 11-Cell 모듈러 파이프라인 (데이터 → 추론 → 토론 → 판결)\n` +
            `- 멀티 에이전트 토론 시스템 (Growth Hacker vs Risk Manager)\n` +
            `- 7가지 메타러너 (S/T/X/DR/R-Learner + LinearDML + Oracle)\n` +
            `- 실시간 인과 드리프트 모니터링\n` +
            `- Next.js 인터랙티브 대시보드\n\n` +
            `MIT 라이선스 오픈소스이며, 학술 벤치마크(IHDP, ACIC, Jobs)에서 검증되었습니다.`,
        category: "philosophy",
    },
    {
        keywords: ["차별점", "차이", "다른", "unique", "경쟁", "비교", "vs", "causalml", "dowhy", "econml"],
        question: "다른 인과추론 도구와 무엇이 다른가요?",
        answer: `⚡ **WhyLab vs 기존 도구 비교:**\n\n` +
            `| 기능 | CausalML | DoWhy | EconML | **WhyLab** |\n` +
            `|---|:---:|:---:|:---:|:---:|\n` +
            `| Meta-Learner | 4종 | ✗ | 3종 | **7종** |\n` +
            `| AI Debate | ✗ | ✗ | ✗ | **✅ 3-Agent** |\n` +
            `| Conformal CI | ✗ | ✗ | ✗ | **✅** |\n` +
            `| Dashboard | ✗ | ✗ | ✗ | **✅ Next.js** |\n` +
            `| DB Connectors | ✗ | ✗ | ✗ | **CSV/SQL/BQ** |\n` +
            `| Drift Monitor | ✗ | ✗ | ✗ | **✅ 실시간** |\n\n` +
            `WhyLab의 핵심 차별점은 **"분석 → 해석 → 판단 → 모니터링"** 전체 사이클을 하나의 플랫폼에서 제공한다는 것입니다.`,
        category: "comparison",
    },

    // ── 아키텍처 ──
    {
        keywords: ["아키텍처", "구조", "architecture", "설계", "cell", "셀", "파이프라인", "pipeline"],
        question: "WhyLab의 아키텍처는 어떻게 되어 있나요?",
        answer: `🏗️ **11-Cell 모듈러 아키텍처**\n\n` +
            `WhyLab은 각 분석 단계를 독립적인 "셀(Cell)"로 분리합니다:\n\n` +
            `\`\`\`\n` +
            `DataCell → CausalCell → MetaLearnerCell → ConformalCell\n` +
            `    ↓                                           ↓\n` +
            `ExplainCell → RefutationCell → SensitivityCell\n` +
            `    ↓                                           ↓\n` +
            `VizCell → ExportCell → ReportCell → DebateCell\n` +
            `\`\`\`\n\n` +
            `**각 셀의 역할:**\n` +
            `- **DataCell**: 합성 데이터 생성 또는 외부 데이터(CSV/SQL/BigQuery) 로드\n` +
            `- **CausalCell**: Double ML 기반 ATE 추정 (Linear/Forest/Auto)\n` +
            `- **MetaLearnerCell**: 7종 메타러너로 CATE 추정\n` +
            `- **ConformalCell**: 분포 무관(Distribution-free) 개인별 신뢰구간\n` +
            `- **DebateCell**: Growth Hacker vs Risk Manager 토론 → 판결\n\n` +
            `모든 셀은 **Orchestrator**가 순서대로 조율합니다.`,
        category: "architecture",
    },
    {
        keywords: ["orchestrator", "오케스트레이터", "조율", "실행순서", "흐름"],
        question: "Orchestrator가 뭔가요?",
        answer: `🎼 **Orchestrator**는 파이프라인의 지휘자입니다.\n\n` +
            `11개 셀을 정해진 순서대로 실행하며, 각 셀의 출력을 다음 셀의 입력으로 전달합니다.\n\n` +
            `**핵심 기능:**\n` +
            `- 셀 간 의존성 자동 해결\n` +
            `- 실패 시 에러 로그 + 부분 결과 반환\n` +
            `- 시나리오 A/B 분기 처리\n\n` +
            `\`orchestrator.run_pipeline(scenario="A")\` 한 줄로 전체 파이프라인이 실행됩니다.`,
        category: "architecture",
    },

    // ── 핵심 방법론 ──
    {
        keywords: ["dml", "double machine learning", "이중기계학습", "causal", "추정방법"],
        question: "Double Machine Learning(DML)이 뭔가요?",
        answer: `📐 **Double Machine Learning (DML)**\n\n` +
            `DML은 Chernozhukov et al. (2018)이 제안한 인과 효과 추정 방법입니다.\n\n` +
            `**핵심 아이디어:**\n` +
            `1. **1단계**: ML로 처치(T)를 예측 → 잔차(residual) 추출\n` +
            `2. **2단계**: ML로 결과(Y)를 예측 → 잔차 추출\n` +
            `3. **3단계**: 두 잔차의 관계로 인과 효과 추정\n\n` +
            `**장점:**\n` +
            `- 고차원 교란변수 처리 가능\n` +
            `- 비선형 관계 포착\n` +
            `- √n-consistent (표본 크기에 따라 정확도 향상)\n\n` +
            `WhyLab은 Linear DML, Causal Forest DML, Auto DML 세 가지 변형을 지원합니다.`,
        category: "methodology",
    },
    {
        keywords: ["meta", "learner", "메타러너", "s-learner", "t-learner", "x-learner", "dr-learner", "r-learner"],
        question: "메타러너가 뭔가요?",
        answer: `🧠 **7종 메타러너 (Meta-Learner)**\n\n` +
            `메타러너는 기존 ML 모델을 "재활용"하여 개인별 처치 효과(CATE)를 추정합니다:\n\n` +
            `| 러너 | 전략 | 장점 |\n` +
            `|---|---|---|\n` +
            `| **S-Learner** | 하나의 모델로 전체 | 단순, 빠름 |\n` +
            `| **T-Learner** | 처치/통제 각각 모델 | 그룹별 최적화 |\n` +
            `| **X-Learner** | 교차 추정 | 표본 불균형에 강함 |\n` +
            `| **DR-Learner** | Doubly Robust | 이중 보호 |\n` +
            `| **R-Learner** | Robinson 분해 | 정규화 내장 |\n` +
            `| **LinearDML** | Double ML 기반 | 해석 가능 |\n` +
            `| **Oracle** | 앙상블 가중 평균 | 최고 성능 |\n\n` +
            `Oracle은 각 러너의 성능을 평가 후 가중 평균하는 WhyLab 고유 앙상블입니다.`,
        category: "methodology",
    },
    {
        keywords: ["cate", "개인", "이질", "heterogeneous", "개인화", "누구"],
        question: "CATE가 뭔가요?",
        answer: `🎯 **CATE (Conditional Average Treatment Effect)**\n\n` +
            `ATE가 "평균적으로 효과가 있는가?"라면, CATE는 **"누구에게 더 효과가 큰가?"**를 답합니다.\n\n` +
            `**예시:**\n` +
            `- 전체 평균(ATE): 쿠폰 효과 +5%\n` +
            `- 20대 남성(CATE): +12% (높음)\n` +
            `- 50대 여성(CATE): -2% (오히려 역효과)\n\n` +
            `WhyLab의 **CATE Explorer**에서 세그먼트별 효과를 시각화하고, 타겟팅 추천을 받을 수 있습니다.`,
        category: "methodology",
    },
    {
        keywords: ["conformal", "신뢰구간", "prediction", "ci", "confidence"],
        question: "Conformal Prediction이 뭔가요?",
        answer: `📏 **Conformal Prediction (적합 예측)**\n\n` +
            `분포 가정 없이 개인별 신뢰구간을 제공하는 방법입니다.\n\n` +
            `**기존 방법 vs Conformal:**\n` +
            `- 기존: "정규분포 가정 하에 95% CI"\n` +
            `- Conformal: "어떤 분포든, 95% 보장"\n\n` +
            `**WhyLab 적용:**\n` +
            `- ConformalCell이 각 개인의 CATE에 대해 신뢰구간을 생성\n` +
            `- Coverage Rate: 실제로 CI가 진짜 값을 포함하는 비율\n` +
            `- 목표: Coverage ≥ 95% (보통 97~99% 달성)`,
        category: "methodology",
    },

    // ── 토론 시스템 ──
    {
        keywords: ["토론", "debate", "에이전트", "agent", "growth", "risk", "판결", "verdict"],
        question: "멀티 에이전트 토론 시스템은 뭔가요?",
        answer: `⚖️ **3-Agent 인과 토론 시스템**\n\n` +
            `세 명의 AI 에이전트가 분석 결과를 두고 실제 조직의 의사결정을 시뮬레이션합니다:\n\n` +
            `1. **🚀 Growth Hacker** — 매출 기회를 극대화하는 관점\n` +
            `   - "ATE가 양수이고, ROI 시뮬레이션에서 수익이 나므로 전면 확대!"\n\n` +
            `2. **🛡️ Risk Manager** — 리스크를 최소화하는 관점\n` +
            `   - "E-value가 낮고, 특정 세그먼트에서 역효과가 우려됩니다"\n\n` +
            `3. **⚖️ Product Owner (Judge)** — 최종 판결\n` +
            `   - 🚀 **Rollout 100%**: 견고한 효과, 전면 시행\n` +
            `   - ⚖️ **A/B Test 5%**: 추가 검증 필요\n` +
            `   - 🛑 **Reject**: 효과 불확실 또는 역효과\n\n` +
            `이 구조는 단순 수치 보고를 넘어 **비즈니스 언어로 전환된 판단**을 제공합니다.`,
        category: "feature",
    },

    // ── 모니터링 ──
    {
        keywords: ["모니터링", "monitoring", "drift", "드리프트", "실시간", "alerter", "scheduler", "알림"],
        question: "실시간 모니터링은 어떻게 작동하나요?",
        answer: `📡 **실시간 인과 드리프트 모니터링**\n\n` +
            `인과 효과는 시간이 지나며 변할 수 있습니다. WhyLab은 이를 자동으로 감지합니다:\n\n` +
            `**3가지 감지 메트릭:**\n` +
            `1. **ATE 변화율**: 기준 대비 ±30% 이상 변동\n` +
            `2. **KL-Divergence**: CATE 분포의 변화\n` +
            `3. **부호 반전**: ATE가 양→음 또는 음→양\n\n` +
            `**작동 방식:**\n` +
            `\`\`\`\n` +
            `Scheduler → Pipeline 실행 → DriftDetector → Alerter\n` +
            `   (주기적)      (ATE/CATE)     (드리프트?)    (Slack/Console)\n` +
            `\`\`\`\n\n` +
            `CLI에서 \`--monitor --interval 30 --slack-webhook $URL\`로 바로 시작할 수 있습니다.`,
        category: "feature",
    },

    // ── 커넥터 ──
    {
        keywords: ["커넥터", "connector", "데이터소스", "csv", "sql", "bigquery", "parquet", "excel", "db", "데이터베이스"],
        question: "어떤 데이터 소스를 지원하나요?",
        answer: `🔌 **10가지 데이터 소스 지원:**\n\n` +
            `| 타입 | 설정 예시 |\n` +
            `|---|---|\n` +
            `| CSV | \`--data sales.csv\` |\n` +
            `| Parquet | \`--data data.parquet\` |\n` +
            `| TSV | \`--data data.tsv\` |\n` +
            `| Excel | \`--data report.xlsx\` |\n` +
            `| PostgreSQL | \`--data "postgresql://user:pw@host/db"\` |\n` +
            `| MySQL | \`--data "mysql://user:pw@host/db"\` |\n` +
            `| SQLite | \`--data "sqlite:///path.db"\` |\n` +
            `| BigQuery | \`--source-type bigquery --db-query "..."\` |\n\n` +
            `URI 패턴으로 자동 감지하며, \`--source-type\`으로 명시할 수도 있습니다.\n` +
            `팩토리 패턴으로 설계되어 새 커넥터 추가가 용이합니다.`,
        category: "feature",
    },

    // ── MCP ──
    {
        keywords: ["mcp", "서버", "server", "tool", "resource", "claude", "프로토콜"],
        question: "MCP 서버가 뭔가요?",
        answer: `🌐 **MCP (Model Context Protocol) 서버**\n\n` +
            `Claude 같은 외부 AI 에이전트가 WhyLab을 도구로 사용할 수 있게 하는 표준 프로토콜 서버입니다.\n\n` +
            `**제공 Tool (7개):**\n` +
            `1. \`run_analysis\` — 전체 파이프라인 실행\n` +
            `2. \`get_debate_verdict\` — AI 판결 조회\n` +
            `3. \`simulate_intervention\` — What-if 시뮬레이션\n` +
            `4. \`ask_rag\` — RAG 질의\n` +
            `5. \`compare_scenarios\` — 시나리오 비교\n` +
            `6. \`run_drift_check\` — 드리프트 체크\n` +
            `7. \`get_monitoring_status\` — 모니터링 상태\n\n` +
            `**제공 Resource (3개):**\n` +
            `- 데이터 메타데이터, 분석 리포트, 벤치마크 결과`,
        category: "feature",
    },

    // ── RAG ──
    {
        keywords: ["rag", "검색", "질의", "qa", "질문답변", "자연어", "에이전트", "페르소나", "persona"],
        question: "RAG 에이전트는 뭔가요?",
        answer: `💬 **RAG (Retrieval-Augmented Generation) 에이전트**\n\n` +
            `분석 결과를 자연어로 질문하고 답변받는 시스템입니다.\n\n` +
            `**3가지 페르소나:**\n` +
            `- 🚀 **Growth Hacker**: 매출 기회 중심 답변\n` +
            `- 🛡️ **Risk Manager**: 리스크 중심 답변\n` +
            `- ⚖️ **Product Owner**: 균형잡힌 종합 답변\n\n` +
            `**사용법:**\n` +
            `\`\`\`bash\n` +
            `python -m engine.cli --query "쿠폰 효과가 있어?" --persona growth_hacker\n` +
            `\`\`\`\n\n` +
            `멀티턴 대화를 지원하며, 분석 결과가 없으면 자동으로 파이프라인을 실행합니다.`,
        category: "feature",
    },

    // ── 사용법 ──
    {
        keywords: ["설치", "install", "시작", "start", "quickstart", "setup"],
        question: "어떻게 시작하나요?",
        answer: `🚀 **Quick Start:**\n\n` +
            `\`\`\`bash\n` +
            `# 1. 클론\n` +
            `git clone https://github.com/Yesol-Pilot/WhyLab.git\n` +
            `cd whylab\n\n` +
            `# 2. Python 환경\n` +
            `conda create -n whylab python=3.10\n` +
            `conda activate whylab\n` +
            `pip install -e .\n\n` +
            `# 3. 파이프라인 실행\n` +
            `python -m engine.cli --scenario A\n\n` +
            `# 4. 대시보드\n` +
            `cd dashboard && npm install && npm run dev\n` +
            `\`\`\`\n\n` +
            `선택적 DB 의존성: \`pip install "whylab[sql]"\` 또는 \`pip install "whylab[bigquery]"\``,
        category: "usage",
    },
    {
        keywords: ["cli", "명령어", "command", "플래그", "flag", "옵션"],
        question: "CLI 사용법을 알려주세요.",
        answer: `⌨️ **CLI v3 주요 플래그:**\n\n` +
            `\`\`\`bash\n` +
            `# 합성 데이터\n` +
            `python -m engine.cli --scenario A|B\n\n` +
            `# 외부 데이터\n` +
            `python -m engine.cli --data "file.csv" --treatment T --outcome Y\n\n` +
            `# DB 연결\n` +
            `python -m engine.cli --data "postgresql://..." --db-query "SELECT ..."\n\n` +
            `# RAG 질의\n` +
            `python -m engine.cli --query "쿠폰 효과?" --persona growth_hacker\n\n` +
            `# 모니터링\n` +
            `python -m engine.cli --monitor --interval 30 --slack-webhook $URL\n` +
            `\`\`\``,
        category: "usage",
    },
    {
        keywords: ["api", "3줄", "코드", "python", "사용예시", "예제", "analyze"],
        question: "Python API 사용법을 알려주세요.",
        answer: `🐍 **3-Line Python API:**\n\n` +
            `\`\`\`python\n` +
            `import whylab\n\n` +
            `results = whylab.analyze(\n` +
            `    data="sales.csv",\n` +
            `    treatment="coupon",\n` +
            `    outcome="purchase"\n` +
            `)\n\n` +
            `print(results.ate)      # 평균 처치 효과\n` +
            `print(results.verdict)  # AI 토론 판결\n` +
            `print(results.cate)     # 개인별 효과\n` +
            `\`\`\`\n\n` +
            `\`whylab.analyze()\`는 전체 11-Cell 파이프라인을 자동 실행하고, 결과를 구조화된 객체로 반환합니다.`,
        category: "usage",
    },

    // ── 시나리오 ──
    {
        keywords: ["시나리오a", "scenario a", "신용", "credit", "한도"],
        question: "시나리오 A는 뭔가요?",
        answer: `💳 **시나리오 A: 신용한도 인상 → 연체율**\n\n` +
            `- **처치(Treatment)**: 신용한도 인상 여부\n` +
            `- **결과(Outcome)**: 연체(default) 여부\n` +
            `- **질문**: "한도를 올리면 연체가 늘까?"\n\n` +
            `100명의 합성 데이터로 DML 추정. ATE가 음수이면 한도 인상이 연체를 줄이는 효과입니다.`,
        category: "usage",
    },
    {
        keywords: ["시나리오b", "scenario b", "쿠폰", "coupon", "마케팅"],
        question: "시나리오 B는 뭔가요?",
        answer: `🎟️ **시나리오 B: 쿠폰 발송 → 가입 전환**\n\n` +
            `- **처치(Treatment)**: 쿠폰 발송 여부\n` +
            `- **결과(Outcome)**: 가입 전환 여부\n` +
            `- **질문**: "쿠폰을 보내면 가입이 늘까?"\n\n` +
            `마케팅 캠페인 효과 측정 시나리오입니다. ATE가 양수이면 쿠폰이 전환을 높이는 효과입니다.`,
        category: "usage",
    },

    // ── 벤치마크 ──
    {
        keywords: ["ihdp", "acic", "jobs", "벤치마크", "benchmark", "학술", "성능", "pehe"],
        question: "벤치마크 결과가 어떻게 되나요?",
        answer: `🏆 **학술 벤치마크 검증 (3종 x 10 반복)**\n\n` +
            `| 데이터셋 | 최고 모델 | PEHE |\n` +
            `|---|---|---|\n` +
            `| **IHDP** | Oracle Ensemble | ~ 0.5 |\n` +
            `| **ACIC** | DR-Learner | 경쟁 수준 |\n` +
            `| **Jobs** | X-Learner | 안정적 |\n\n` +
            `IHDP(747 samples), ACIC(4802 samples), Jobs(722 samples) 3종 표준 벤치마크에서\n` +
            `CausalML, EconML 등과 동등 이상의 성능을 달성했습니다.`,
        category: "methodology",
    },

    // ── 철학 & 비전 ──
    {
        keywords: ["왜", "why", "철학", "vision", "비전", "목표", "목적"],
        question: "WhyLab의 비전은 무엇인가요?",
        answer: `🌟 **WhyLab의 비전: "Why"에 답하는 AI**\n\n` +
            `기존 데이터 분석은 "무엇이 일어났는가(What)"에 머뭅니다.\n` +
            `WhyLab은 **"왜 일어났는가(Why)"**에 답합니다.\n\n` +
            `**핵심 원칙:**\n` +
            `1. 🎯 **인과 > 상관**: 결정은 인과관계에 기반해야 합니다\n` +
            `2. 🤖 **AI + 인간**: AI는 분석하고, 인간은 판단합니다\n` +
            `3. 📊 **투명성**: 모든 결과는 검증 가능하고 설명 가능해야 합니다\n` +
            `4. 🔄 **연속 감시**: 인과 효과는 변하므로, 지속적으로 모니터링해야 합니다\n\n` +
            `> *"데이터가 아닌 의사결정을 혁신한다"* — WhyLab`,
        category: "philosophy",
    },
    {
        keywords: ["living", "ledger", "리빙레저"],
        question: "Living Ledger가 뭔가요?",
        answer: `📖 **Living Ledger** 는 WhyLab의 연구 비전 문서입니다.\n\n` +
            `"살아있는 장부"라는 이름처럼, 데이터가 자체적으로 인과관계를 기록하고 업데이트하는 시스템을 꿈꿉니다.\n\n` +
            `현재는 논문 레벨의 비전 문서로 존재하며, 아키텍처 문서(\`docs/architecture.md\`)에 WhyLab과의 매핑이 기록되어 있습니다.`,
        category: "philosophy",
    },

    // ── 기술 스택 ──
    {
        keywords: ["기술", "스택", "tech", "stack", "라이브러리", "의존성", "dependency"],
        question: "기술 스택이 뭔가요?",
        answer: `🔧 **기술 스택:**\n\n` +
            `**Engine (Python):**\n` +
            `- EconML, CausalML — 인과추론 핵심\n` +
            `- LightGBM (GPU 지원) — 메타러너 백본\n` +
            `- DuckDB — 제로카피 데이터 전처리\n` +
            `- SQLAlchemy — DB 커넥터\n` +
            `- ChromaDB — RAG 벡터 스토어\n\n` +
            `**Dashboard (TypeScript):**\n` +
            `- Next.js 16 + React 19\n` +
            `- Tailwind CSS v4\n` +
            `- Recharts — 데이터 시각화\n` +
            `- ReactFlow — DAG 시각화\n` +
            `- Framer Motion — 애니메이션\n\n` +
            `Python 3.9~3.13 호환, MIT 라이선스.`,
        category: "architecture",
    },

    // ── 대시보드 ──
    {
        keywords: ["대시보드", "dashboard", "ui", "프론트", "시각화", "visualization"],
        question: "대시보드에서 뭘 볼 수 있나요?",
        answer: `🖥️ **대시보드 구성 요소:**\n\n` +
            `1. **ROI Simulator** — 정책 강도 조절 → 이익/부실률 실시간 예측\n` +
            `2. **CATE Explorer** — 세그먼트별 개인 효과 분포\n` +
            `3. **Causal Graph (DAG)** — 인과 구조 시각화\n` +
            `4. **Stats Cards** — ATE, 샘플 수, 모델 타입 등 핵심 지표\n` +
            `5. **Conformal Band** — 개인별 신뢰구간 차트\n` +
            `6. **AI Debate Verdict** — Growth vs Risk 토론 결과\n` +
            `7. **Sensitivity Report** — E-value, Overlap, GATES\n` +
            `8. **Model Comparison** — 메타러너 성능 비교\n` +
            `9. **Chat Panel** — 저예요! 🤖 분석 결과에 대해 물어보세요\n\n` +
            `URL: [whylab.vercel.app](https://whylab.vercel.app/dashboard)`,
        category: "feature",
    },

    // ── 견고성 관련 ──
    {
        keywords: ["refutation", "검증", "placebo", "bootstrap", "random cause"],
        question: "견고성 검증은 어떻게 하나요?",
        answer: `🔬 **3가지 Refutation 검증:**\n\n` +
            `1. **Placebo Test**: 가짜 처치 변수로 대체 → 효과가 0에 가까운지 확인\n` +
            `   - Pass: 원래 결과가 우연이 아님\n\n` +
            `2. **Bootstrap CI**: 부트스트랩 리샘플링으로 신뢰구간 계산\n` +
            `   - 원래 ATE가 부트스트랩 CI 안에 있으면 안정적\n\n` +
            `3. **Random Common Cause**: 랜덤 교란 변수 추가\n` +
            `   - ATE가 크게 변하지 않으면 견고함\n\n` +
            `모든 검증은 RefutationCell에서 자동 실행되며, Status Cards에 Pass/Fail이 표시됩니다.`,
        category: "methodology",
    },
    {
        keywords: ["evalue", "e-value", "미관측", "unobserved", "교란"],
        question: "E-value가 뭔가요?",
        answer: `🔍 **E-value (Evidence Value)**\n\n` +
            `미관측 교란변수(Unobserved Confounder)에 대한 견고성 지표입니다.\n\n` +
            `**해석:**\n` +
            `- E-value = 3.0 → "관찰된 효과를 설명하려면, 미관측 교란이 처치/결과 모두에 3배 이상 영향을 미쳐야 함"\n` +
            `- 높을수록 견고 (≥2.0: 양호, ≥3.0: 강건)\n\n` +
            `WhyLab은 CI bound E-value도 함께 보고하여 보수적 평가를 제공합니다.`,
        category: "methodology",
    },

    // ── ROI Simulator ──
    {
        keywords: ["roi", "simulator", "시뮬레이터", "시뮬레이션", "what-if", "만약"],
        question: "ROI Simulator는 뭔가요?",
        answer: `💰 **ROI (Policy) Simulator**\n\n` +
            `"만약 정책 강도를 바꾸면 결과가 어떻게 될까?"를 실시간으로 시뮬레이션합니다.\n\n` +
            `**조절 가능한 파라미터:**\n` +
            `- 신용 한도 상향 크기 ($0 ~ $2,000)\n` +
            `- 타겟 유저 범위 (상위 5% ~ 전체)\n\n` +
            `**출력:**\n` +
            `- 예상 순이익 (Net Profit)\n` +
            `- 예상 부실률 (Default Rate)\n` +
            `- Profit Sensitivity Curve (수익 민감도 곡선)\n` +
            `- AI 에이전트 의견 (Growth Hacker vs Risk Manager)\n\n` +
            `슬라이더를 움직이며 실시간으로 변화를 확인할 수 있습니다.`,
        category: "feature",
    },
];

/* ──────────────────────────────────────
 * 3. 지식 매칭 엔진
 * ────────────────────────────────────── */
export function searchKnowledge(query: string): KnowledgeEntry | null {
    const q = query.toLowerCase();

    // 1-Pass: 매칭 스코어 계산
    let bestMatch: KnowledgeEntry | null = null;
    let bestScore = 0;

    for (const entry of KNOWLEDGE_BASE) {
        let score = 0;
        for (const kw of entry.keywords) {
            if (q.includes(kw)) {
                score += kw.length; // 긴 키워드 매치일수록 높은 점수
            }
        }
        if (score > bestScore) {
            bestScore = score;
            bestMatch = entry;
        }
    }

    // 최소 2글자 이상의 키워드가 매치되어야 반환
    return bestScore >= 2 ? bestMatch : null;
}

/* ──────────────────────────────────────
 * 4. 추천 질문 생성
 * ────────────────────────────────────── */
export const PROJECT_SUGGESTIONS = [
    "WhyLab이 뭔가요?",
    "다른 도구와 뭐가 달라요?",
    "메타러너가 뭔가요?",
    "멀티 에이전트 토론 시스템은?",
    "어떻게 시작하나요?",
    "모니터링은 어떻게 하나요?",
];
