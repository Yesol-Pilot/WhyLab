/**
 * WhyLab Project Comprehensive Knowledge Base
 *
 * Structures all knowledge about the project's architecture,
 * methodology, usage, and philosophy for chatbot access.
 */

/* ──────────────────────────────────────
 * 1. Knowledge Entry Type
 * ────────────────────────────────────── */
export interface KnowledgeEntry {
    keywords: string[];        // matching keywords (lowercase)
    question: string;          // representative question
    answer: string;            // answer (markdown)
    category: "architecture" | "methodology" | "usage" | "philosophy" | "feature" | "team" | "comparison";
}

/* ──────────────────────────────────────
 * 2. Full Knowledge Base
 * ────────────────────────────────────── */
export const KNOWLEDGE_BASE: KnowledgeEntry[] = [
    // ── Project Overview ──
    {
        keywords: ["whylab", "project", "introduction", "overview", "what is"],
        question: "What is WhyLab?",
        answer: `🧬 **WhyLab** is a causal inference-based decision intelligence engine.\n\n` +
            `**Core Value:** Supports business decision-making through "causation, not just correlation."\n\n` +
            `**Key Features:**\n` +
            `- 11-Cell Modular Pipeline (Data → Inference → Debate → Verdict)\n` +
            `- Multi-Agent Debate System (Growth Hacker vs Risk Manager)\n` +
            `- 7 Meta-Learners (S/T/X/DR/R-Learner + LinearDML + Oracle)\n` +
            `- Real-time Causal Drift Monitoring\n` +
            `- Next.js Interactive Dashboard\n\n` +
            `MIT licensed open source, validated on academic benchmarks (IHDP, ACIC, Jobs).`,
        category: "philosophy",
    },
    {
        keywords: ["difference", "unique", "compare", "comparison", "vs", "causalml", "dowhy", "econml"],
        question: "How is WhyLab different from other causal inference tools?",
        answer: `⚡ **WhyLab vs Existing Tools:**\n\n` +
            `| Feature | CausalML | DoWhy | EconML | **WhyLab** |\n` +
            `|---|:---:|:---:|:---:|:---:|\n` +
            `| Meta-Learner | 4 types | ✗ | 3 types | **7 types** |\n` +
            `| AI Debate | ✗ | ✗ | ✗ | **✅ 3-Agent** |\n` +
            `| Conformal CI | ✗ | ✗ | ✗ | **✅** |\n` +
            `| Dashboard | ✗ | ✗ | ✗ | **✅ Next.js** |\n` +
            `| DB Connectors | ✗ | ✗ | ✗ | **CSV/SQL/BQ** |\n` +
            `| Drift Monitor | ✗ | ✗ | ✗ | **✅ Real-time** |\n\n` +
            `WhyLab's key differentiator is providing the entire **"Analysis → Interpretation → Judgment → Monitoring"** cycle in a single platform.`,
        category: "comparison",
    },

    // ── Architecture ──
    {
        keywords: ["architecture", "structure", "design", "cell", "pipeline"],
        question: "What is WhyLab's architecture?",
        answer: `🏗️ **11-Cell Modular Architecture**\n\n` +
            `WhyLab separates each analysis stage into independent "Cells":\n\n` +
            `\`\`\`\n` +
            `DataCell → CausalCell → MetaLearnerCell → ConformalCell\n` +
            `    ↓                                           ↓\n` +
            `ExplainCell → RefutationCell → SensitivityCell\n` +
            `    ↓                                           ↓\n` +
            `VizCell → ExportCell → ReportCell → DebateCell\n` +
            `\`\`\`\n\n` +
            `**Role of each Cell:**\n` +
            `- **DataCell**: Synthetic data generation or external data (CSV/SQL/BigQuery) loading\n` +
            `- **CausalCell**: Double ML-based ATE estimation (Linear/Forest/Auto)\n` +
            `- **MetaLearnerCell**: CATE estimation with 7 meta-learners\n` +
            `- **ConformalCell**: Distribution-free individual confidence intervals\n` +
            `- **DebateCell**: Growth Hacker vs Risk Manager debate → verdict\n\n` +
            `All cells are orchestrated sequentially by the **Orchestrator**.`,
        category: "architecture",
    },
    {
        keywords: ["orchestrator", "orchestration", "execution order", "flow"],
        question: "What is the Orchestrator?",
        answer: `🎼 **Orchestrator** is the conductor of the pipeline.\n\n` +
            `It executes 11 cells in a prescribed order, passing each cell's output as input to the next.\n\n` +
            `**Key Features:**\n` +
            `- Automatic dependency resolution between cells\n` +
            `- Error logging + partial result return on failure\n` +
            `- Scenario A/B branching support\n\n` +
            `A single call to \`orchestrator.run_pipeline(scenario="A")\` runs the entire pipeline.`,
        category: "architecture",
    },

    // ── Core Methodology ──
    {
        keywords: ["dml", "double machine learning", "causal", "estimation method"],
        question: "What is Double Machine Learning (DML)?",
        answer: `📐 **Double Machine Learning (DML)**\n\n` +
            `DML is a causal effect estimation method proposed by Chernozhukov et al. (2018).\n\n` +
            `**Core Idea:**\n` +
            `1. **Stage 1**: Predict treatment (T) with ML → extract residual\n` +
            `2. **Stage 2**: Predict outcome (Y) with ML → extract residual\n` +
            `3. **Stage 3**: Estimate causal effect from the relationship between the two residuals\n\n` +
            `**Advantages:**\n` +
            `- Handles high-dimensional confounders\n` +
            `- Captures non-linear relationships\n` +
            `- √n-consistent (accuracy improves with sample size)\n\n` +
            `WhyLab supports three variants: Linear DML, Causal Forest DML, and Auto DML.`,
        category: "methodology",
    },
    {
        keywords: ["meta", "learner", "meta-learner", "s-learner", "t-learner", "x-learner", "dr-learner", "r-learner"],
        question: "What are Meta-Learners?",
        answer: `🧠 **7 Meta-Learners**\n\n` +
            `Meta-learners "repurpose" existing ML models to estimate individual treatment effects (CATE):\n\n` +
            `| Learner | Strategy | Advantage |\n` +
            `|---|---|---|\n` +
            `| **S-Learner** | Single model for all | Simple, fast |\n` +
            `| **T-Learner** | Separate models for treated/control | Group-optimal |\n` +
            `| **X-Learner** | Cross estimation | Robust to sample imbalance |\n` +
            `| **DR-Learner** | Doubly Robust | Double protection |\n` +
            `| **R-Learner** | Robinson decomposition | Built-in regularization |\n` +
            `| **LinearDML** | Double ML-based | Interpretable |\n` +
            `| **Oracle** | Ensemble weighted average | Best performance |\n\n` +
            `Oracle is a WhyLab-specific ensemble that evaluates each learner's performance and computes a weighted average.`,
        category: "methodology",
    },
    {
        keywords: ["cate", "individual", "heterogeneous", "personalized", "who"],
        question: "What is CATE?",
        answer: `🎯 **CATE (Conditional Average Treatment Effect)**\n\n` +
            `If ATE answers "Is there an effect on average?", CATE answers **"Who benefits more?"**\n\n` +
            `**Example:**\n` +
            `- Overall average (ATE): Coupon effect +5%\n` +
            `- Males in 20s (CATE): +12% (high)\n` +
            `- Females in 50s (CATE): -2% (adverse effect)\n\n` +
            `WhyLab's **CATE Explorer** visualizes effects by segment and provides targeting recommendations.`,
        category: "methodology",
    },
    {
        keywords: ["conformal", "confidence interval", "prediction", "ci", "confidence"],
        question: "What is Conformal Prediction?",
        answer: `📏 **Conformal Prediction**\n\n` +
            `A method that provides individual confidence intervals without distribution assumptions.\n\n` +
            `**Traditional vs Conformal:**\n` +
            `- Traditional: "95% CI under normal distribution assumption"\n` +
            `- Conformal: "95% guaranteed regardless of distribution"\n\n` +
            `**WhyLab Application:**\n` +
            `- ConformalCell generates confidence intervals for each individual's CATE\n` +
            `- Coverage Rate: the proportion of CIs that contain the true value\n` +
            `- Target: Coverage ≥ 95% (typically achieves 97~99%)`,
        category: "methodology",
    },

    // ── Debate System ──
    {
        keywords: ["debate", "agent", "growth", "risk", "verdict", "multi-agent"],
        question: "What is the Multi-Agent Debate System?",
        answer: `⚖️ **3-Agent Causal Debate System**\n\n` +
            `Three AI agents simulate real organizational decision-making over analysis results:\n\n` +
            `1. **🚀 Growth Hacker** — Maximizes revenue opportunities\n` +
            `   - "ATE is positive and ROI simulation shows profit — full rollout!"\n\n` +
            `2. **🛡️ Risk Manager** — Minimizes risk\n` +
            `   - "E-value is low and adverse effects are a concern in certain segments"\n\n` +
            `3. **⚖️ Product Owner (Judge)** — Final verdict\n` +
            `   - 🚀 **Rollout 100%**: Robust effect, full deployment\n` +
            `   - ⚖️ **A/B Test 5%**: Additional validation needed\n` +
            `   - 🛑 **Reject**: Uncertain effect or adverse impact\n\n` +
            `This structure goes beyond simple numerical reporting to provide **judgment translated into business language**.`,
        category: "feature",
    },

    // ── Monitoring ──
    {
        keywords: ["monitoring", "drift", "real-time", "alerter", "scheduler", "alert"],
        question: "How does real-time monitoring work?",
        answer: `📡 **Real-time Causal Drift Monitoring**\n\n` +
            `Causal effects can change over time. WhyLab automatically detects such changes:\n\n` +
            `**3 Detection Metrics:**\n` +
            `1. **ATE Change Rate**: ±30% or more deviation from baseline\n` +
            `2. **KL-Divergence**: Changes in CATE distribution\n` +
            `3. **Sign Reversal**: ATE flipping from positive→negative or vice versa\n\n` +
            `**How It Works:**\n` +
            `\`\`\`\n` +
            `Scheduler → Pipeline Run → DriftDetector → Alerter\n` +
            `   (periodic)    (ATE/CATE)     (drift?)     (Slack/Console)\n` +
            `\`\`\`\n\n` +
            `Start from CLI with \`--monitor --interval 30 --slack-webhook $URL\`.`,
        category: "feature",
    },

    // ── Connectors ──
    {
        keywords: ["connector", "data source", "csv", "sql", "bigquery", "parquet", "excel", "db", "database"],
        question: "What data sources are supported?",
        answer: `🔌 **10 Data Sources Supported:**\n\n` +
            `| Type | Configuration Example |\n` +
            `|---|---|\n` +
            `| CSV | \`--data sales.csv\` |\n` +
            `| Parquet | \`--data data.parquet\` |\n` +
            `| TSV | \`--data data.tsv\` |\n` +
            `| Excel | \`--data report.xlsx\` |\n` +
            `| PostgreSQL | \`--data "postgresql://user:pw@host/db"\` |\n` +
            `| MySQL | \`--data "mysql://user:pw@host/db"\` |\n` +
            `| SQLite | \`--data "sqlite:///path.db"\` |\n` +
            `| BigQuery | \`--source-type bigquery --db-query "..."\` |\n\n` +
            `Auto-detected via URI pattern; can also be specified explicitly with \`--source-type\`.\n` +
            `Designed with the factory pattern for easy addition of new connectors.`,
        category: "feature",
    },

    // ── MCP ──
    {
        keywords: ["mcp", "server", "tool", "resource", "claude", "protocol"],
        question: "What is the MCP Server?",
        answer: `🌐 **MCP (Model Context Protocol) Server**\n\n` +
            `A standard protocol server that enables external AI agents like Claude to use WhyLab as a tool.\n\n` +
            `**Available Tools (7):**\n` +
            `1. \`run_analysis\` — Run full pipeline\n` +
            `2. \`get_debate_verdict\` — Retrieve AI verdict\n` +
            `3. \`simulate_intervention\` — What-if simulation\n` +
            `4. \`ask_rag\` — RAG query\n` +
            `5. \`compare_scenarios\` — Scenario comparison\n` +
            `6. \`run_drift_check\` — Drift check\n` +
            `7. \`get_monitoring_status\` — Monitoring status\n\n` +
            `**Available Resources (3):**\n` +
            `- Data metadata, Analysis reports, Benchmark results`,
        category: "feature",
    },

    // ── RAG ──
    {
        keywords: ["rag", "search", "query", "qa", "natural language", "agent", "persona"],
        question: "What is the RAG Agent?",
        answer: `💬 **RAG (Retrieval-Augmented Generation) Agent**\n\n` +
            `A system for asking natural language questions about analysis results and receiving answers.\n\n` +
            `**3 Personas:**\n` +
            `- 🚀 **Growth Hacker**: Revenue opportunity-focused answers\n` +
            `- 🛡️ **Risk Manager**: Risk-focused answers\n` +
            `- ⚖️ **Product Owner**: Balanced comprehensive answers\n\n` +
            `**Usage:**\n` +
            `\`\`\`bash\n` +
            `python -m engine.cli --query "Is the coupon effective?" --persona growth_hacker\n` +
            `\`\`\`\n\n` +
            `Supports multi-turn conversations; automatically runs the pipeline if no analysis results exist.`,
        category: "feature",
    },

    // ── Usage ──
    {
        keywords: ["install", "start", "quickstart", "setup", "getting started"],
        question: "How do I get started?",
        answer: `🚀 **Quick Start:**\n\n` +
            `\`\`\`bash\n` +
            `# 1. Clone\n` +
            `git clone https://github.com/Yesol-Pilot/WhyLab.git\n` +
            `cd whylab\n\n` +
            `# 2. Python environment\n` +
            `conda create -n whylab python=3.10\n` +
            `conda activate whylab\n` +
            `pip install -e .\n\n` +
            `# 3. Run pipeline\n` +
            `python -m engine.cli --scenario A\n\n` +
            `# 4. Dashboard\n` +
            `cd dashboard; npm install; npm run dev\n` +
            `\`\`\`\n\n` +
            `Optional DB dependencies: \`pip install "whylab[sql]"\` or \`pip install "whylab[bigquery]"\``,
        category: "usage",
    },
    {
        keywords: ["cli", "command", "flag", "option"],
        question: "How do I use the CLI?",
        answer: `⌨️ **CLI v3 Key Flags:**\n\n` +
            `\`\`\`bash\n` +
            `# Synthetic data\n` +
            `python -m engine.cli --scenario A|B\n\n` +
            `# External data\n` +
            `python -m engine.cli --data "file.csv" --treatment T --outcome Y\n\n` +
            `# DB connection\n` +
            `python -m engine.cli --data "postgresql://..." --db-query "SELECT ..."\n\n` +
            `# RAG query\n` +
            `python -m engine.cli --query "Coupon effect?" --persona growth_hacker\n\n` +
            `# Monitoring\n` +
            `python -m engine.cli --monitor --interval 30 --slack-webhook $URL\n` +
            `\`\`\``,
        category: "usage",
    },
    {
        keywords: ["api", "3-line", "code", "python", "example", "analyze"],
        question: "How do I use the Python API?",
        answer: `🐍 **3-Line Python API:**\n\n` +
            `\`\`\`python\n` +
            `import whylab\n\n` +
            `results = whylab.analyze(\n` +
            `    data="sales.csv",\n` +
            `    treatment="coupon",\n` +
            `    outcome="purchase"\n` +
            `)\n\n` +
            `print(results.ate)      # Average Treatment Effect\n` +
            `print(results.verdict)  # AI Debate Verdict\n` +
            `print(results.cate)     # Individual Treatment Effects\n` +
            `\`\`\`\n\n` +
            `\`whylab.analyze()\` automatically runs the full 11-Cell pipeline and returns results as a structured object.`,
        category: "usage",
    },

    // ── Scenarios ──
    {
        keywords: ["scenario a", "credit", "limit"],
        question: "What is Scenario A?",
        answer: `💳 **Scenario A: Credit Limit Increase → Default Rate**\n\n` +
            `- **Treatment**: Whether credit limit was increased\n` +
            `- **Outcome**: Whether default occurred\n` +
            `- **Question**: "Does increasing the limit reduce defaults?"\n\n` +
            `DML estimation with 100 synthetic samples. A negative ATE indicates that limit increases reduce defaults.`,
        category: "usage",
    },
    {
        keywords: ["scenario b", "coupon", "marketing"],
        question: "What is Scenario B?",
        answer: `🎟️ **Scenario B: Coupon Distribution → Signup Conversion**\n\n` +
            `- **Treatment**: Whether a coupon was sent\n` +
            `- **Outcome**: Whether signup conversion occurred\n` +
            `- **Question**: "Does sending coupons increase signups?"\n\n` +
            `A marketing campaign effectiveness measurement scenario. A positive ATE indicates coupons increase conversion.`,
        category: "usage",
    },

    // ── Benchmarks ──
    {
        keywords: ["ihdp", "acic", "jobs", "benchmark", "academic", "performance", "pehe"],
        question: "What are the benchmark results?",
        answer: `🏆 **Academic Benchmark Validation (3 datasets × 10 iterations)**\n\n` +
            `| Dataset | Best Model | PEHE |\n` +
            `|---|---|---|\n` +
            `| **IHDP** | Oracle Ensemble | ~ 0.5 |\n` +
            `| **ACIC** | DR-Learner | Competitive |\n` +
            `| **Jobs** | X-Learner | Stable |\n\n` +
            `Achieved comparable or superior performance to CausalML, EconML, etc. on 3 standard benchmarks:\n` +
            `IHDP (747 samples), ACIC (4802 samples), Jobs (722 samples).`,
        category: "methodology",
    },

    // ── Philosophy & Vision ──
    {
        keywords: ["why", "philosophy", "vision", "goal", "purpose"],
        question: "What is WhyLab's vision?",
        answer: `🌟 **WhyLab's Vision: AI that Answers "Why"**\n\n` +
            `Traditional data analysis stops at "What happened."\n` +
            `WhyLab answers **"Why did it happen."**\n\n` +
            `**Core Principles:**\n` +
            `1. 🎯 **Causation > Correlation**: Decisions should be based on causal relationships\n` +
            `2. 🤖 **AI + Human**: AI analyzes, humans decide\n` +
            `3. 📊 **Transparency**: All results must be verifiable and explainable\n` +
            `4. 🔄 **Continuous Monitoring**: Causal effects change, so they must be continuously monitored\n\n` +
            `> *"Innovate decisions, not just data"* — WhyLab`,
        category: "philosophy",
    },
    {
        keywords: ["living", "ledger"],
        question: "What is Living Ledger?",
        answer: `📖 **Living Ledger** is WhyLab's research vision document.\n\n` +
            `As the name "Living Ledger" suggests, it envisions a system where data autonomously records and updates causal relationships.\n\n` +
            `Currently exists as a paper-level vision document, with WhyLab mapping documented in the architecture docs (\`docs/architecture.md\`).`,
        category: "philosophy",
    },

    // ── Tech Stack ──
    {
        keywords: ["tech", "stack", "library", "dependency"],
        question: "What is the tech stack?",
        answer: `🔧 **Tech Stack:**\n\n` +
            `**Engine (Python):**\n` +
            `- EconML, CausalML — Causal inference core\n` +
            `- LightGBM (GPU support) — Meta-learner backbone\n` +
            `- DuckDB — Zero-copy data preprocessing\n` +
            `- SQLAlchemy — DB connectors\n` +
            `- ChromaDB — RAG vector store\n\n` +
            `**Dashboard (TypeScript):**\n` +
            `- Next.js 16 + React 19\n` +
            `- Tailwind CSS v4\n` +
            `- Recharts — Data visualization\n` +
            `- ReactFlow — DAG visualization\n` +
            `- Framer Motion — Animations\n\n` +
            `Compatible with Python 3.9~3.13, MIT License.`,
        category: "architecture",
    },

    // ── Dashboard ──
    {
        keywords: ["dashboard", "ui", "frontend", "visualization"],
        question: "What can I see on the dashboard?",
        answer: `🖥️ **Dashboard Components:**\n\n` +
            `1. **ROI Simulator** — Adjust policy intensity → Real-time profit/default rate prediction\n` +
            `2. **CATE Explorer** — Individual effect distribution by segment\n` +
            `3. **Causal Graph (DAG)** — Causal structure visualization\n` +
            `4. **Stats Cards** — Key metrics: ATE, sample size, model type, etc.\n` +
            `5. **Conformal Band** — Individual confidence interval chart\n` +
            `6. **AI Debate Verdict** — Growth vs Risk debate results\n` +
            `7. **Sensitivity Report** — E-value, Overlap, GATES\n` +
            `8. **Model Comparison** — Meta-learner performance comparison\n` +
            `9. **Chat Panel** — That's me! 🤖 Ask me about analysis results\n\n` +
            `URL: [whylab.vercel.app](https://whylab.vercel.app/dashboard)`,
        category: "feature",
    },

    // ── Robustness ──
    {
        keywords: ["refutation", "validation", "placebo", "bootstrap", "random cause"],
        question: "How is robustness validated?",
        answer: `🔬 **3 Refutation Tests:**\n\n` +
            `1. **Placebo Test**: Replace with a fake treatment variable → Check if effect is near 0\n` +
            `   - Pass: Original result is not due to chance\n\n` +
            `2. **Bootstrap CI**: Calculate confidence intervals via bootstrap resampling\n` +
            `   - Stable if original ATE falls within bootstrap CI\n\n` +
            `3. **Random Common Cause**: Add a random confounder variable\n` +
            `   - Robust if ATE doesn't change significantly\n\n` +
            `All tests run automatically in RefutationCell, with Pass/Fail displayed in Status Cards.`,
        category: "methodology",
    },
    {
        keywords: ["evalue", "e-value", "unobserved", "confounder"],
        question: "What is E-value?",
        answer: `🔍 **E-value (Evidence Value)**\n\n` +
            `A robustness indicator against unobserved confounders.\n\n` +
            `**Interpretation:**\n` +
            `- E-value = 3.0 → "To explain the observed effect, an unobserved confounder would need to have 3x or greater impact on both treatment and outcome"\n` +
            `- Higher is more robust (≥2.0: good, ≥3.0: strong)\n\n` +
            `WhyLab also reports the CI-bound E-value for a conservative evaluation.`,
        category: "methodology",
    },

    // ── ROI Simulator ──
    {
        keywords: ["roi", "simulator", "simulation", "what-if"],
        question: "What is the ROI Simulator?",
        answer: `💰 **ROI (Policy) Simulator**\n\n` +
            `Simulates in real-time: "What if we change the policy intensity?"\n\n` +
            `**Adjustable Parameters:**\n` +
            `- Credit limit increase amount ($0 ~ $2,000)\n` +
            `- Target user range (top 5% ~ all users)\n\n` +
            `**Output:**\n` +
            `- Expected Net Profit\n` +
            `- Expected Default Rate\n` +
            `- Profit Sensitivity Curve\n` +
            `- AI Agent opinions (Growth Hacker vs Risk Manager)\n\n` +
            `Move the sliders to see changes in real-time.`,
        category: "feature",
    },
];

/* ──────────────────────────────────────
 * 3. Knowledge Matching Engine
 * ────────────────────────────────────── */
export function searchKnowledge(query: string): KnowledgeEntry | null {
    const q = query.toLowerCase();

    // 1-Pass: Calculate matching score
    let bestMatch: KnowledgeEntry | null = null;
    let bestScore = 0;

    for (const entry of KNOWLEDGE_BASE) {
        let score = 0;
        for (const kw of entry.keywords) {
            if (q.includes(kw)) {
                score += kw.length; // Longer keyword matches score higher
            }
        }
        if (score > bestScore) {
            bestScore = score;
            bestMatch = entry;
        }
    }

    // Return only if at least 2 characters of keyword matched
    return bestScore >= 2 ? bestMatch : null;
}

/* ──────────────────────────────────────
 * 4. Suggested Questions
 * ────────────────────────────────────── */
export const PROJECT_SUGGESTIONS = [
    "What is WhyLab?",
    "How is it different from other tools?",
    "What are Meta-Learners?",
    "What is the Multi-Agent Debate System?",
    "How do I get started?",
    "How does monitoring work?",
];
