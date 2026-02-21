import { CausalAnalysisResult } from "@/types";

/**
 * 시나리오 A: 신용 한도 (Continuous Treatment)
 * 가설: 신용 한도를 늘리면 연체율이 낮아질까? (역인과 관계 주의)
 */
export const scenarioA_Mock: CausalAnalysisResult = {
    ate: {
        value: -0.05,
        ci_lower: -0.07,
        ci_upper: -0.03,
        alpha: 0.05,
        description: "A $1M increase in credit limit reduces default rate by 5%p (avg)"
    },
    cate_distribution: {
        mean: -0.05,
        std: 0.02,
        min: -0.15,
        max: 0.01,
        histogram: {
            bin_edges: [-0.15, -0.12, -0.09, -0.06, -0.03, 0.00, 0.03],
            counts: [50, 150, 400, 250, 100, 50]
        }
    },
    segments: [
        { name: "20s", dimension: "age", n: 300, cate_mean: -0.02, cate_ci_lower: -0.04, cate_ci_upper: 0.00 },
        { name: "30s", dimension: "age", n: 400, cate_mean: -0.06, cate_ci_lower: -0.08, cate_ci_upper: -0.04 },
        { name: "40s+", dimension: "age", n: 300, cate_mean: -0.07, cate_ci_lower: -0.09, cate_ci_upper: -0.05 },
        { name: "Low Income", dimension: "income", n: 350, cate_mean: -0.08, cate_ci_lower: -0.10, cate_ci_upper: -0.06 },
        { name: "High Income", dimension: "income", n: 350, cate_mean: -0.03, cate_ci_lower: -0.05, cate_ci_upper: -0.01 }
    ],
    dag: {
        nodes: [
            { id: "income", label: "Income", role: "confounder" },
            { id: "age", label: "Age", role: "confounder" },
            { id: "credit_score", label: "Credit Score", role: "confounder" },
            { id: "limit", label: "Credit Limit (T)", role: "treatment" },
            { id: "default", label: "Default (Y)", role: "outcome" }
        ],
        edges: [
            { source: "income", target: "limit" },
            { source: "income", target: "default" },
            { source: "income", target: "credit_score" },
            { source: "age", target: "limit" },
            { source: "age", target: "default" },
            { source: "credit_score", target: "limit" },
            { source: "credit_score", target: "default" },
            { source: "limit", target: "default" }
        ]
    },
    metadata: {
        generated_at: "2026-02-12T12:00:00",
        scenario: "A",
        model_type: "AutoML (LinearDML Win)",
        n_samples: 1000,
        feature_names: ["income", "age", "credit_score"],
        treatment_col: "credit_limit",
        outcome_col: "is_default"
    },
    sensitivity: {
        status: "Pass",
        placebo_test: { status: "Pass", p_value: 0.85, mean_effect: 0.001 },
        random_common_cause: { status: "Pass", stability: 0.98, mean_effect: -0.051 }
    }
};

/**
 * 시나리오 B: 투자 쿠폰 (Binary Treatment)
 * 가설: 쿠폰을 주면 투자 서비스에 가입할까? (이질적 효과 분석)
 */
export const scenarioB_Mock: CausalAnalysisResult = {
    ate: {
        value: 0.12,
        ci_lower: 0.10,
        ci_upper: 0.14,
        alpha: 0.05,
        description: "Coupon issuance increases subscription probability by 12%p"
    },
    cate_distribution: {
        mean: 0.12,
        std: 0.08,
        min: -0.05,
        max: 0.30,
        histogram: {
            bin_edges: [-0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            counts: [20, 50, 100, 300, 300, 150, 80]
        }
    },
    segments: [
        { name: "20s", dimension: "age", n: 400, cate_mean: 0.18, cate_ci_lower: 0.15, cate_ci_upper: 0.21 },
        { name: "40s", dimension: "age", n: 300, cate_mean: 0.05, cate_ci_lower: 0.02, cate_ci_upper: 0.08 },
        { name: "New Users", dimension: "tenure", n: 200, cate_mean: 0.25, cate_ci_lower: 0.20, cate_ci_upper: 0.30 },
        { name: "Long-term Users", dimension: "tenure", n: 500, cate_mean: 0.08, cate_ci_lower: 0.06, cate_ci_upper: 0.10 }
    ],
    dag: {
        nodes: [
            { id: "age", label: "Age", role: "confounder" },
            { id: "income", label: "Income", role: "confounder" },
            { id: "coupon", label: "Coupon (T)", role: "treatment" },
            { id: "join", label: "Subscription (Y)", role: "outcome" },
            { id: "app_usage", label: "App Usage", role: "confounder" }
        ],
        edges: [
            { source: "age", target: "coupon" },
            { source: "age", target: "join" },
            { source: "income", target: "join" }, // income -> join (coupon targeting X)
            { source: "app_usage", target: "coupon" },
            { source: "app_usage", target: "join" },
            { source: "coupon", target: "join" }
        ]
    },
    metadata: {
        generated_at: "2026-02-12T12:00:00",
        scenario: "B",
        model_type: "AutoML (CausalForest Win)",
        n_samples: 1000,
        feature_names: ["age", "income", "app_usage"],
        treatment_col: "coupon_sent",
        outcome_col: "is_joined"
    },
    sensitivity: {
        status: "Pass",
        placebo_test: { status: "Pass", p_value: 0.72, mean_effect: 0.005 },
        random_common_cause: { status: "Pass", stability: 0.95, mean_effect: 0.115 }
    }
};

/**
 * 시나리오 C: 개인 커리어 (Synthetic Simulation)
 * 가설: 새로운 기술(Skill)을 습득하면 연봉(Market Value)이 오를까?
 */
export const scenarioC_Mock: CausalAnalysisResult = {
    ate: {
        value: 15.0,
        ci_lower: 12.5,
        ci_upper: 17.5,
        alpha: 0.05,
        description: "Acquiring a core skill (e.g. GenAI) raises market value by 15% (simulated)"
    },
    cate_distribution: {
        mean: 15.0,
        std: 5.0,
        min: 5.0,
        max: 25.0,
        histogram: {
            bin_edges: [0, 5, 10, 15, 20, 25, 30],
            counts: [20, 100, 300, 400, 150, 30]
        }
    },
    segments: [
        { name: "Junior", dimension: "level", n: 400, cate_mean: 20.0, cate_ci_lower: 18.0, cate_ci_upper: 22.0 },
        { name: "Senior", dimension: "level", n: 300, cate_mean: 10.0, cate_ci_lower: 8.0, cate_ci_upper: 12.0 },
        { name: "R&D", dimension: "role", n: 200, cate_mean: 18.0, cate_ci_lower: 15.0, cate_ci_upper: 21.0 },
        { name: "Management", dimension: "role", n: 100, cate_mean: 5.0, cate_ci_lower: 2.0, cate_ci_upper: 8.0 }
    ],
    dag: {
        nodes: [
            { id: "education", label: "Education", role: "confounder" },
            { id: "experience", label: "Experience (yrs)", role: "confounder" },
            { id: "new_skill", label: "New Skill (T)", role: "treatment" },
            { id: "salary_increase", label: "Salary Increase (Y)", role: "outcome" },
            { id: "network", label: "Networking", role: "confounder" }
        ],
        edges: [
            { source: "education", target: "salary_increase" },
            { source: "experience", target: "salary_increase" },
            { source: "experience", target: "new_skill" },
            { source: "network", target: "new_skill" },
            { source: "new_skill", target: "salary_increase" }
        ]
    },
    metadata: {
        generated_at: "2026-02-15T15:00:00",
        scenario: "C",
        model_type: "AutoML (Synthetic Learner)",
        n_samples: 50000,
        feature_names: ["education", "experience", "network"],
        treatment_col: "acquired_new_skill",
        outcome_col: "salary_ncrease_pct"
    },
    sensitivity: {
        status: "Pass",
        placebo_test: { status: "Pass", p_value: 0.92, mean_effect: 0.001 },
        random_common_cause: { status: "Pass", stability: 0.99, mean_effect: 14.8 }
    }
};
