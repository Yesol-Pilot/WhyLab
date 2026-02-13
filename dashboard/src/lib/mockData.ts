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
        description: "신용 한도 100만 원 증가 시 연체율 5%p 감소 (평균)"
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
        { name: "20대", dimension: "age", n: 300, cate_mean: -0.02, cate_ci_lower: -0.04, cate_ci_upper: 0.00 },
        { name: "30대", dimension: "age", n: 400, cate_mean: -0.06, cate_ci_lower: -0.08, cate_ci_upper: -0.04 },
        { name: "40대 이상", dimension: "age", n: 300, cate_mean: -0.07, cate_ci_lower: -0.09, cate_ci_upper: -0.05 },
        { name: "저소득", dimension: "income", n: 350, cate_mean: -0.08, cate_ci_lower: -0.10, cate_ci_upper: -0.06 },
        { name: "고소득", dimension: "income", n: 350, cate_mean: -0.03, cate_ci_lower: -0.05, cate_ci_upper: -0.01 }
    ],
    dag: {
        nodes: [
            { id: "income", label: "소득 (Income)", role: "confounder" },
            { id: "age", label: "나이 (Age)", role: "confounder" },
            { id: "credit_score", label: "신용점수 (Score)", role: "confounder" },
            { id: "limit", label: "신용 한도 (T)", role: "treatment" },
            { id: "default", label: "연체 여부 (Y)", role: "outcome" }
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
        description: "쿠폰 지급 시 가입 확률 12%p 증가"
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
        { name: "20대", dimension: "age", n: 400, cate_mean: 0.18, cate_ci_lower: 0.15, cate_ci_upper: 0.21 },
        { name: "40대", dimension: "age", n: 300, cate_mean: 0.05, cate_ci_lower: 0.02, cate_ci_upper: 0.08 },
        { name: "최근가입자", dimension: "tenure", n: 200, cate_mean: 0.25, cate_ci_lower: 0.20, cate_ci_upper: 0.30 },
        { name: "장기사용자", dimension: "tenure", n: 500, cate_mean: 0.08, cate_ci_lower: 0.06, cate_ci_upper: 0.10 }
    ],
    dag: {
        nodes: [
            { id: "age", label: "나이", role: "confounder" },
            { id: "income", label: "소득", role: "confounder" },
            { id: "coupon", label: "쿠폰 (T)", role: "treatment" },
            { id: "join", label: "가입 (Y)", role: "outcome" },
            { id: "app_usage", label: "앱 사용량", role: "confounder" }
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
