"use client";

import { motion } from "framer-motion";
import { Bot, TrendingDown, TrendingUp, Sparkles, Target, Lightbulb } from "lucide-react";
import { CausalAnalysisResult } from "@/types";

interface Props {
    data: CausalAnalysisResult;
}

export default function AIInsightPanel({ data }: Props) {
    const insights = data.ai_insights;
    if (!insights) return null;

    const isSignificant = insights.significance === "유의함";
    const qualityColor =
        insights.model_quality === "excellent" ? "text-green-400" :
            insights.model_quality === "good" ? "text-blue-400" :
                insights.model_quality === "moderate" ? "text-yellow-400" : "text-red-400";

    const qualityBg =
        insights.model_quality === "excellent" ? "bg-green-500/10 border-green-500/20" :
            insights.model_quality === "good" ? "bg-blue-500/10 border-blue-500/20" :
                insights.model_quality === "moderate" ? "bg-yellow-500/10 border-yellow-500/20" :
                    "bg-red-500/10 border-red-500/20";

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="glass-card space-y-5"
        >
            {/* 헤더 */}
            <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-brand-500/10 border border-brand-500/20">
                    <Bot className="w-5 h-5 text-brand-400" />
                </div>
                <div>
                    <h2 className="text-lg font-bold text-white flex items-center gap-2">
                        AI Interpretation
                        <span className="text-xs px-2 py-0.5 rounded-full bg-brand-500/20 text-brand-300 font-normal">
                            {insights.generated_by === "llm" ? "Gemini" : "Rule-Based"}
                        </span>
                    </h2>
                    <p className="text-xs text-slate-500">자동 생성된 인과추론 결과 해석</p>
                </div>
            </div>

            {/* 헤드라인 */}
            <div className={`p-4 rounded-lg border ${isSignificant ? "bg-green-500/5 border-green-500/20" : "bg-yellow-500/5 border-yellow-500/20"}`}>
                <p className="text-sm font-semibold text-white flex items-center gap-2">
                    {insights.effect_direction === "감소"
                        ? <TrendingDown className="w-4 h-4 text-green-400" />
                        : <TrendingUp className="w-4 h-4 text-red-400" />
                    }
                    {insights.headline}
                </p>
            </div>

            {/* 요약 */}
            <div className="pl-4 border-l-2 border-brand-500/30">
                <p className="text-sm text-slate-300 leading-relaxed">
                    {insights.summary}
                </p>
            </div>

            {/* 메트릭 그리드 */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <MetricChip
                    label="통계 유의성"
                    value={insights.significance}
                    color={isSignificant ? "text-green-400" : "text-yellow-400"}
                />
                <MetricChip
                    label="효과 크기"
                    value={insights.effect_size === "large" ? "대(Large)" : insights.effect_size === "medium" ? "중(Medium)" : "소(Small)"}
                    color="text-slate-300"
                />
                <MetricChip
                    label="모델 품질"
                    value={`${insights.model_quality_label} (${insights.correlation})`}
                    color={qualityColor}
                />
                <MetricChip
                    label="RMSE"
                    value={insights.rmse.toFixed(4)}
                    color="text-slate-300"
                />
            </div>

            {/* 주요 드라이버 */}
            {insights.top_drivers.length > 0 && (
                <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1">
                        <Target className="w-3 h-3" /> Top Drivers (SHAP)
                    </p>
                    <div className="flex flex-wrap gap-2">
                        {insights.top_drivers.map((d, i) => (
                            <div
                                key={d.feature}
                                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-sm"
                            >
                                <span className="text-brand-300 font-mono text-xs">#{i + 1}</span>
                                <span className="text-white">{d.feature}</span>
                                <span className="text-slate-500 text-xs">{(d.importance * 100).toFixed(1)}%</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* 권고사항 */}
            <div className="p-4 rounded-lg bg-brand-500/5 border border-brand-500/10">
                <p className="text-xs text-brand-300 uppercase tracking-wider mb-1 flex items-center gap-1">
                    <Lightbulb className="w-3 h-3" /> Recommendation
                </p>
                <p className="text-sm text-slate-300">{insights.recommendation}</p>
            </div>
        </motion.div>
    );
}

function MetricChip({ label, value, color }: { label: string; value: string; color: string }) {
    return (
        <div className="p-3 rounded-lg bg-white/5 border border-white/5">
            <p className="text-xs text-slate-500 mb-1">{label}</p>
            <p className={`text-sm font-semibold ${color}`}>{value}</p>
        </div>
    );
}
