"use client";

import { motion } from "framer-motion";
import { TrendingUp, TrendingDown, Activity, AlertCircle } from "lucide-react";
import { CausalAnalysisResult } from "@/types";

export default function StatsCards({ data }: { data: CausalAnalysisResult }) {
    const { ate, metadata } = data;
    const isPositive = ate.value > 0;

    return (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            {/* ATE Card */}
            <motion.div
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
                className="glass-card flex items-center gap-4 relative overflow-hidden"
            >
                <div className="absolute -right-4 -top-4 w-24 h-24 bg-brand-500/10 rounded-full blur-2xl" />
                <div className="p-3 rounded-lg bg-brand-500/20 text-brand-400">
                    <Activity className="w-6 h-6" />
                </div>
                <div>
                    <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Average Treatment Effect</p>
                    <div className="flex items-baseline gap-2">
                        <h3 className="text-2xl font-bold text-white">{ate.value.toFixed(4)}</h3>
                        <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${isPositive ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                            {isPositive ? '+' : ''}{(ate.value * 100).toFixed(2)}%
                        </span>
                    </div>
                    <p className="text-slate-500 text-[10px] mt-1">95% CI: [{ate.ci_lower.toFixed(3)}, {ate.ci_upper.toFixed(3)}]</p>
                </div>
            </motion.div>

            {/* Sample Size Card */}
            <motion.div
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
                className="glass-card flex items-center gap-4"
            >
                <div className="p-3 rounded-lg bg-accent-cyan/20 text-accent-cyan">
                    <TrendingUp className="w-6 h-6" />
                </div>
                <div>
                    <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Sample Size</p>
                    <h3 className="text-2xl font-bold text-white">{metadata.n_samples.toLocaleString()}</h3>
                    <p className="text-slate-500 text-[10px] mt-1">Analyze {metadata.n_samples} users</p>
                </div>
            </motion.div>

            {/* Treatment Variable */}
            <motion.div
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
                className="glass-card flex items-center gap-4"
            >
                <div className="p-3 rounded-lg bg-accent-pink/20 text-accent-pink">
                    <AlertCircle className="w-6 h-6" />
                </div>
                <div>
                    <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Treatment</p>
                    <h3 className="text-lg font-bold text-white truncate max-w-[120px]" title={metadata.treatment_col}>
                        {metadata.treatment_col}
                    </h3>
                    <p className="text-slate-500 text-[10px] mt-1">Target Variable</p>
                </div>
            </motion.div>

            {/* Outcome Variable */}
            <motion.div
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}
                className="glass-card flex items-center gap-4"
            >
                <div className="p-3 rounded-lg bg-emerald-500/20 text-emerald-400">
                    <TrendingDown className="w-6 h-6" />
                </div>
                <div>
                    <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Outcome</p>
                    <h3 className="text-lg font-bold text-white truncate max-w-[120px]" title={metadata.outcome_col}>
                        {metadata.outcome_col}
                    </h3>
                    <p className="text-slate-500 text-[10px] mt-1">Target Metric</p>
                </div>
            </motion.div>

            {/* Row 2: Advanced Metrics */}

            {/* Model Info (AutoML Winner) */}
            <motion.div
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}
                className="glass-card flex items-center gap-4 border-l-4 border-brand-500"
            >
                <div className="p-3 rounded-lg bg-brand-500/10 text-brand-300">
                    <Activity className="w-6 h-6" />
                </div>
                <div>
                    <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">AutoML Winner</p>
                    <h3 className="text-lg font-bold text-white">{metadata.model_type}</h3>
                    <p className="text-slate-500 text-[10px] mt-1">RMSE Optimized</p>
                </div>
            </motion.div>

            {/* Sensitivity Analysis (Placebo) */}
            <motion.div
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}
                className="glass-card flex items-center gap-4 border-l-4 border-blue-500"
            >
                <div>
                    <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Robustness Check</p>
                    <div className="flex items-center gap-2">
                        <span className={`px-2 py-0.5 rounded text-xs font-bold ${data.sensitivity?.placebo_test?.status === 'Pass' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                            {data.sensitivity?.placebo_test?.status || 'N/A'}
                        </span>
                        <span className="text-xs text-slate-500">Placebo Test</span>
                    </div>
                    <p className="text-slate-500 text-[10px] mt-1">
                        p-val: {data.sensitivity?.placebo_test?.p_value?.toFixed(3) ?? 'N/A'}
                    </p>
                </div>
            </motion.div>
        </div>
    );
}
