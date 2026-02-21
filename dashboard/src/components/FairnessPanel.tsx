"use client";

import React from "react";
import { clsx } from "clsx";
import { CheckCircle2, AlertTriangle, HelpCircle } from "lucide-react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";

interface SubgroupMetric {
    group: string;
    n: number;
    mean_cate: number;
    positive_ratio: number;
}

interface FairnessResultData {
    attribute: string;
    causal_parity_gap: number;
    disparate_impact_ratio: number;
    equalized_cate_score: number;
    counterfactual_fairness_idx: number;
    is_fair: boolean;
    violations: string[];
    subgroups: SubgroupMetric[];
}

interface FairnessPanelProps {
    data: FairnessResultData;
}

const MetricCard = ({ title, value, threshold, passed, description }: any) => (
    <div className={clsx(
        "bg-slate-800/50 rounded-xl p-4 border transition-all",
        passed ? "border-emerald-500/20 shadow-[0_0_10px_rgba(16,185,129,0.05)]" : "border-red-500/30 shadow-[0_0_10px_rgba(239,68,68,0.1)]"
    )}>
        <div className="flex items-start justify-between mb-2">
            <div>
                <h4 className="text-sm font-medium text-slate-300">{title}</h4>
                <p className="text-xs text-slate-500">{description}</p>
            </div>
            {passed ? <CheckCircle2 className="w-5 h-5 text-emerald-500" /> : <AlertTriangle className="w-5 h-5 text-red-500" />}
        </div>
        <div className="flex items-baseline gap-2">
            <span className={clsx("text-2xl font-bold", passed ? "text-white" : "text-red-400")}>
                {value.toFixed(3)}
            </span>
            <span className="text-xs text-slate-500">
                (Threshold: {threshold})
            </span>
        </div>
    </div>
);

export default function FairnessPanel({ data }: FairnessPanelProps) {
    if (!data) return <div className="text-center text-slate-500 py-10">No data available.</div>;

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-lg font-bold text-white flex items-center gap-2">
                        Sensitive Attribute: <span className="text-brand-400 font-mono">{data.attribute}</span>
                    </h2>
                    <p className="text-sm text-slate-400">
                        {data.is_fair ? "All fairness criteria are met." : `${data.violations.length} violation(s) detected.`}
                    </p>
                </div>
                <div className={clsx(
                    "px-3 py-1 rounded-full text-xs font-bold border",
                    data.is_fair ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" : "bg-red-500/10 text-red-400 border-red-500/20"
                )}>
                    {data.is_fair ? "PASS" : "FAIL"}
                </div>
            </div>

            {/* 4대 지표 카드 */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <MetricCard
                    title="Causal Parity Gap"
                    value={data.causal_parity_gap}
                    threshold="< 0.1"
                    passed={data.causal_parity_gap < 0.1}
                    description="Avg treatment effect gap between groups"
                />
                <MetricCard
                    title="Disparate Impact"
                    value={data.disparate_impact_ratio}
                    threshold="≥ 0.8"
                    passed={data.disparate_impact_ratio >= 0.8}
                    description="Balance of positive effect ratios"
                />
                <MetricCard
                    title="Equalized CATE"
                    value={data.equalized_cate_score}
                    threshold="≥ 0.8"
                    passed={data.equalized_cate_score >= 0.8}
                    description="Similarity of treatment effect distributions"
                />
                <MetricCard
                    title="Counterfactual"
                    value={data.counterfactual_fairness_idx}
                    threshold="≥ 0.7"
                    passed={data.counterfactual_fairness_idx >= 0.7}
                    description="Counterfactual fairness index"
                />
            </div>

            {/* 서브그룹 차트 */}
            <div className="bg-slate-900/50 rounded-xl p-6 border border-white/5">
                <h3 className="text-base font-medium text-white mb-6">Mean CATE by Subgroup</h3>
                <div className="h-[250px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data.subgroups} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                            <XAxis dataKey="group" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                            <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc' }}
                                itemStyle={{ color: '#f8fafc' }}
                                cursor={{ fill: '#334155', opacity: 0.2 }}
                            />
                            <Bar dataKey="mean_cate" name="Mean CATE" fill="#8b5cf6" radius={[4, 4, 0, 0]} barSize={40} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* 위반 사항 목록 (위반 시만 표시) */}
            {data.violations.length > 0 && (
                <div className="bg-red-500/5 rounded-xl p-4 border border-red-500/10">
                    <h3 className="text-sm font-bold text-red-400 mb-2 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4" />
                        Violations
                    </h3>
                    <ul className="space-y-1">
                        {data.violations.map((v, i) => (
                            <li key={i} className="text-xs text-red-300 ml-5 list-disc">
                                {v}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}
