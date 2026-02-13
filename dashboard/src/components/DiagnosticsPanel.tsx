"use client";

import { motion } from "framer-motion";
import { Shield, ShieldCheck, ShieldAlert, BarChart3, Layers, AlertTriangle, Info } from "lucide-react";
import { CausalAnalysisResult } from "@/types";

interface Props {
    data: CausalAnalysisResult;
}

function StatusBadge({ status }: { status: string }) {
    const color =
        status === "Pass" ? "bg-green-500/20 text-green-400 border-green-500/20" :
            status === "Fail" ? "bg-red-500/20 text-red-400 border-red-500/20" :
                status === "Info" ? "bg-blue-500/20 text-blue-400 border-blue-500/20" :
                    "bg-slate-500/20 text-slate-400 border-slate-500/20";
    return (
        <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${color}`}>
            {status}
        </span>
    );
}

export default function DiagnosticsPanel({ data }: Props) {
    const s = data.sensitivity;
    if (!s) return null;

    const overallIcon = s.status === "Pass"
        ? <ShieldCheck className="w-5 h-5 text-green-400" />
        : <ShieldAlert className="w-5 h-5 text-yellow-400" />;

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="glass-card space-y-5"
        >
            {/* 헤더 */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-purple-500/10 border border-purple-500/20">
                        <Shield className="w-5 h-5 text-purple-400" />
                    </div>
                    <div>
                        <h2 className="text-lg font-bold text-white">Statistical Diagnostics</h2>
                        <p className="text-xs text-slate-500">견고성(Robustness) 심화 검증</p>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    {overallIcon}
                    <StatusBadge status={s.status} />
                </div>
            </div>

            {/* 4칸 그리드 — 기본 테스트 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {/* Placebo */}
                <DiagCard
                    title="Placebo Treatment"
                    desc="처치 변수를 무작위로 섞어 가짜 효과가 0인지 확인"
                    status={s.placebo_test.status}
                    metrics={[
                        { label: "Mean Effect", value: s.placebo_test.mean_effect.toFixed(4) },
                        { label: "P-value", value: s.placebo_test.p_value.toFixed(3) },
                    ]}
                />
                {/* Random Common Cause */}
                <DiagCard
                    title="Random Common Cause"
                    desc="무작위 교란 변수를 추가해도 ATE가 안정적인지 확인"
                    status={s.random_common_cause.status}
                    metrics={[
                        { label: "Stability", value: (s.random_common_cause.stability * 100).toFixed(1) + "%" },
                        { label: "Mean Effect", value: s.random_common_cause.mean_effect.toFixed(4) },
                    ]}
                />
            </div>

            {/* 고급 진단 — E-value + Overlap */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {/* E-value */}
                {s.e_value && s.e_value.status !== "Not Run" && (
                    <DiagCard
                        title="E-value (교란 견고성)"
                        desc={s.e_value.interpretation}
                        status={s.e_value.status}
                        metrics={[
                            { label: "E-value (Point)", value: s.e_value.point.toFixed(2) },
                            { label: "E-value (CI)", value: s.e_value.ci_bound.toFixed(2) },
                        ]}
                        icon={<AlertTriangle className="w-4 h-4" />}
                    />
                )}

                {/* Overlap */}
                {s.overlap && s.overlap.status !== "Not Run" && (
                    <DiagCard
                        title="Overlap (Positivity)"
                        desc={s.overlap.interpretation}
                        status={s.overlap.status}
                        metrics={[
                            { label: "Overlap Score", value: String(s.overlap.overlap_score) },
                            ...(s.overlap.ps_stats ? [
                                { label: "PS Treated μ", value: String(s.overlap.ps_stats.treated_mean) },
                                { label: "PS Control μ", value: String(s.overlap.ps_stats.control_mean) },
                            ] : []),
                            ...(s.overlap.pct_extreme_weights !== undefined
                                ? [{ label: "Extreme Wt%", value: s.overlap.pct_extreme_weights + "%" }]
                                : []),
                        ]}
                        icon={<Layers className="w-4 h-4" />}
                    />
                )}
            </div>

            {/* GATES/CLAN */}
            {s.gates && s.gates.status !== "Not Run" && s.gates.groups.length > 0 && (
                <div className="space-y-3">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <BarChart3 className="w-4 h-4 text-blue-400" />
                            <span className="text-sm font-semibold text-white">GATES / CLAN Analysis</span>
                            <StatusBadge status={s.gates.status} />
                        </div>
                        <div className="flex items-center gap-2 text-xs">
                            <span className="text-slate-500">F-stat</span>
                            <span className="text-white font-mono">{s.gates.f_statistic}</span>
                        </div>
                    </div>

                    <p className="text-xs text-slate-400">{s.gates.heterogeneity}</p>

                    {/* GATES 바 차트 */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                        {s.gates.groups.map((g) => {
                            const maxAbs = Math.max(
                                ...s.gates!.groups.map((x) => Math.abs(x.mean_cate)),
                                0.001
                            );
                            const barWidth = Math.min(100, (Math.abs(g.mean_cate) / maxAbs) * 100);
                            const isNeg = g.mean_cate < 0;

                            return (
                                <div
                                    key={g.group_id}
                                    className="p-3 rounded-lg bg-white/5 border border-white/5 space-y-2"
                                >
                                    <div className="flex items-center justify-between">
                                        <span className="text-xs font-bold text-white">{g.label}</span>
                                        <span className="text-xs text-slate-500">n={g.n}</span>
                                    </div>
                                    {/* 바 */}
                                    <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all duration-500 ${isNeg ? "bg-green-500" : "bg-red-500"}`}
                                            style={{ width: `${barWidth}%` }}
                                        />
                                    </div>
                                    <div className="text-center">
                                        <span className={`text-sm font-mono font-bold ${isNeg ? "text-green-400" : "text-red-400"}`}>
                                            {g.mean_cate > 0 ? "+" : ""}{g.mean_cate.toFixed(4)}
                                        </span>
                                        <p className="text-[10px] text-slate-500">
                                            [{g.ci_lower.toFixed(4)}, {g.ci_upper.toFixed(4)}]
                                        </p>
                                    </div>
                                    {/* CLAN Features */}
                                    {Object.keys(g.clan_features).length > 0 && (
                                        <div className="text-[10px] text-slate-500 space-y-0.5 pt-1 border-t border-white/5">
                                            {Object.entries(g.clan_features).slice(0, 3).map(([f, v]) => (
                                                <div key={f} className="flex justify-between">
                                                    <span>{f}</span>
                                                    <span className="text-slate-400">{v}</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </motion.div>
    );
}

function DiagCard({
    title, desc, status, metrics, icon,
}: {
    title: string;
    desc: string;
    status: string;
    metrics: { label: string; value: string }[];
    icon?: React.ReactNode;
}) {
    return (
        <div className="p-4 rounded-lg bg-white/5 border border-white/5 space-y-3">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    {icon || <Info className="w-4 h-4 text-slate-400" />}
                    <span className="text-sm font-semibold text-white">{title}</span>
                </div>
                <StatusBadge status={status} />
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">{desc}</p>
            <div className="flex flex-wrap gap-3">
                {metrics.map((m) => (
                    <div key={m.label}>
                        <p className="text-[10px] text-slate-500">{m.label}</p>
                        <p className="text-sm font-mono text-white">{m.value}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}
