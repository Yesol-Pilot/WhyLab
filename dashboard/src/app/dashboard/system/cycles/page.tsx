"use client";

import { useEffect, useState, useCallback } from "react";
import { ScrollText, RefreshCw, CheckCircle, AlertTriangle, XCircle, Clock, FlaskConical, Brain, Scale, ChevronDown, ChevronUp } from "lucide-react";

interface CycleLog {
    message: string;
    level: string;
    agent_id: string | null;
    timestamp: string | null;
}

interface Cycle {
    id: number;
    started_at: string | null;
    ended_at: string | null;
    status: string;
    hypotheses: number;
    experiments: number;
    reviews: number;
    verdict: string | null;
    log_count: number;
    logs?: CycleLog[];
}

interface Stats {
    total_cycles: number;
    completed_cycles: number;
    total_hypotheses: number;
    total_experiments: number;
    total_reviews: number;
    verdicts: { ACCEPT: number; REVISE: number; REJECT: number };
}

const VERDICT_STYLE: Record<string, { icon: typeof CheckCircle; color: string; bg: string; label: string }> = {
    ACCEPT: { icon: CheckCircle, color: "text-emerald-600", bg: "bg-emerald-50 border-emerald-200", label: "Accepted" },
    REVISE: { icon: AlertTriangle, color: "text-amber-600", bg: "bg-amber-50 border-amber-200", label: "Revision Requested" },
    REJECT: { icon: XCircle, color: "text-red-600", bg: "bg-red-50 border-red-200", label: "Rejected" },
};

function formatTime(iso: string | null): string {
    if (!iso) return "—";
    const d = new Date(iso);
    return d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export default function CyclesPage() {
    const [cycles, setCycles] = useState<Cycle[]>([]);
    const [stats, setStats] = useState<Stats | null>(null);
    const [expanded, setExpanded] = useState<number | null>(null);

    const fetchCycles = useCallback(async () => {
        try {
            const res = await fetch("http://localhost:4001/system/cycles");
            const data = await res.json();
            setCycles(data.cycles || []);
            setStats(data.stats || null);
        } catch (error) {
            console.error("Cycle fetch error:", error);
        }
    }, []);

    useEffect(() => { fetchCycles(); }, [fetchCycles]);

    return (
        <div className="p-8 space-y-6 bg-slate-50 min-h-screen">
            {/* 헤더 */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight text-slate-900 flex items-center gap-3">
                        <ScrollText className="w-8 h-8 text-indigo-600" />
                        Research Cycles
                    </h1>
                    <p className="text-slate-500 text-sm mt-1">Autonomous research cycle history — Hypothesis → Experiment → Review</p>
                </div>
                <button onClick={fetchCycles} className="p-2 bg-white border rounded-lg hover:bg-slate-50">
                    <RefreshCw className="w-4 h-4" />
                </button>
            </div>

            {/* 통계 카드 */}
            {stats && (
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                    <StatCard label="Total Cycles" value={stats.total_cycles} icon={<ScrollText className="w-5 h-5 text-indigo-500" />} />
                    <StatCard label="Completed" value={stats.completed_cycles} icon={<CheckCircle className="w-5 h-5 text-emerald-500" />} />
                    <StatCard label="Hypotheses" value={stats.total_hypotheses} icon={<Brain className="w-5 h-5 text-purple-500" />} />
                    <StatCard label="Experiments" value={stats.total_experiments} icon={<FlaskConical className="w-5 h-5 text-cyan-500" />} />
                    <StatCard label="Reviews" value={stats.total_reviews} icon={<Scale className="w-5 h-5 text-amber-500" />} />
                    <StatCard
                        label="Approval Rate"
                        value={stats.total_cycles > 0
                            ? `${Math.round((stats.verdicts.ACCEPT / Math.max(1, stats.total_cycles)) * 100)}%`
                            : "—"}
                        icon={<CheckCircle className="w-5 h-5 text-green-500" />}
                    />
                </div>
            )}

            {/* 사이클 타임라인 */}
            <div className="space-y-4">
                {cycles.length === 0 ? (
                    <div className="text-center py-20 text-slate-400">
                        <ScrollText className="w-12 h-12 mx-auto mb-4 opacity-30" />
                        <p>No completed research cycles yet.</p>
                        <p className="text-sm mt-1">Try activating the Coordinator in the Control Room.</p>
                    </div>
                ) : (
                    cycles.map((cycle) => {
                        const isExpanded = expanded === cycle.id;
                        const verdictInfo = cycle.verdict ? VERDICT_STYLE[cycle.verdict] : null;
                        const VerdictIcon = verdictInfo?.icon || Clock;

                        return (
                            <div key={cycle.id} className="bg-white rounded-xl border shadow-sm overflow-hidden">
                                {/* 사이클 헤더 */}
                                <button
                                    onClick={() => setExpanded(isExpanded ? null : cycle.id)}
                                    className="w-full flex items-center justify-between p-4 hover:bg-slate-50 transition-colors text-left"
                                >
                                    <div className="flex items-center gap-4">
                                        {/* 사이클 번호 */}
                                        <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-sm ${cycle.status === "COMPLETE"
                                            ? "bg-gradient-to-br from-indigo-500 to-purple-600"
                                            : "bg-slate-300 animate-pulse"
                                            }`}>
                                            #{cycle.id}
                                        </div>

                                        <div>
                                            <div className="flex items-center gap-2">
                                                <span className="font-semibold text-slate-900">
                                                    Research Cycle #{cycle.id}
                                                </span>
                                                {verdictInfo && (
                                                    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${verdictInfo.bg} ${verdictInfo.color}`}>
                                                        <VerdictIcon className="w-3 h-3" />
                                                        {verdictInfo.label}
                                                    </span>
                                                )}
                                                {cycle.status === "RUNNING" && (
                                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-blue-50 border-blue-200 border text-blue-600 animate-pulse">
                                                        <Clock className="w-3 h-3" /> In Progress
                                                    </span>
                                                )}
                                            </div>
                                            <div className="text-xs text-slate-400 mt-0.5">
                                                {formatTime(cycle.started_at)}
                                                {cycle.ended_at && ` → ${formatTime(cycle.ended_at)}`}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-6">
                                        {/* 메트릭 뱃지 */}
                                        <div className="hidden md:flex items-center gap-3">
                                            <MetricBadge icon={<Brain className="w-3.5 h-3.5" />} value={cycle.hypotheses} label="Hypotheses" color="text-purple-500" />
                                            <MetricBadge icon={<FlaskConical className="w-3.5 h-3.5" />} value={cycle.experiments} label="Experiments" color="text-cyan-500" />
                                            <MetricBadge icon={<Scale className="w-3.5 h-3.5" />} value={cycle.reviews} label="Reviews" color="text-amber-500" />
                                        </div>

                                        <span className="text-xs text-slate-300">{cycle.log_count} entries</span>
                                        {isExpanded ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
                                    </div>
                                </button>

                                {/* 로그 상세 */}
                                {isExpanded && cycle.logs && (
                                    <div className="border-t bg-slate-900 max-h-64 overflow-y-auto">
                                        <div className="p-4 space-y-1 font-mono text-xs">
                                            {cycle.logs.map((log, i) => (
                                                <div key={i} className="flex gap-2">
                                                    <span className="text-slate-500 w-20 shrink-0">
                                                        {log.timestamp ? new Date(log.timestamp).toLocaleTimeString("en-US") : ""}
                                                    </span>
                                                    <span className={
                                                        log.level === "ERROR" ? "text-red-400"
                                                            : log.level === "WARNING" ? "text-yellow-400"
                                                                : log.message.includes("═══") ? "text-cyan-400 font-bold"
                                                                    : "text-green-300"
                                                    }>
                                                        {log.message}
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })
                )}
            </div>
        </div>
    );
}

function StatCard({ label, value, icon }: { label: string; value: number | string; icon: React.ReactNode }) {
    return (
        <div className="bg-white rounded-xl border p-4 flex items-center gap-3">
            <div className="p-2 bg-slate-50 rounded-lg">{icon}</div>
            <div>
                <div className="text-2xl font-bold text-slate-900">{value}</div>
                <div className="text-xs text-slate-500">{label}</div>
            </div>
        </div>
    );
}

function MetricBadge({ icon, value, label, color }: { icon: React.ReactNode; value: number; label: string; color: string }) {
    return (
        <div className={`flex items-center gap-1 ${color}`}>
            {icon}
            <span className="text-sm font-medium text-slate-700">{value}</span>
            <span className="text-xs text-slate-400">{label}</span>
        </div>
    );
}
