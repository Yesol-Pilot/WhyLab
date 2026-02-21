"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { Rocket, Power, PowerOff, Loader2, Activity, Clock, CheckCircle, AlertTriangle, Zap } from "lucide-react";

interface PhaseEntry {
    phase: string;
    timestamp: string;
}

interface CycleHistory {
    cycle: number;
    started_at: string;
    ended_at?: string;
    status: string;
    phases: PhaseEntry[];
    error?: string;
}

interface AutopilotStatus {
    running: boolean;
    current_phase: string;
    cycle_count: number;
    started_at: string | null;
    last_cycle_at: string | null;
    history: CycleHistory[];
    interval_seconds: number;
}

const PHASE_LABELS: Record<string, { label: string; color: string; icon: string }> = {
    IDLE: { label: "Idle", color: "text-slate-400", icon: "⏸️" },
    STARTING: { label: "Starting", color: "text-blue-500", icon: "🚀" },
    RESEARCH_CYCLE: { label: "Research Cycle", color: "text-purple-500", icon: "🔬" },
    EVOLUTION: { label: "Agent Evolution", color: "text-emerald-500", icon: "🧬" },
    FORUM: { label: "Academic Forum", color: "text-orange-500", icon: "💬" },
    REPORT_GENERATION: { label: "Report Generation", color: "text-indigo-500", icon: "📄" },
    WAITING: { label: "Waiting for Next Cycle", color: "text-slate-400", icon: "⏳" },
    STOPPING: { label: "Stopping", color: "text-red-400", icon: "🛑" },
};

export default function AutopilotPage() {
    const [status, setStatus] = useState<AutopilotStatus | null>(null);
    const [loading, setLoading] = useState(false);
    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

    const fetchStatus = useCallback(async () => {
        try {
            const res = await fetch("http://localhost:4001/system/autopilot/status");
            const data = await res.json();
            setStatus(data);
        } catch (err) {
            console.error("Autopilot status error:", err);
        }
    }, []);

    // 실시간 폴링 (2초)
    useEffect(() => {
        fetchStatus();
        intervalRef.current = setInterval(fetchStatus, 2000);
        return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
    }, [fetchStatus]);

    const handleStart = async () => {
        setLoading(true);
        try {
            await fetch("http://localhost:4001/system/autopilot/start", { method: "POST" });
            await fetchStatus();
        } finally {
            setLoading(false);
        }
    };

    const handleStop = async () => {
        setLoading(true);
        try {
            await fetch("http://localhost:4001/system/autopilot/stop", { method: "POST" });
            await fetchStatus();
        } finally {
            setLoading(false);
        }
    };

    const phase = status ? PHASE_LABELS[status.current_phase] || PHASE_LABELS.IDLE : PHASE_LABELS.IDLE;

    return (
        <div className="p-8 space-y-6 bg-slate-50 min-h-screen">
            {/* 헤더 */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight text-slate-900 flex items-center gap-3">
                        <Rocket className="w-8 h-8 text-indigo-600" />
                        Autopilot
                    </h1>
                    <p className="text-slate-500 text-sm mt-1">
                        Fully autonomous research loop — Research Cycle → Evolution → Forum → Report, no user intervention required
                    </p>
                </div>
            </div>

            {/* 메인 제어 패널 */}
            <div className={`rounded-2xl border-2 p-8 transition-all duration-500 ${status?.running
                ? "bg-gradient-to-br from-indigo-50 to-purple-50 border-indigo-300 shadow-lg shadow-indigo-100"
                : "bg-white border-slate-200"
                }`}>
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-6">
                        {/* 상태 인디케이터 */}
                        <div className={`w-20 h-20 rounded-full flex items-center justify-center text-3xl ${status?.running
                            ? "bg-gradient-to-br from-indigo-500 to-purple-600 animate-pulse shadow-lg"
                            : "bg-slate-100"
                            }`}>
                            {status?.running ? "🚀" : "⏸️"}
                        </div>

                        <div>
                            <h2 className="text-2xl font-bold text-slate-900">
                                {status?.running ? "Autonomous Research Running" : "Autonomous Research Idle"}
                            </h2>
                            <div className="flex items-center gap-3 mt-2">
                                <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium ${status?.running ? "bg-indigo-100 text-indigo-700" : "bg-slate-100 text-slate-500"
                                    }`}>
                                    <span>{phase.icon}</span>
                                    {phase.label}
                                </span>
                                {status && status.cycle_count > 0 && (
                                    <span className="text-sm text-slate-500">
                                        {status.cycle_count} cycles completed
                                    </span>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* 시작/정지 버튼 */}
                    <button
                        onClick={status?.running ? handleStop : handleStart}
                        disabled={loading}
                        className={`flex items-center gap-3 px-8 py-4 rounded-2xl font-bold text-lg transition-all disabled:opacity-50 ${status?.running
                            ? "bg-red-500 hover:bg-red-600 text-white shadow-lg shadow-red-200"
                            : "bg-gradient-to-r from-indigo-600 to-purple-600 hover:opacity-90 text-white shadow-lg shadow-indigo-200"
                            }`}
                    >
                        {loading ? (
                            <Loader2 className="w-6 h-6 animate-spin" />
                        ) : status?.running ? (
                            <PowerOff className="w-6 h-6" />
                        ) : (
                            <Power className="w-6 h-6" />
                        )}
                        {status?.running ? "Stop" : "Start Autonomous Research"}
                    </button>
                </div>

                {/* 파이프라인 시각화 */}
                {status?.running && (
                    <div className="mt-8 flex items-center justify-between">
                        {["RESEARCH_CYCLE", "EVOLUTION", "FORUM", "REPORT_GENERATION"].map((p, i) => {
                            const pInfo = PHASE_LABELS[p];
                            const isActive = status.current_phase === p;
                            const isPast = ["RESEARCH_CYCLE", "EVOLUTION", "FORUM", "REPORT_GENERATION"]
                                .indexOf(status.current_phase) > i;
                            return (
                                <div key={p} className="flex items-center flex-1">
                                    <div className={`flex flex-col items-center gap-2 flex-1 ${isActive ? "scale-110" : ""
                                        } transition-transform`}>
                                        <div className={`w-12 h-12 rounded-full flex items-center justify-center text-xl ${isActive
                                            ? "bg-gradient-to-br from-indigo-500 to-purple-600 shadow-lg animate-pulse"
                                            : isPast
                                                ? "bg-emerald-100"
                                                : "bg-slate-100"
                                            }`}>
                                            {isPast ? "✅" : pInfo.icon}
                                        </div>
                                        <span className={`text-xs font-medium ${isActive ? "text-indigo-700" : "text-slate-400"
                                            }`}>
                                            {pInfo.label}
                                        </span>
                                    </div>
                                    {i < 3 && (
                                        <div className={`h-0.5 w-full mx-2 ${isPast ? "bg-emerald-300" : isActive ? "bg-indigo-300 animate-pulse" : "bg-slate-200"
                                            }`} />
                                    )}
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            {/* 통계 카드 */}
            {status && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <StatCard
                        label="Total Cycles"
                        value={status.cycle_count}
                        icon={<Activity className="w-5 h-5 text-indigo-500" />}
                    />
                    <StatCard
                        label="Current Phase"
                        value={phase.label}
                        icon={<Zap className="w-5 h-5 text-amber-500" />}
                    />
                    <StatCard
                        label="Started At"
                        value={status.started_at ? new Date(status.started_at).toLocaleTimeString("en-US") : "—"}
                        icon={<Clock className="w-5 h-5 text-cyan-500" />}
                    />
                    <StatCard
                        label="Last Cycle"
                        value={status.last_cycle_at ? new Date(status.last_cycle_at).toLocaleTimeString("en-US") : "—"}
                        icon={<CheckCircle className="w-5 h-5 text-emerald-500" />}
                    />
                </div>
            )}

            {/* 사이클 히스토리 */}
            {status && status.history.length > 0 && (
                <div className="bg-white rounded-xl border shadow-sm">
                    <div className="p-4 border-b">
                        <h3 className="font-bold text-slate-900 flex items-center gap-2">
                            <Activity className="w-4 h-4" />
                            Cycle History
                        </h3>
                    </div>
                    <div className="divide-y">
                        {status.history.slice().reverse().map((cycle, i) => (
                            <div key={i} className="p-4 flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold text-white ${cycle.status === "COMPLETE"
                                        ? "bg-emerald-500"
                                        : cycle.status === "ERROR"
                                            ? "bg-red-500"
                                            : "bg-slate-300"
                                        }`}>
                                        #{cycle.cycle}
                                    </div>
                                    <div>
                                        <span className="text-sm font-medium text-slate-900">
                                            Cycle #{cycle.cycle}
                                        </span>
                                        <div className="text-xs text-slate-400">
                                            {new Date(cycle.started_at).toLocaleString("en-US")}
                                        </div>
                                    </div>
                                </div>

                                <div className="flex items-center gap-2">
                                    {cycle.phases.map((ph, j) => {
                                        const pLabel = PHASE_LABELS[ph.phase];
                                        return pLabel ? (
                                            <span key={j} className="text-sm" title={pLabel.label}>
                                                {pLabel.icon}
                                            </span>
                                        ) : null;
                                    })}
                                </div>

                                <span className={`px-2 py-1 rounded-full text-xs font-medium ${cycle.status === "COMPLETE"
                                    ? "bg-emerald-50 text-emerald-700"
                                    : cycle.status === "ERROR"
                                        ? "bg-red-50 text-red-700"
                                        : "bg-slate-50 text-slate-500"
                                    }`}>
                                    {cycle.status === "COMPLETE" ? "Complete" : cycle.status === "ERROR" ? "Error" : cycle.status}
                                </span>

                                {cycle.error && (
                                    <div className="text-xs text-red-500 flex items-center gap-1">
                                        <AlertTriangle className="w-3 h-3" />
                                        {cycle.error.slice(0, 50)}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

function StatCard({ label, value, icon }: { label: string; value: string | number; icon: React.ReactNode }) {
    return (
        <div className="bg-white rounded-xl border p-4 flex items-center gap-3">
            {icon}
            <div>
                <div className="text-xs text-slate-400">{label}</div>
                <div className="text-lg font-bold text-slate-900">{value}</div>
            </div>
        </div>
    );
}
