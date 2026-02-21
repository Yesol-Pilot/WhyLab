"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Play, Pause, RefreshCw } from "lucide-react";
import AgentNetwork from "@/components/AgentNetwork";
import LiveLogFeed from "@/components/LiveLogFeed";

interface LogMessage {
    id: string;
    agent_id: string;
    message: string;
    timestamp: string;
}

export default function LiveRoomPage() {
    const [logs, setLogs] = useState<LogMessage[]>([]);
    const [agenda, setAgenda] = useState<any>(null); // State for current agenda
    const [isPaused, setIsPaused] = useState(false);
    const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

    const [mounted, setMounted] = useState(false);

    const fetchLogs = async () => {
        if (isPaused) return;
        try {
            const res = await fetch("http://localhost:4001/system/logs?limit=50");
            if (res.ok) {
                const data = await res.json();
                // 백엔드는 created_at, 프론트엔드는 timestamp 사용 → 매핑
                const mapped = data.map((log: any) => ({
                    id: log.id?.toString() || Math.random().toString(),
                    agent_id: log.agent_id || "System",
                    message: log.message || "",
                    timestamp: log.created_at || log.timestamp || new Date().toISOString(),
                }));
                const sorted = mapped.sort((a: LogMessage, b: LogMessage) =>
                    new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
                );
                setLogs(sorted);
                setLastUpdate(new Date());
            }
        } catch (e) {
            console.error("Failed to fetch logs", e);
        }
    };

    const fetchAgenda = async () => {
        try {
            const res = await fetch("http://localhost:4001/system/director/agenda");
            if (res.ok) {
                const data = await res.json();
                setAgenda(data);
            }
        } catch (e) {
            console.error("Failed to fetch agenda", e);
        }
    };

    useEffect(() => {
        setMounted(true); // Client-side check

        // Initial Fetch
        fetchLogs();
        fetchAgenda();

        const interval = setInterval(() => {
            fetchLogs();
            // Fetch agenda less frequently
            if (Math.random() > 0.8) fetchAgenda();
        }, 1000);
        return () => clearInterval(interval);
    }, [isPaused]);

    if (!mounted) return null; // Prevent hydration mismatch

    return (
        <main className="min-h-screen bg-slate-950 text-white p-6 relative overflow-hidden">
            {/* Background Ambience */}
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-brand-500 via-accent-cyan to-accent-pink animate-gradient-x" />

            {/* Header */}
            <header className="flex items-center justify-between mb-8 z-10 relative">
                <div className="flex items-center gap-4">
                    <Link href="/dashboard" className="p-2 rounded-lg hover:bg-white/10 transition-colors">
                        <ArrowLeft className="w-5 h-5 text-slate-400" />
                    </Link>
                    <div>
                        <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
                            <span className="w-3 h-3 rounded-full bg-red-500 animate-pulse shadow-[0_0_10px_rgba(239,68,68,0.5)]" />
                            Research Control Room
                        </h1>
                        <p className="text-slate-500 text-sm font-mono mt-1">
                            LIVE MONITORING · AUTONOMOUS AGENT ORCHESTRATION
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    <div className="text-right hidden sm:block">
                        <div className="text-xs text-slate-500 font-mono">LAST UPDATE</div>
                        <div className="text-sm font-mono text-brand-400">
                            {lastUpdate.toLocaleTimeString()}
                        </div>
                    </div>
                    <button
                        onClick={() => setIsPaused(!isPaused)}
                        className={`p-3 rounded-xl border ${isPaused ? 'bg-yellow-500/10 border-yellow-500/50 text-yellow-500' : 'bg-slate-800 border-white/10 text-slate-300 hover:text-white'} transition-all`}
                    >
                        {isPaused ? <Play className="w-5 h-5 fill-current" /> : <Pause className="w-5 h-5 fill-current" />}
                    </button>
                </div>
            </header>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 z-10 relative max-w-[1600px] mx-auto">
                <LiveContent agenda={agenda} logs={logs} />
            </div>
        </main>
    );
}

// Split content to avoid hydration issues with useClient if needed, but here simple refactor
function LiveContent({ agenda, logs }: any) {
    return (
        <>
            {/* Left: Visualization */}
            <div className="space-y-6">
                {/* Director's Directive Banner */}
                <div className="bg-gradient-to-r from-blue-900/40 to-slate-900/40 border border-blue-500/30 p-4 rounded-xl flex items-start gap-4 shadow-lg backdrop-blur-sm relative overflow-hidden group">
                    <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity">
                        <span className="text-6xl">📢</span>
                    </div>
                    <div className="min-w-[40px] h-[40px] rounded-full bg-blue-500/20 flex items-center justify-center border border-blue-400/50">
                        <span className="text-xl">👔</span>
                    </div>
                    <div>
                        <h3 className="text-xs font-bold text-blue-400 uppercase tracking-widest mb-1">Current Research Directive</h3>
                        <div className="text-lg font-bold text-white mb-1">
                            {agenda ? agenda.title : "Awaiting Directive..."}
                        </div>
                        <p className="text-sm text-slate-400 leading-relaxed max-w-xl">
                            {agenda ? agenda.description : "The Lab Director is analyzing global trends to select the next research topic."}
                        </p>
                        {agenda && (
                            <div className="flex gap-3 mt-3">
                                <Badge label="Impact" value={agenda.impact} color="red" />
                                <Badge label="Difficulty" value={agenda.difficulty} color="yellow" />
                            </div>
                        )}
                    </div>
                </div>

                <section>
                    <div className="flex items-center justify-between mb-3 px-1">
                        <h2 className="text-lg font-semibold text-slate-200">Neural Activity</h2>
                        <span className="text-xs text-slate-500 border border-slate-700 px-2 py-0.5 rounded">REAL-TIME</span>
                    </div>
                    <AgentNetwork logs={logs} />
                </section>

                <section className="bg-slate-900/30 p-5 rounded-2xl border border-white/5">
                    <h3 className="text-sm font-bold text-slate-400 mb-3 uppercase tracking-wider">System Status</h3>
                    <div className="grid grid-cols-3 gap-4 text-center">
                        <StatusMetric label="CPU Usage" value="42%" trend="stabilized" />
                        <StatusMetric label="Memory" value="1.2GB" trend="increasing" />
                        <StatusMetric label="Active Agents" value="4" trend="optimal" />
                    </div>
                </section>
            </div>

            {/* Right: Logs & Feed */}
            <div className="space-y-6">
                <section className="h-full flex flex-col">
                    <div className="flex items-center justify-between mb-3 px-1">
                        <h2 className="text-lg font-semibold text-slate-200">Stream Output</h2>
                        <span className="text-xs text-brand-400 font-mono">/var/log/hive_mind.log</span>
                    </div>
                    <LiveLogFeed logs={logs} />
                </section>
            </div>
        </>
    );
}

function StatusMetric({ label, value, trend }: any) {
    return (
        <div className="p-3 rounded-lg bg-black/20 border border-white/5">
            <div className="text-xs text-slate-500 mb-1">{label}</div>
            <div className="text-xl font-bold text-white font-mono">{value}</div>
            <div className="text-[10px] text-slate-600 mt-1 uppercase">{trend}</div>
        </div>
    );
}

function Badge({ label, value, color }: any) {
    const getColorClass = (c: string) => {
        switch (c) {
            case "red": return "bg-red-500/10 text-red-400 border-red-500/20";
            case "yellow": return "bg-yellow-500/10 text-yellow-400 border-yellow-500/20";
            default: return "bg-slate-500/10 text-slate-400 border-slate-500/20";
        }
    };

    return (
        <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getColorClass(color)} uppercase`}>
            {label}: {value}
        </span>
    );
}
