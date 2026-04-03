"use client";

import { useEffect, useState, useCallback } from "react";
import { MessageSquare, Play, Users, ChevronRight } from "lucide-react";

interface DebateEntry {
    phase: string;
    role: string;
    agent_name: string;
    icon: string;
    content: string;
    timestamp: string;
}

interface Consensus {
    type: string;
    label: string;
    text: string;
    votes: Record<string, string>;
}

interface ForumResult {
    topic: { topic: string; context: string; domain: string };
    debate_log: DebateEntry[];
    consensus: Consensus;
    generated_at: string;
    total_statements: number;
}

const PHASE_LABELS: Record<string, { label: string; color: string; bg: string }> = {
    AGENDA: { label: "Agenda", color: "text-slate-600", bg: "bg-slate-100" },
    OPENING: { label: "Opening", color: "text-blue-600", bg: "bg-blue-50" },
    REBUTTAL: { label: "Rebuttal", color: "text-orange-600", bg: "bg-orange-50" },
    CONSENSUS: { label: "Consensus", color: "text-emerald-600", bg: "bg-emerald-50" },
    VOTE: { label: "Vote", color: "text-purple-600", bg: "bg-purple-50" },
};

const ROLE_STYLES: Record<string, { gradient: string; border: string }> = {
    Theorist: { gradient: "from-purple-500 to-indigo-600", border: "border-purple-200" },
    Engineer: { gradient: "from-cyan-500 to-teal-600", border: "border-cyan-200" },
    Critic: { gradient: "from-amber-500 to-orange-600", border: "border-amber-200" },
    System: { gradient: "from-slate-400 to-slate-600", border: "border-slate-200" },
};

export default function ForumPage() {
    const [result, setResult] = useState<ForumResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [animIdx, setAnimIdx] = useState(0);
    const [animating, setAnimating] = useState(false);

    const runDebate = useCallback(async () => {
        setLoading(true);
        setResult(null);
        setAnimIdx(0);
        try {
            const res = await fetch("http://localhost:4001/system/forum", { method: "POST" });
            const data = await res.json();
            setResult(data);
            // 순차 애니메이션
            setAnimating(true);
            setAnimIdx(0);
        } catch (error) {
            console.error("Forum error:", error);
        } finally {
            setLoading(false);
        }
    }, []);

    // 순차적으로 발언 표시
    useEffect(() => {
        if (!animating || !result) return;
        if (animIdx >= result.debate_log.length) {
            setAnimating(false);
            return;
        }
        const timer = setTimeout(() => setAnimIdx((i) => i + 1), 600);
        return () => clearTimeout(timer);
    }, [animIdx, animating, result]);

    return (
        <div className="p-8 space-y-6 bg-slate-50 min-h-screen">
            {/* 헤더 */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight text-slate-900 flex items-center gap-3">
                        <MessageSquare className="w-8 h-8 text-indigo-600" />
                        Academic Forum
                    </h1>
                    <p className="text-slate-500 text-sm mt-1">Agent academic debate — Opening → Rebuttal → Consensus</p>
                </div>
                <button
                    onClick={runDebate}
                    disabled={loading}
                    className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-semibold hover:opacity-90 transition disabled:opacity-50"
                >
                    <Play className="w-4 h-4" />
                    {loading ? "Debate in progress..." : "Start New Debate"}
                </button>
            </div>

            {/* 논제 카드 */}
            {result && (
                <div className="bg-gradient-to-r from-indigo-600 to-purple-700 rounded-2xl p-6 text-white">
                    <div className="flex items-center gap-2 text-sm opacity-80 mb-2">
                        <Users className="w-4 h-4" />
                        {result.topic.domain} · {result.total_statements} statements
                    </div>
                    <h2 className="text-xl font-bold">{result.topic.topic}</h2>
                    <p className="text-sm opacity-70 mt-2">{result.topic.context}</p>
                </div>
            )}

            {/* 토론 타임라인 */}
            {result && (
                <div className="space-y-3">
                    {result.debate_log.slice(0, animating ? animIdx : result.debate_log.length).map((entry, i) => {
                        const phase = PHASE_LABELS[entry.phase] || PHASE_LABELS.AGENDA;
                        const roleStyle = ROLE_STYLES[entry.role] || ROLE_STYLES.System;

                        return (
                            <div
                                key={i}
                                className={`flex gap-4 animate-fadeIn ${entry.role === "System" ? "pl-0" : "pl-8"
                                    }`}
                                style={{ animationDelay: `${i * 100}ms` }}
                            >
                                {/* 아바타 */}
                                <div className={`w-10 h-10 rounded-full bg-gradient-to-br ${roleStyle.gradient} flex items-center justify-center text-white text-lg shrink-0 shadow-md`}>
                                    {entry.icon}
                                </div>

                                {/* 발언 내용 */}
                                <div className={`flex-1 bg-white rounded-xl border ${roleStyle.border} p-4 shadow-sm`}>
                                    <div className="flex items-center gap-2 mb-2">
                                        <span className="font-bold text-sm text-slate-900">
                                            {entry.agent_name}
                                        </span>
                                        <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${phase.bg} ${phase.color}`}>
                                            {phase.label}
                                        </span>
                                        {entry.role !== "System" && (
                                            <span className="text-[10px] text-slate-300">{entry.role}</span>
                                        )}
                                    </div>
                                    <p className="text-sm text-slate-700 leading-relaxed">
                                        {entry.content}
                                    </p>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {/* 합의 결과 */}
            {result && !animating && (
                <div className={`rounded-2xl border-2 p-6 ${result.consensus.type === "CONSENSUS"
                    ? "bg-emerald-50 border-emerald-300"
                    : result.consensus.type === "PARTIAL"
                        ? "bg-amber-50 border-amber-300"
                        : "bg-slate-50 border-slate-300"
                    }`}>
                    <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
                        <ChevronRight className="w-5 h-5" />
                        {result.consensus.label}
                    </h3>
                    <p className="text-sm text-slate-700 leading-relaxed mb-4">
                        {result.consensus.text}
                    </p>
                    <div className="flex gap-4">
                        {Object.entries(result.consensus.votes).map(([role, vote]) => (
                            <div key={role} className="flex items-center gap-2 text-sm">
                                <span className="font-medium text-slate-600">{role}:</span>
                                <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${vote === "찬성" ? "bg-emerald-100 text-emerald-700"
                                    : vote === "조건부 찬성" ? "bg-amber-100 text-amber-700"
                                        : "bg-slate-100 text-slate-600"
                                    }`}>
                                    {vote}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* 빈 상태 */}
            {!result && !loading && (
                <div className="text-center py-20 text-slate-400">
                    <MessageSquare className="w-12 h-12 mx-auto mb-4 opacity-30" />
                    <p>No debates have been conducted yet.</p>
                    <p className="text-sm mt-1">Click &ldquo;Start New Debate&rdquo; to begin.</p>
                </div>
            )}

            <style jsx>{`
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(12px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .animate-fadeIn {
                    animation: fadeIn 0.4s ease-out forwards;
                }
            `}</style>
        </div>
    );
}
