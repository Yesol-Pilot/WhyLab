"use client";

import { useEffect, useState, useCallback } from "react";
import { GitBranch, Zap, RefreshCw, Award, ArrowRight } from "lucide-react";

interface AgentNode {
    id: string;
    name: string;
    role: string;
    generation: number;
    parent_id: string | null;
    status: string;
    config: Record<string, unknown> | null;
    created_at: string | null;
}

const ROLE_COLORS: Record<string, { bg: string; border: string; text: string; glow: string }> = {
    Theorist: { bg: "bg-purple-500/10", border: "border-purple-500/40", text: "text-purple-400", glow: "shadow-purple-500/20" },
    Engineer: { bg: "bg-emerald-500/10", border: "border-emerald-500/40", text: "text-emerald-400", glow: "shadow-emerald-500/20" },
    Critic: { bg: "bg-amber-500/10", border: "border-amber-500/40", text: "text-amber-400", glow: "shadow-amber-500/20" },
    Coordinator: { bg: "bg-blue-500/10", border: "border-blue-500/40", text: "text-blue-400", glow: "shadow-blue-500/20" },
};

const ROLE_ICONS: Record<string, string> = {
    Theorist: "🧠",
    Engineer: "🔬",
    Critic: "⚖️",
    Coordinator: "🎯",
};

export default function EvolutionPage() {
    const [agents, setAgents] = useState<AgentNode[]>([]);
    const [isEvolving, setIsEvolving] = useState(false);
    const [result, setResult] = useState<{ evolved_agents: { name: string; role: string; specialization: string; parent_score: number }[] } | null>(null);

    const fetchTree = useCallback(async () => {
        try {
            const res = await fetch("http://localhost:4001/system/evolution-tree");
            const data = await res.json();
            setAgents(data.agents || []);
        } catch (error) {
            console.error("Tree fetch error:", error);
        }
    }, []);

    useEffect(() => { fetchTree(); }, [fetchTree]);

    const triggerEvolution = async () => {
        setIsEvolving(true);
        setResult(null);
        try {
            const res = await fetch("http://localhost:4001/system/evolve", { method: "POST" });
            const data = await res.json();
            setResult(data);
            await fetchTree(); // 진화 후 트리 갱신
        } catch (error) {
            console.error("Evolution error:", error);
        } finally {
            setIsEvolving(false);
        }
    };

    // 세대별 그룹화
    const generations: Record<number, AgentNode[]> = {};
    agents.forEach(agent => {
        const gen = agent.generation || 1;
        if (!generations[gen]) generations[gen] = [];
        generations[gen].push(agent);
    });
    const genKeys = Object.keys(generations).map(Number).sort((a, b) => a - b);

    return (
        <div className="p-8 space-y-6 bg-slate-50 min-h-screen">
            {/* 헤더 */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight text-slate-900 flex items-center gap-3">
                        <GitBranch className="w-8 h-8 text-purple-600" />
                        Agent Evolution
                    </h1>
                    <p className="text-slate-500 text-sm mt-1">Performance-based agent evolution and specialization timeline</p>
                </div>
                <div className="flex items-center gap-3">
                    <button onClick={fetchTree} className="p-2 bg-white border rounded-lg hover:bg-slate-50">
                        <RefreshCw className="w-4 h-4" />
                    </button>
                    <button
                        onClick={triggerEvolution}
                        disabled={isEvolving}
                        className="px-5 py-2.5 bg-gradient-to-r from-purple-600 via-fuchsia-600 to-pink-600 text-white rounded-lg flex items-center gap-2 hover:from-purple-700 hover:via-fuchsia-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-wait transition-all shadow-lg shadow-purple-500/25"
                    >
                        <Zap className={`w-4 h-4 ${isEvolving ? "animate-pulse" : ""}`} />
                        {isEvolving ? "Evolving..." : "Run Evolution"}
                    </button>
                </div>
            </div>

            {/* 진화 결과 알림 */}
            {result && result.evolved_agents && result.evolved_agents.length > 0 && (
                <div className="bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-xl p-4">
                    <div className="flex items-center gap-2 text-purple-700 font-semibold mb-2">
                        <Award className="w-5 h-5" />
                        Evolution complete! {result.evolved_agents.length} Gen 2 agents created
                    </div>
                    <div className="flex flex-wrap gap-3">
                        {result.evolved_agents.map((agent, i) => (
                            <div key={i} className="bg-white/80 rounded-lg px-3 py-2 border border-purple-100 text-sm">
                                <span className="font-bold text-purple-600">{agent.name}</span>
                                <span className="text-slate-500 ml-1">({agent.role})</span>
                                <div className="text-xs text-slate-400 mt-0.5">
                                    {agent.specialization} · Parent score: {agent.parent_score}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* 세대별 타임라인 */}
            <div className="space-y-8">
                {genKeys.map((gen, genIdx) => (
                    <div key={gen} className="relative">
                        {/* 세대 헤더 */}
                        <div className="flex items-center gap-3 mb-4">
                            <div className={`px-3 py-1 rounded-full text-sm font-bold ${gen === 1
                                ? "bg-slate-800 text-white"
                                : "bg-gradient-to-r from-purple-600 to-pink-600 text-white"
                                }`}>
                                Gen {gen}
                            </div>
                            <div className="flex-1 h-px bg-slate-200"></div>
                            <span className="text-xs text-slate-400">
                                {generations[gen].length} agents
                            </span>
                        </div>

                        {/* 에이전트 카드 그리드 */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                            {generations[gen].map((agent) => {
                                const colors = ROLE_COLORS[agent.role] || ROLE_COLORS.Coordinator;
                                const icon = ROLE_ICONS[agent.role] || "🤖";
                                const parent = agent.parent_id
                                    ? agents.find(a => a.id === agent.parent_id)
                                    : null;
                                const specialization = agent.config && typeof agent.config === 'object'
                                    ? (agent.config as Record<string, string>).specialization
                                    : null;

                                return (
                                    <div
                                        key={agent.id}
                                        className={`rounded-xl border-2 p-4 transition-all hover:shadow-lg ${colors.bg} ${colors.border} ${colors.glow} ${gen > 1 ? "animate-pulse-once" : ""
                                            }`}
                                    >
                                        {/* 에이전트 헤더 */}
                                        <div className="flex items-start justify-between mb-3">
                                            <div className="flex items-center gap-2">
                                                <span className="text-2xl">{icon}</span>
                                                <div>
                                                    <h3 className="font-bold text-slate-900">{agent.name}</h3>
                                                    <span className={`text-xs font-medium ${colors.text}`}>
                                                        {agent.role}
                                                    </span>
                                                </div>
                                            </div>
                                            <span className={`text-xs px-2 py-0.5 rounded-full ${agent.status === "IDLE"
                                                ? "bg-green-100 text-green-700"
                                                : agent.status === "WORKING"
                                                    ? "bg-yellow-100 text-yellow-700 animate-pulse"
                                                    : "bg-red-100 text-red-700"
                                                }`}>
                                                {agent.status}
                                            </span>
                                        </div>

                                        {/* 전문 분야 (Gen 2+) */}
                                        {specialization && (
                                            <div className="mb-2 text-xs bg-white/60 rounded-md px-2 py-1 text-slate-600">
                                                🎯 {specialization}
                                            </div>
                                        )}

                                        {/* 부모 정보 (Gen 2+) */}
                                        {parent && (
                                            <div className="flex items-center gap-1 text-xs text-slate-400">
                                                <ArrowRight className="w-3 h-3" />
                                                {ROLE_ICONS[parent.role]} Branched from {parent.name}
                                            </div>
                                        )}

                                        {/* ID */}
                                        <div className="mt-2 text-xs font-mono text-slate-300">
                                            {agent.id}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>

                        {/* 세대 간 연결선 */}
                        {genIdx < genKeys.length - 1 && (
                            <div className="flex justify-center my-4">
                                <div className="flex flex-col items-center">
                                    <div className="w-px h-6 bg-gradient-to-b from-purple-300 to-pink-300"></div>
                                    <Zap className="w-4 h-4 text-purple-500" />
                                    <div className="w-px h-6 bg-gradient-to-b from-pink-300 to-purple-300"></div>
                                </div>
                            </div>
                        )}
                    </div>
                ))}
            </div>

            {/* 빈 상태 */}
            {agents.length === 0 && (
                <div className="text-center py-20 text-slate-400">
                    <GitBranch className="w-12 h-12 mx-auto mb-4 opacity-30" />
                    <p>No agents registered yet.</p>
                </div>
            )}

            {/* 통계 카드 */}
            {agents.length > 0 && (
                <div className="grid grid-cols-4 gap-4">
                    <div className="bg-white rounded-xl border p-4 text-center">
                        <div className="text-3xl font-bold text-slate-900">{agents.length}</div>
                        <div className="text-xs text-slate-500 mt-1">Total Agents</div>
                    </div>
                    <div className="bg-white rounded-xl border p-4 text-center">
                        <div className="text-3xl font-bold text-purple-600">{genKeys.length}</div>
                        <div className="text-xs text-slate-500 mt-1">Generations</div>
                    </div>
                    <div className="bg-white rounded-xl border p-4 text-center">
                        <div className="text-3xl font-bold text-emerald-600">
                            {agents.filter(a => a.generation && a.generation > 1).length}
                        </div>
                        <div className="text-xs text-slate-500 mt-1">Evolved Agents</div>
                    </div>
                    <div className="bg-white rounded-xl border p-4 text-center">
                        <div className="text-3xl font-bold text-amber-600">
                            {new Set(agents.map(a => a.role)).size}
                        </div>
                        <div className="text-xs text-slate-500 mt-1">Role Types</div>
                    </div>
                </div>
            )}
        </div>
    );
}
