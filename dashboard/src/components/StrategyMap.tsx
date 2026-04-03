"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { BrainCircuit, Trophy, Target, Zap } from "lucide-react";

interface MethodStats {
    name: string;
    count: number;
    avg_reward: number;
    ucb1_score: number;
    generation: number;
}

interface StrategyData {
    categories: {
        [key: string]: {
            total_methods: number;
            methods: MethodStats[];
            last_10_selections: any[];
        };
    };
}

export default function StrategyMap() {
    const [data, setData] = useState<StrategyData | null>(null);
    const [loading, setLoading] = useState(true);

    // Poll strategy data every 3 seconds
    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await fetch("http://localhost:4001/system/methods");
                if (res.ok) {
                    const json = await res.json();
                    setData(json);
                }
            } catch (e) {
                console.error("MethodRegistry fetch failed", e);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 3000);
        return () => clearInterval(interval);
    }, []);

    if (loading) return <div className="text-slate-400">Loading Strategy Map...</div>;
    if (!data) return <div className="text-red-400">Failed to load strategy data.</div>;

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                    <BrainCircuit className="w-6 h-6 text-brand-400" />
                    Adaptive Strategy Map
                </h2>
                <div className="text-xs text-slate-500 bg-slate-800/50 px-3 py-1 rounded-full border border-slate-700">
                    Sync: Real-time (3s)
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Theorist Strategy */}
                <StrategyCard
                    role="Theorist"
                    icon={<Target className="w-5 h-5 text-blue-400" />}
                    category="hypothesis"
                    data={data.categories.hypothesis}
                    color="blue"
                />

                {/* Engineer Strategy */}
                <StrategyCard
                    role="Engineer"
                    icon={<Zap className="w-5 h-5 text-yellow-400" />}
                    category="experiment"
                    data={data.categories.experiment}
                    color="yellow"
                />

                {/* Critic Strategy */}
                <StrategyCard
                    role="Critic"
                    icon={<Trophy className="w-5 h-5 text-red-400" />}
                    category="review"
                    data={data.categories.review}
                    color="red"
                />
            </div>
        </div>
    );
}

function StrategyCard({ role, icon, category, data, color }: any) {
    const sortedMethods = [...(data?.methods || [])].sort((a, b) => b.count - a.count);
    const totalCalls = sortedMethods.reduce((sum: number, m: any) => sum + m.count, 0);

    const getBarColor = (c: string) => {
        switch (c) {
            case "blue": return "bg-blue-500";
            case "yellow": return "bg-yellow-500";
            case "red": return "bg-red-500";
            default: return "bg-slate-500";
        }
    };

    const getTextColor = (c: string) => {
        switch (c) {
            case "blue": return "text-blue-400";
            case "yellow": return "text-yellow-400";
            case "red": return "text-red-400";
            default: return "text-slate-400";
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="glass-card p-5 rounded-xl border border-white/10 bg-slate-900/50"
        >
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    {icon}
                    <h3 className={`font-bold text-lg ${getTextColor(color)}`}>{role}</h3>
                </div>
                <span className="text-xs text-slate-500">Total: {totalCalls}</span>
            </div>

            <div className="space-y-3">
                {sortedMethods.length === 0 && (
                    <div className="text-sm text-slate-500 text-center py-4">No data yet</div>
                )}

                {sortedMethods.map((m: MethodStats) => {
                    const percent = totalCalls > 0 ? (m.count / totalCalls) * 100 : 0;
                    return (
                        <div key={m.name} className="relative group">
                            <div className="flexjustify-between text-xs mb-1">
                                <span className="text-slate-300 font-medium truncate max-w-[120px]" title={m.name}>
                                    {m.name}
                                </span>
                                <span className="text-slate-400 ml-auto">
                                    {m.count}x <span className="text-slate-600">|</span> ⭐{m.avg_reward?.toFixed(2) ?? "0.00"}
                                </span>
                            </div>

                            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${percent}%` }}
                                    className={`h-full ${getBarColor(color)}/80`}
                                />
                            </div>

                            {/* Tooltip */}
                            <div className="absolute left-0 -top-8 hidden group-hover:block bg-black/90 text-xs text-white px-2 py-1 rounded border border-white/10 whitespace-nowrap z-10">
                                UCB1: {m.ucb1_score?.toFixed(3) ?? "0.000"} | Gen {m.generation ?? 0}
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Latest Selection Badge */}
            <div className="mt-4 pt-3 border-t border-white/5 flex items-center justify-between">
                <span className="text-xs text-slate-500">Latest Strategy</span>
                {data?.last_10_selections?.length > 0 ? (
                    <span className={`text-xs font-bold px-2 py-0.5 rounded bg-${color}-500/10 ${getTextColor(color)}`}>
                        {data.last_10_selections[data.last_10_selections.length - 1].selected}
                    </span>
                ) : (
                    <span className="text-xs text-slate-600">-</span>
                )}
            </div>
        </motion.div>
    );
}
