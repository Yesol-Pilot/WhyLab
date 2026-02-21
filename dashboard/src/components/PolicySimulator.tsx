"use client";

import React, { useState, useEffect, useRef } from 'react';
import {
    LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { Loader2, AlertCircle } from "lucide-react";

interface PolicySimulatorProps {
    baseLimit?: number;
    baseDefaultRate?: number;
}

export default function PolicySimulator({
    baseLimit = 1000,
    baseDefaultRate = 0.02
}: PolicySimulatorProps) {
    const [intensity, setIntensity] = useState(500); // 한도/처치 상향액
    const [targetPercent, setTargetPercent] = useState(20); // 상위 % 유저
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<any>(null);

    // Debounce를 위한 ref
    const timeoutRef = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
        const sessionId = localStorage.getItem("whylab_session_id");
        if (!sessionId) {
            setError("No Session found. Please upload data first.");
            return;
        }

        const fetchSimulation = async () => {
            setLoading(true);
            setError(null);
            try {
                const res = await fetch("/api/analysis/simulate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        session_id: sessionId,
                        intensity: intensity,
                        target_percent: targetPercent,
                        cost_per_unit: 1.0 // 예: $1 당 훈련비 $1
                    })
                });

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || "Simulation failed");
                }

                const json = await res.json();
                setResult(json.result);
            } catch (err: any) {
                console.error(err);
                setError(err.message || "Failed to simulate");
            } finally {
                setLoading(false);
            }
        };

        // Debounce: 500ms 동안 변화가 없으면 API 호출
        if (timeoutRef.current) clearTimeout(timeoutRef.current);

        timeoutRef.current = setTimeout(() => {
            fetchSimulation();
        }, 500);

        return () => {
            if (timeoutRef.current) clearTimeout(timeoutRef.current);
        };
    }, [intensity, targetPercent]);

    // 결과 데이터 파싱
    const current = result?.current || { net_profit: 0, roi: 0, total_benefit: 0, total_cost: 0, avg_outcome_boost: 0 };
    const chartData = result?.sensitivity || [];

    // Agent Opinions (Based on Real ROI)
    const getAgentVerdict = () => {
        if (!result) return { verdict: "PENDING", judge: "Calculating..." };

        const roi = current.roi;
        const profit = current.net_profit;

        if (roi > 10 && profit > 0) {
            return {
                verdict: "APPROVED",
                color: "text-green-400",
                advocate: `ROI ${roi.toFixed(1)}% achieved. Expected profit: $${Math.floor(profit).toLocaleString()}.`,
                critic: "Profitability sufficient relative to risk.",
                judge: "🚀 Approved"
            };
        } else if (profit > 0) {
            return {
                verdict: "CONDITIONAL",
                color: "text-yellow-400",
                advocate: "Profitable but ROI is low.",
                critic: "Cost efficiency needs review.",
                judge: "⚖️ Conditional"
            };
        } else {
            return {
                verdict: "REJECTED",
                color: "text-red-400",
                advocate: "Revenue increase effect is minimal.",
                critic: `Costs exceed revenue (Loss: $${Math.abs(Math.floor(profit)).toLocaleString()}).`,
                judge: "🛑 Rejected"
            };
        }
    };

    const agent = getAgentVerdict();

    return (
        <div className="flex flex-col gap-6 w-full h-full text-slate-200">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                        <span className="w-3 h-8 bg-gradient-to-b from-brand-400 to-brand-600 rounded-full shadow-[0_0_10px_rgba(74,222,128,0.5)]"></span>
                        AI Policy Simulator
                        {loading && <Loader2 className="w-4 h-4 animate-spin text-slate-500" />}
                    </h2>
                    <p className="text-slate-400 text-sm mt-1">
                        Real-time causal effect simulation based on DoseResponse model
                    </p>
                </div>
                {agent.verdict !== "PENDING" && (
                    <div className={`px-4 py-2 rounded-xl border bg-opacity-10 backdrop-blur-md ${agent.verdict === 'APPROVED' ? 'bg-green-500 border-green-500 text-green-400' : agent.verdict === 'CONDITIONAL' ? 'bg-yellow-500 border-yellow-500 text-yellow-400' : 'bg-red-500 border-red-500 text-red-400'}`}>
                        <span className="text-xs font-bold tracking-wider opacity-80">VERDICT</span>
                        <div className="text-xl font-black">{agent.judge}</div>
                    </div>
                )}
            </div>

            {error && (
                <div className="bg-red-900/20 border border-red-800 text-red-300 p-4 rounded-lg flex items-center gap-3">
                    <AlertCircle className="w-5 h-5" />
                    {error}
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-full">
                {/* Left: Controls & Agent Chat */}
                <div className="lg:col-span-4 flex flex-col gap-6">
                    {/* Controls */}
                    <div className="p-6 rounded-2xl bg-[#0F172A] border border-slate-800 shadow-xl space-y-6">
                        <div className="space-y-4">
                            <div className="flex justify-between items-end">
                                <label className="text-sm font-medium text-slate-300">Treatment Intensity</label>
                                <span className="text-xl font-mono text-green-400 font-bold">+{intensity}</span>
                            </div>
                            <input
                                type="range" min="0" max="2000" step="100"
                                value={intensity} onChange={(e) => setIntensity(Number(e.target.value))}
                                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-green-500"
                            />
                        </div>

                        <div className="space-y-4 pt-4 border-t border-slate-800">
                            <div className="flex justify-between items-end">
                                <label className="text-sm font-medium text-slate-300">Target Scope (Top %)</label>
                                <span className="text-xl font-mono text-blue-400 font-bold">Top {targetPercent}%</span>
                            </div>
                            <input
                                type="range" min="5" max="100" step="5"
                                value={targetPercent} onChange={(e) => setTargetPercent(Number(e.target.value))}
                                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                            />
                        </div>
                    </div>

                    {/* Agent Opinions */}
                    <div className="flex-1 p-6 rounded-2xl bg-[#0F172A] border border-slate-800 shadow-xl space-y-4 overflow-y-auto">
                        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest">AI Agent Analysis</h3>

                        <div className="flex gap-3">
                            <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400 text-xs font-bold border border-blue-500/30">G</div>
                            <div className="bg-slate-800/50 p-3 rounded-r-xl rounded-bl-xl border border-slate-700 text-sm text-slate-300 w-full">
                                <div className="text-xs text-blue-400 font-bold mb-1">Benefit Analyst</div>
                                {agent.advocate || "Analyzing..."}
                            </div>
                        </div>

                        <div className="flex gap-3">
                            <div className="w-8 h-8 rounded-full bg-red-500/20 flex items-center justify-center text-red-400 text-xs font-bold border border-red-500/30">R</div>
                            <div className="bg-slate-800/50 p-3 rounded-r-xl rounded-bl-xl border border-slate-700 text-sm text-slate-300 w-full">
                                <div className="text-xs text-red-400 font-bold mb-1">Cost Analyst</div>
                                {agent.critic || "Analyzing..."}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right: Big Matrix & Charts */}
                <div className="lg:col-span-8 flex flex-col gap-6">
                    {/* Big Numbers */}
                    <div className="grid grid-cols-2 gap-4">
                        <div className="p-6 rounded-2xl bg-gradient-to-br from-green-500/10 to-transparent border border-green-500/20 shadow-[0_0_20px_rgba(34,197,94,0.1)]">
                            <div className="text-sm text-green-300 font-medium mb-1">Net Profit (Benefit - Cost)</div>
                            <div className="text-4xl font-black text-white tracking-tight">
                                ${current.net_profit?.toLocaleString(undefined, { maximumFractionDigits: 0 }) || 0}
                            </div>
                            <div className={`mt-2 text-xs font-mono inline-block px-2 py-1 rounded ${current.roi > 0 ? 'text-green-400 bg-green-500/10' : 'text-red-400 bg-red-500/10'}`}>
                                ROI {current.roi?.toFixed(1) || 0}%
                            </div>
                        </div>
                        <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-800 to-transparent border border-slate-700">
                            <div className="text-sm text-slate-400 font-medium mb-1">Total Benefit (Revenue)</div>
                            <div className="text-4xl font-black text-slate-200 tracking-tight">
                                ${current.total_benefit?.toLocaleString(undefined, { maximumFractionDigits: 0 }) || 0}
                            </div>
                            <div className="mt-2 text-xs text-slate-500">
                                Target: {current.target_users?.toLocaleString()} users
                            </div>
                        </div>
                    </div>

                    {/* Chart Area */}
                    <div className="flex-1 p-6 rounded-2xl bg-[#0F172A] border border-slate-800 shadow-xl relative min-h-[300px]">
                        <h3 className="text-sm font-bold text-slate-400 mb-6 flex justify-between">
                            <span>Profit Sensitivity Curve</span>
                            <span className="text-xs font-normal opacity-50">Impact of Treatment Intensity</span>
                        </h3>
                        <ResponsiveContainer width="100%" height="85%">
                            <AreaChart data={chartData}>
                                <defs>
                                    <linearGradient id="profitGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                                <XAxis dataKey="intensity" stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} />
                                <YAxis stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(value) => `$${value}`} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                                    formatter={((value: any) => [`$${Number(value ?? 0).toLocaleString()}`, 'Profit']) as any}
                                />
                                <ReferenceLine x={intensity} stroke="#f59e0b" strokeDasharray="3 3" />
                                <Area type="monotone" dataKey="profit" stroke="#22c55e" strokeWidth={3} fillOpacity={1} fill="url(#profitGradient)" />
                            </AreaChart>
                        </ResponsiveContainer>

                        {/* Current Position Marker Text */}
                        <div className="absolute top-4 right-6 text-right">
                            <div className="text-xs text-slate-500">Current Simulation</div>
                            <div className="font-mono text-amber-500">Intensity +{intensity}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
