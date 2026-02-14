"use client";

import React, { useState, useEffect } from 'react';
import {
    LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';

interface PolicySimulatorProps {
    baseLimit?: number;
    baseDefaultRate?: number;
}

export default function PolicySimulator({
    baseLimit = 1000,
    baseDefaultRate = 0.02
}: PolicySimulatorProps) {
    const [intensity, setIntensity] = useState(500); // 한도 상향액
    const [targetPercent, setTargetPercent] = useState(20); // 상위 % 유저

    // Mock Data Calculation (based on Causal Inference logic)
    // Revenue: 한도 상향액 * 15% 이자 * 대상 유저 수
    // Cost: 한도 상향액 * 부실률(비선형 증가) * 대상 유저 수
    const totalUsers = 10000;
    const targetVolume = totalUsers * (targetPercent / 100);

    // 부실률: 한도 올릴수록, 타겟 넓힐수록 기하급수적 증가 (Risk)
    const riskFactor = 1 + (intensity / 2000) + (targetPercent / 50);
    const predictedDefaultRate = baseDefaultRate * riskFactor;

    const revenue = intensity * 0.15 * targetVolume;
    const cost = (baseLimit + intensity) * predictedDefaultRate * targetVolume;
    const netProfit = revenue - cost;
    const roi = (netProfit / (cost + 1e-10)) * 100;

    // Agent Opinions
    const getAgentVerdict = () => {
        if (netProfit > 500000 && predictedDefaultRate < 0.04) {
            return {
                verdict: "APPROVED",
                color: "text-green-400",
                advocate: "LTV +18% 예상. 적극 추진 권장.",
                critic: "리스크 관리 범위 내 (Safe).",
                judge: "🚀 전면 배포 (Rollout 100%) 승인"
            };
        } else if (netProfit > 0) {
            return {
                verdict: "CONDITIONAL",
                color: "text-yellow-400",
                advocate: "매출 증대 기회 있음.",
                critic: "부실률 상승 주의 (Warning).",
                judge: "⚖️ 20% 유저 대상 A/B 테스트 권장"
            };
        } else {
            return {
                verdict: "REJECTED",
                color: "text-red-400",
                advocate: "매출은 오르지만...",
                critic: "부실 비용이 수익을 초과함 (Danger).",
                judge: "🛑 기각 (리소스 회수)"
            };
        }
    };

    const agent = getAgentVerdict();

    // Chart Data Generation (Sensitivity Curve)
    const chartData = [];
    for (let i = 0; i <= 2000; i += 200) {
        const r_factor = 1 + (i / 2000) + (targetPercent / 50);
        const p_default = baseDefaultRate * r_factor;
        const rev = i * 0.15 * targetVolume;
        const cst = (baseLimit + i) * p_default * targetVolume;
        chartData.push({
            intensity: i,
            profit: rev - cst,
            risk: p_default * 100
        });
    }

    return (
        <div className="flex flex-col gap-6 w-full h-full text-slate-200">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                        <span className="w-3 h-8 bg-gradient-to-b from-green-400 to-green-600 rounded-full shadow-[0_0_10px_rgba(74,222,128,0.5)]"></span>
                        ROI Simulator
                        <span className="px-2 py-0.5 text-xs font-mono bg-slate-800 border border-slate-700 rounded text-slate-400">BETA</span>
                    </h2>
                    <p className="text-slate-400 text-sm mt-1">
                        AI 에이전트가 예측한 정책 변경 시뮬레이션 결과입니다.
                    </p>
                </div>
                <div className={`px-4 py-2 rounded-xl border bg-opacity-10 backdrop-blur-md ${agent.verdict === 'APPROVED' ? 'bg-green-500 border-green-500 text-green-400' : agent.verdict === 'CONDITIONAL' ? 'bg-yellow-500 border-yellow-500 text-yellow-400' : 'bg-red-500 border-red-500 text-red-400'}`}>
                    <span className="text-xs font-bold tracking-wider opacity-80">VERDICT</span>
                    <div className="text-xl font-black">{agent.judge}</div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-full">
                {/* Left: Controls & Agent Chat */}
                <div className="lg:col-span-4 flex flex-col gap-6">
                    {/* Controls */}
                    <div className="p-6 rounded-2xl bg-[#0F172A] border border-slate-800 shadow-xl space-y-6">
                        <div className="space-y-4">
                            <div className="flex justify-between items-end">
                                <label className="text-sm font-medium text-slate-300">신용 한도 상향 (Intensity)</label>
                                <span className="text-xl font-mono text-green-400 font-bold">+${intensity}</span>
                            </div>
                            <input
                                type="range" min="0" max="2000" step="100"
                                value={intensity} onChange={(e) => setIntensity(Number(e.target.value))}
                                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-green-500"
                            />
                            <div className="flex justify-between text-xs text-slate-500 font-mono">
                                <span>$0</span>
                                <span>$2,000</span>
                            </div>
                        </div>

                        <div className="space-y-4 pt-4 border-t border-slate-800">
                            <div className="flex justify-between items-end">
                                <label className="text-sm font-medium text-slate-300">타겟 유저 (Risk Score 상위)</label>
                                <span className="text-xl font-mono text-blue-400 font-bold">Top {targetPercent}%</span>
                            </div>
                            <input
                                type="range" min="5" max="100" step="5"
                                value={targetPercent} onChange={(e) => setTargetPercent(Number(e.target.value))}
                                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                            />
                            <div className="flex justify-between text-xs text-slate-500 font-mono">
                                <span>Elite (5%)</span>
                                <span>All (100%)</span>
                            </div>
                        </div>
                    </div>

                    {/* Agent Opinions */}
                    <div className="flex-1 p-6 rounded-2xl bg-[#0F172A] border border-slate-800 shadow-xl space-y-4 overflow-y-auto">
                        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest">Agent Debate</h3>

                        <div className="flex gap-3">
                            <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400 text-xs font-bold border border-blue-500/30">G</div>
                            <div className="bg-slate-800/50 p-3 rounded-r-xl rounded-bl-xl border border-slate-700 text-sm text-slate-300">
                                <div className="text-xs text-blue-400 font-bold mb-1">Growth Hacker</div>
                                {agent.advocate}
                            </div>
                        </div>

                        <div className="flex gap-3">
                            <div className="w-8 h-8 rounded-full bg-red-500/20 flex items-center justify-center text-red-400 text-xs font-bold border border-red-500/30">R</div>
                            <div className="bg-slate-800/50 p-3 rounded-r-xl rounded-bl-xl border border-slate-700 text-sm text-slate-300">
                                <div className="text-xs text-red-400 font-bold mb-1">Risk Manager</div>
                                {agent.critic}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right: Big Matrix & Charts */}
                <div className="lg:col-span-8 flex flex-col gap-6">
                    {/* Big Numbers */}
                    <div className="grid grid-cols-2 gap-4">
                        <div className="p-6 rounded-2xl bg-gradient-to-br from-green-500/10 to-transparent border border-green-500/20 shadow-[0_0_20px_rgba(34,197,94,0.1)]">
                            <div className="text-sm text-green-300 font-medium mb-1">예상 순이익 (Net Profit)</div>
                            <div className="text-4xl font-black text-white tracking-tight">
                                ${netProfit.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                            </div>
                            <div className="mt-2 text-xs font-mono text-green-400 bg-green-500/10 inline-block px-2 py-1 rounded">
                                ROI {roi.toFixed(1)}%
                            </div>
                        </div>
                        <div className={`p-6 rounded-2xl bg-gradient-to-br from-slate-800 to-transparent border ${predictedDefaultRate > 0.04 ? 'border-red-500/30 bg-red-500/5' : 'border-slate-700'}`}>
                            <div className="text-sm text-slate-400 font-medium mb-1">예상 부실률 (Default Rate)</div>
                            <div className={`text-4xl font-black tracking-tight ${predictedDefaultRate > 0.04 ? 'text-red-400' : 'text-slate-200'}`}>
                                {(predictedDefaultRate * 100).toFixed(2)}%
                            </div>
                            <div className="mt-2 text-xs text-slate-500">
                                허용 한도: 4.0%
                            </div>
                        </div>
                    </div>

                    {/* Chart Area */}
                    <div className="flex-1 p-6 rounded-2xl bg-[#0F172A] border border-slate-800 shadow-xl relative min-h-[300px]">
                        <h3 className="text-sm font-bold text-slate-400 mb-6 flex justify-between">
                            <span>Profit Sensitivity Curve</span>
                            <span className="text-xs font-normal opacity-50">Impact of Credit Limit Increase</span>
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
                                <YAxis stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(value) => `$${value / 1000}k`} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                                    formatter={(value: number | undefined) => [
                                        `$${(value || 0).toLocaleString()}`,
                                        'Profit'
                                    ]}
                                />
                                <ReferenceLine x={intensity} stroke="#f59e0b" strokeDasharray="3 3" />
                                <Area type="monotone" dataKey="profit" stroke="#22c55e" strokeWidth={3} fillOpacity={1} fill="url(#profitGradient)" />
                            </AreaChart>
                        </ResponsiveContainer>

                        {/* Current Position Marker Text */}
                        <div className="absolute top-4 right-6 text-right">
                            <div className="text-xs text-slate-500">Current Simulation</div>
                            <div className="font-mono text-amber-500">Limit +${intensity}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
