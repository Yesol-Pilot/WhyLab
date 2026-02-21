"use client";

import React from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Area, ComposedChart, ReferenceLine, ReferenceDot } from "recharts";
import { Info } from "lucide-react";

interface DoseResponseResult {
    t_grid: number[];
    dr_curve: number[];
    ci_lower?: number[];
    ci_upper?: number[];
    optimal_dose: number;
    optimal_effect: number;
    has_effect: boolean;
}

interface DoseResponseChartProps {
    data: DoseResponseResult;
    treatmentName?: string;
    outcomeName?: string;
}

export default function DoseResponseChart({ data, treatmentName = "Treatment", outcomeName = "Response" }: DoseResponseChartProps) {
    if (!data) return <div className="text-center text-slate-500">No data available.</div>;

    // Recharts용 데이터 변환
    const chartData = data.t_grid.map((t, i) => ({
        t,
        y: data.dr_curve[i],
        lower: data.ci_lower ? data.ci_lower[i] : null,
        upper: data.ci_upper ? data.ci_upper[i] : null,
        // Range Area를 위한 데이터 (upper - lower)
        range: data.ci_lower && data.ci_upper ? [data.ci_lower[i], data.ci_upper[i]] : null
    }));

    return (
        <div className="w-full h-full min-h-[400px] flex flex-col">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h3 className="text-lg font-bold text-white mb-1">Dose-Response Curve</h3>
                    <p className="text-sm text-slate-400">
                        How {outcomeName} varies with {treatmentName} concentration
                    </p>
                </div>
                <div className="flex gap-4">
                    <div className="text-right">
                        <div className="text-xs text-slate-500 uppercase tracking-wider">Optimal Dose</div>
                        <div className="text-xl font-bold text-accent-cyan font-mono">
                            {data.optimal_dose.toFixed(2)}
                        </div>
                    </div>
                    <div className="text-right">
                        <div className="text-xs text-slate-500 uppercase tracking-wider">Max Effect</div>
                        <div className="text-xl font-bold text-brand-400 font-mono">
                            {data.optimal_effect.toFixed(2)}
                        </div>
                    </div>
                </div>
            </div>

            <div className="flex-1 w-full bg-slate-900/50 rounded-xl border border-white/5 p-4">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                        <defs>
                            <linearGradient id="ciGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.2} />
                                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                        <XAxis
                            dataKey="t"
                            stroke="#94a3b8"
                            fontSize={12}
                            tickLine={false}
                            axisLine={false}
                            label={{ value: treatmentName, position: 'bottom', fill: '#64748b', fontSize: 12, offset: 0 }}
                        />
                        <YAxis
                            stroke="#94a3b8"
                            fontSize={12}
                            tickLine={false}
                            axisLine={false}
                            label={{ value: outcomeName, angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 12 }}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc' }}
                            itemStyle={{ color: '#f8fafc' }}
                            formatter={(value: any) => typeof value === 'number' ? value.toFixed(3) : value}
                            labelFormatter={(label) => `${treatmentName}: ${Number(label).toFixed(2)}`}
                        />

                        {/* 신뢰구간 (Area) */}
                        {data.ci_lower && (
                            <Area
                                type="monotone"
                                dataKey="range"
                                stroke="none"
                                fill="url(#ciGradient)"
                                name="95% CI"
                            />
                        )}

                        {/* 메인 곡선 */}
                        <Line
                            type="monotone"
                            dataKey="y"
                            stroke="#c084fc"
                            strokeWidth={3}
                            dot={false}
                            name="Response"
                        />

                        {/* 최적 용량 포인트 */}
                        <ReferenceLine x={data.optimal_dose} stroke="#22d3ee" strokeDasharray="3 3" />
                        <ReferenceDot
                            x={data.optimal_dose}
                            y={data.optimal_effect}
                            r={6}
                            fill="#22d3ee"
                            stroke="#fff"
                            strokeWidth={2}
                        />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>

            <div className="mt-4 flex gap-6 justify-center">
                <div className="flex items-center gap-2 text-xs text-slate-400">
                    <span className="w-3 h-3 rounded-full bg-brand-400"></span>
                    Mean Response Curve
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-400">
                    <span className="w-3 h-3 rounded-full bg-brand-500/30 border border-brand-500"></span>
                    95% Confidence Interval
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-400">
                    <span className="w-3 h-3 rounded-full bg-accent-cyan"></span>
                    Optimal Dose
                </div>
            </div>
        </div>
    );
}
