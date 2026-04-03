"use client";

import React, { useState } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine,
    ComposedChart, Line
} from 'recharts';
import { CausalAnalysisResult } from "@/types";
import { clsx } from 'clsx';
import { motion } from 'framer-motion';

export default function CausalCharts({ data }: { data: CausalAnalysisResult }) {
    const [activeTab, setActiveTab] = useState<'distribution' | 'segments'>('distribution');

    // Histogram Data Transformation
    const histData = data.cate_distribution.histogram.bin_edges.slice(0, -1).map((edge, i) => ({
        range: `${edge.toFixed(2)} ~ ${data.cate_distribution.histogram.bin_edges[i + 1].toFixed(2)}`,
        count: data.cate_distribution.histogram.counts[i],
        mid: (edge + data.cate_distribution.histogram.bin_edges[i + 1]) / 2
    }));

    // Segment Data Transformation
    const segmentData = data.segments.map(seg => ({
        name: seg.name,
        cate: seg.cate_mean,
        ci_min: seg.cate_ci_lower,
        ci_max: seg.cate_ci_upper,
        size: seg.n
    }));

    return (
        <div className="glass-card h-full flex flex-col">
            {/* Header & Tabs */}
            <div className="flex justify-between items-center mb-6">
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <span className="w-2 h-6 bg-brand-500 rounded-full"></span>
                    Effective Analysis
                </h3>
                <div className="flex bg-slate-800/50 p-1 rounded-lg border border-white/5">
                    <button
                        onClick={() => setActiveTab('distribution')}
                        className={clsx("px-3 py-1.5 text-xs font-medium rounded transition-all", activeTab === 'distribution' ? "bg-white/10 text-white" : "text-slate-400 hover:text-white")}
                    >
                        Distribution
                    </button>
                    <button
                        onClick={() => setActiveTab('segments')}
                        className={clsx("px-3 py-1.5 text-xs font-medium rounded transition-all", activeTab === 'segments' ? "bg-white/10 text-white" : "text-slate-400 hover:text-white")}
                    >
                        Segments
                    </button>
                </div>
            </div>

            {/* Chart Area */}
            <div className="flex-1 w-full min-h-[250px]">
                <ResponsiveContainer width="100%" height="100%">
                    {activeTab === 'distribution' ? (
                        <BarChart data={histData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                            <XAxis dataKey="range" stroke="#94a3b8" tick={{ fontSize: 10 }} interval={0} />
                            <YAxis stroke="#94a3b8" tick={{ fontSize: 10 }} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#fff' }}
                                cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                            />
                            <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Users" />
                            <ReferenceLine x={data.cate_distribution.mean} stroke="#22d3ee" strokeDasharray="3 3" label={{ value: 'Mean', fill: '#22d3ee', fontSize: 10 }} />
                        </BarChart>
                    ) : (
                        <ComposedChart data={segmentData} layout="vertical" margin={{ top: 20, right: 30, left: 40, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={true} vertical={false} />
                            <XAxis type="number" stroke="#94a3b8" tick={{ fontSize: 10 }} />
                            <YAxis dataKey="name" type="category" stroke="#94a3b8" tick={{ fontSize: 11 }} width={80} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#fff' }}
                                cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                            />
                            <Bar dataKey="cate" fill="#22d3ee" barSize={20} radius={[0, 4, 4, 0]} name="CATE (Mean)">
                            </Bar>
                            {/* Error Bars visual trick could be added here, but omitted for simplicity */}
                        </ComposedChart>
                    )}
                </ResponsiveContainer>
            </div>

            {/* Footer Insight */}
            <div className="mt-4 pt-4 border-t border-white/5 text-xs text-slate-400">
                {activeTab === 'distribution'
                    ? `평균 처치 효과(CATE)는 ${data.cate_distribution.mean.toFixed(3)}이며, 표준편차는 ${data.cate_distribution.std.toFixed(3)}입니다.`
                    : `세그먼트별 효과 차이가 관찰됩니다. (데이터 기반)`
                }
            </div>
        </div>
    );
}
