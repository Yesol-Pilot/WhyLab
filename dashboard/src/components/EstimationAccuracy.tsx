"use client";

import React from 'react';
import { CausalAnalysisResult } from "@/types";
import { motion } from 'framer-motion';

/**
 * Estimation Accuracy Panel — 논문 수준의 핵심
 *
 * 합성 데이터의 Ground Truth(true_cate)와 DML 추정치(estimated_cate)를 비교하여
 * 모델의 실제 성능을 정량적으로 검증합니다.
 */
export default function EstimationAccuracy({ data }: { data: CausalAnalysisResult }) {
    const acc = data.estimation_accuracy;

    if (!acc) {
        return null;
    }

    const coveragePct = (acc.coverage_rate * 100).toFixed(1);
    const coverageOk = acc.coverage_rate >= 0.90;

    const metrics = [
        {
            label: "RMSE",
            value: acc.rmse.toFixed(4),
            desc: "추정 오차 (낮을수록 정확)",
            color: "text-cyan-400",
        },
        {
            label: "MAE",
            value: acc.mae.toFixed(4),
            desc: "평균 절대 오차",
            color: "text-blue-400",
        },
        {
            label: "Bias",
            value: `${acc.bias > 0 ? '+' : ''}${acc.bias.toFixed(4)}`,
            desc: "체계적 편향 (0에 가까울수록 양호)",
            color: Math.abs(acc.bias) < 0.01 ? "text-emerald-400" : "text-amber-400",
        },
        {
            label: "Coverage",
            value: `${coveragePct}%`,
            desc: `95% CI가 Ground Truth를 포함하는 비율`,
            color: coverageOk ? "text-emerald-400" : "text-red-400",
            badge: coverageOk ? "Pass" : "Fail",
            badgeColor: coverageOk ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400",
        },
        {
            label: "Correlation",
            value: acc.correlation.toFixed(3),
            desc: "추정 ↔ 실제 방향성 일치도",
            color: acc.correlation > 0.8 ? "text-emerald-400" : acc.correlation > 0.5 ? "text-amber-400" : "text-red-400",
        },
    ];

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card"
        >
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h2 className="text-xl font-bold">📊 Estimation Accuracy</h2>
                    <p className="text-gray-400 text-sm mt-1">
                        Ground Truth (합성 데이터) vs DML 추정치 비교 · N={acc.n_samples.toLocaleString()}
                    </p>
                </div>
                <div className="text-xs text-gray-500 bg-gray-800/50 px-3 py-1 rounded-full">
                    Synthetic Data Validation
                </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
                {metrics.map((m) => (
                    <div key={m.label} className="bg-slate-800/50 rounded-lg p-4 border border-white/5">
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-xs text-gray-400 font-medium">{m.label}</span>
                            {m.badge && (
                                <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${m.badgeColor}`}>
                                    {m.badge}
                                </span>
                            )}
                        </div>
                        <div className={`text-2xl font-bold font-mono ${m.color}`}>
                            {m.value}
                        </div>
                        <div className="text-[11px] text-gray-500 mt-1">
                            {m.desc}
                        </div>
                    </div>
                ))}
            </div>
        </motion.div>
    );
}
