"use client";

import { CausalAnalysisResult } from "@/types";
import { motion } from "framer-motion";
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from "recharts";
import { Shield, ShieldCheck, ShieldX } from "lucide-react";

interface Props {
    data: CausalAnalysisResult;
}

export default function SensitivityReport({ data }: Props) {
    const { sensitivity } = data;
    if (!sensitivity) return null;

    const isPass = sensitivity.status === "Pass";

    const placeboData = [
        {
            name: "Original ATE",
            value: Math.abs(data.ate.value),
            fill: "#60a5fa",
        },
        {
            name: "Placebo Effect",
            value: Math.abs(sensitivity.placebo_test?.mean_effect ?? 0),
            fill: sensitivity.placebo_test?.status === "Pass" ? "#34d399" : "#f87171",
        },
    ];

    const stabilityData = [
        {
            name: "Stability",
            value: (sensitivity.random_common_cause?.stability ?? 0) * 100,
        },
    ];

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-6"
        >
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                    {isPass ? (
                        <ShieldCheck className="w-6 h-6 text-emerald-400" />
                    ) : (
                        <ShieldX className="w-6 h-6 text-red-400" />
                    )}
                    <h3 className="text-lg font-semibold text-white">
                        Robustness Check
                    </h3>
                </div>
                <span
                    className={`px-3 py-1 rounded-full text-sm font-bold ${isPass
                            ? "bg-emerald-500/20 text-emerald-400"
                            : "bg-red-500/20 text-red-400"
                        }`}
                >
                    {sensitivity.status}
                </span>
            </div>

            {/* Placebo Test Chart */}
            <div className="mb-6">
                <p className="text-sm text-gray-400 mb-2">
                    Placebo Treatment Test
                    <span className="ml-2 text-gray-500">
                        (p = {sensitivity.placebo_test?.p_value?.toFixed(3) ?? "N/A"})
                    </span>
                </p>
                <ResponsiveContainer width="100%" height={120}>
                    <BarChart data={placeboData} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                        <XAxis type="number" tick={{ fill: "#9ca3af", fontSize: 12 }} />
                        <YAxis
                            type="category"
                            dataKey="name"
                            tick={{ fill: "#9ca3af", fontSize: 12 }}
                            width={110}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: "#1f2937",
                                border: "1px solid #374151",
                                borderRadius: "8px",
                            }}
                        />
                        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                            {placeboData.map((entry, index) => (
                                <Cell key={index} fill={entry.fill} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
                <p className="text-xs text-gray-500 mt-1">
                    Placebo Effect should be near zero if causal relationship is genuine.
                </p>
            </div>

            {/* Stability Gauge */}
            <div>
                <p className="text-sm text-gray-400 mb-2">
                    Random Common Cause Stability
                </p>
                <div className="relative h-4 bg-gray-700 rounded-full overflow-hidden">
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{
                            width: `${stabilityData[0].value}%`,
                        }}
                        transition={{ duration: 1, ease: "easeOut" }}
                        className={`h-full rounded-full ${stabilityData[0].value > 90
                                ? "bg-emerald-400"
                                : stabilityData[0].value > 70
                                    ? "bg-yellow-400"
                                    : "bg-red-400"
                            }`}
                    />
                </div>
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>0%</span>
                    <span className="text-white font-bold">
                        {stabilityData[0].value.toFixed(1)}%
                    </span>
                    <span>100%</span>
                </div>
            </div>
        </motion.div>
    );
}
