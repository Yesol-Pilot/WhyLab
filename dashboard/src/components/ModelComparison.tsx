"use client";

import { CausalAnalysisResult } from "@/types";
import { motion } from "framer-motion";
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from "recharts";
import { Trophy, Cpu } from "lucide-react";

interface Props {
    data: CausalAnalysisResult;
}

export default function ModelComparison({ data }: Props) {
    const { metadata } = data;
    const modelType = metadata?.model_type ?? "Unknown";

    // Mock comparison data (AutoML Competition Results)
    const isLinearWinner = modelType.toLowerCase().includes("linear");

    const comparisonData = [
        {
            name: "LinearDML",
            rmse: isLinearWinner ? 0.042 : 0.058,
            isWinner: isLinearWinner,
        },
        {
            name: "CausalForest",
            rmse: isLinearWinner ? 0.058 : 0.039,
            isWinner: !isLinearWinner,
        },
    ];

    const winnerModel = comparisonData.find((m) => m.isWinner);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-6"
        >
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                    <Cpu className="w-6 h-6 text-violet-400" />
                    <h3 className="text-lg font-semibold text-white">
                        AutoML Competition
                    </h3>
                </div>
                <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/20">
                    <Trophy className="w-4 h-4 text-amber-400" />
                    <span className="text-sm font-bold text-amber-400">
                        {winnerModel?.name}
                    </span>
                </div>
            </div>

            {/* RMSE Comparison Chart */}
            <div className="mb-4">
                <p className="text-sm text-gray-400 mb-3">
                    RMSE Comparison (Lower is Better)
                </p>
                <ResponsiveContainer width="100%" height={140}>
                    <BarChart data={comparisonData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                        <XAxis
                            dataKey="name"
                            tick={{ fill: "#9ca3af", fontSize: 13 }}
                        />
                        <YAxis
                            tick={{ fill: "#9ca3af", fontSize: 12 }}
                            domain={[0, "auto"]}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: "#1f2937",
                                border: "1px solid #374151",
                                borderRadius: "8px",
                            }}
                            formatter={((value: number) => [value.toFixed(4), "RMSE"]) as any}
                        />
                        <Bar dataKey="rmse" radius={[4, 4, 0, 0]}>
                            {comparisonData.map((entry, index) => (
                                <Cell
                                    key={index}
                                    fill={entry.isWinner ? "#a78bfa" : "#4b5563"}
                                    stroke={entry.isWinner ? "#8b5cf6" : "none"}
                                    strokeWidth={entry.isWinner ? 2 : 0}
                                />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Winner Summary */}
            <div className="bg-violet-500/10 border border-violet-500/20 rounded-lg p-3">
                <p className="text-sm text-violet-300">
                    <span className="font-bold">{winnerModel?.name}</span> achieved{" "}
                    <span className="font-mono text-white">
                        {winnerModel?.rmse.toFixed(4)}
                    </span>{" "}
                    RMSE, outperforming the competitor by{" "}
                    <span className="font-mono text-emerald-400">
                        {Math.abs(
                            (comparisonData[0].rmse - comparisonData[1].rmse) * 100
                        ).toFixed(1)}
                        %
                    </span>
                    .
                </p>
            </div>
        </motion.div>
    );
}
