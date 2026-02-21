'use client';

import React from 'react';
import { motion } from 'framer-motion';

// --- Types ---
interface Evidence {
    claim: string;
    type: string;
    strength: number;
    source: string;
}

interface LLMDebateData {
    llm_active: boolean;
    model: string;
    advocate_argument: string;
    critic_argument: string;
    judge_verdict: string;
}

interface DebateData {
    verdict: "CAUSAL" | "NOT_CAUSAL" | "UNCERTAIN" | "UNKNOWN";
    confidence: number;
    pro_score: number;
    con_score: number;
    rounds: number;
    recommendation: string;
    pro_evidence: Evidence[];
    con_evidence: Evidence[];
    llm_debate?: LLMDebateData;
}

interface DebateVerdictProps {
    data: {
        debate?: DebateData;
    };
}

// --- Component ---
export default function DebateVerdict({ data }: DebateVerdictProps) {
    const debate = data.debate;

    if (!debate) {
        return (
            <div className="p-6 bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700">
                <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">⚖️ Causal Verdict</h3>
                <p className="text-gray-500 dark:text-gray-400">No data available.</p>
            </div>
        );
    }

    // 판결 색상 및 아이콘
    const getVerdictStyle = (verdict: string) => {
        switch (verdict) {
            case 'CAUSAL':
                return { color: 'text-green-600 dark:text-green-400', bg: 'bg-green-50 dark:bg-green-900/20', icon: '✅', label: 'Causal Relationship Confirmed' };
            case 'NOT_CAUSAL':
                return { color: 'text-red-600 dark:text-red-400', bg: 'bg-red-50 dark:bg-red-900/20', icon: '❌', label: 'Causal Relationship Rejected' };
            case 'UNCERTAIN':
                return { color: 'text-amber-600 dark:text-amber-400', bg: 'bg-amber-50 dark:bg-amber-900/20', icon: '⚠️', label: 'Verdict Pending (Uncertain)' };
            default:
                return { color: 'text-gray-600 dark:text-gray-400', bg: 'bg-gray-50 dark:bg-gray-800', icon: '❓', label: 'Unable to Determine (Unknown)' };
        }
    };

    const style = getVerdictStyle(debate.verdict);
    const totalScore = debate.pro_score + debate.con_score || 1;
    const proPercent = (debate.pro_score / totalScore) * 100;

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-6 bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 mb-6"
        >
            {/* 헤더 */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    <span className="text-3xl">{style.icon}</span>
                    <div>
                        <h2 className={`text-xl font-bold ${style.color}`}>{style.label}</h2>
                        <p className="text-sm text-gray-500 dark:text-gray-400">
                            Confidence {(debate.confidence * 100).toFixed(1)}% • {debate.rounds} rounds of consensus
                        </p>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    {debate.llm_debate?.llm_active && (
                        <div className="px-2 py-0.5 rounded-full text-xs font-medium bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300">
                            🤖 {debate.llm_debate.model}
                        </div>
                    )}
                    <div className={`px-3 py-1 rounded-full text-xs font-medium ${style.bg} ${style.color}`}>
                        AI Multi-Agent Debate
                    </div>
                </div>
            </div>

            {/* 게이지 바 */}
            <div className="relative h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden mb-6">
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${proPercent}%` }}
                    transition={{ duration: 1, ease: "easeOut" }}
                    className="absolute h-full bg-gradient-to-r from-green-400 to-green-600"
                />
                {/* 중앙선 */}
                <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-white/50 z-10" />
            </div>

            <div className="flex justify-between text-sm font-medium mb-6 px-1">
                <div className="text-green-600 dark:text-green-400">
                    Advocate {debate.pro_score.toFixed(1)}
                </div>
                <div className="text-red-600 dark:text-red-400">
                    Critic {debate.con_score.toFixed(1)}
                </div>
            </div>

            {/* 증거 카드 그리드 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* 옹호 측 증거 */}
                <div className="space-y-3">
                    <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-green-500" />
                        Key Advocate Arguments (Top 3)
                    </h3>
                    {debate.pro_evidence.slice(0, 3).map((e, i) => (
                        <EvidenceCard key={i} evidence={e} type="pro" />
                    ))}
                    {debate.pro_evidence.length === 0 && <EmptyCard />}
                </div>

                {/* 비판 측 증거 */}
                <div className="space-y-3">
                    <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-red-500" />
                        Key Critic Arguments (Top 3)
                    </h3>
                    {debate.con_evidence.slice(0, 3).map((e, i) => (
                        <EvidenceCard key={i} evidence={e} type="con" />
                    ))}
                    {debate.con_evidence.length === 0 && <EmptyCard />}
                </div>
            </div>

            {/* LLM 자연어 토론 (Phase 9) */}
            {debate.llm_debate && (debate.llm_debate.advocate_argument || debate.llm_debate.critic_argument) && (
                <div className="mt-6 space-y-3">
                    <h3 className="text-sm font-semibold text-gray-600 dark:text-gray-400 flex items-center gap-2">
                        <span className="text-base">🎙️</span>
                        LLM Agent Debate
                        {!debate.llm_debate.llm_active && (
                            <span className="text-xs text-gray-400">(Rule-based Fallback)</span>
                        )}
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {debate.llm_debate.advocate_argument && (
                            <LLMArgumentCard
                                title="Growth Hacker (Advocate)"
                                content={debate.llm_debate.advocate_argument}
                                variant="pro"
                            />
                        )}
                        {debate.llm_debate.critic_argument && (
                            <LLMArgumentCard
                                title="Risk Manager (Critic)"
                                content={debate.llm_debate.critic_argument}
                                variant="con"
                            />
                        )}
                    </div>
                    {debate.llm_debate.judge_verdict && (
                        <div className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-200 dark:border-slate-700">
                            <div className="flex items-start gap-3">
                                <span className="text-xl">⚖️</span>
                                <div>
                                    <h4 className="text-sm font-bold text-slate-800 dark:text-slate-200 mb-1">
                                        Product Owner Final Verdict
                                    </h4>
                                    <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed whitespace-pre-line">
                                        {debate.llm_debate.judge_verdict}
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* 추천 사항 */}
            <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-100 dark:border-blue-800">
                <div className="flex items-start gap-3">
                    <span className="text-xl">💡</span>
                    <div>
                        <h4 className="text-sm font-bold text-blue-800 dark:text-blue-300 mb-1">AI Recommendation</h4>
                        <p className="text-sm text-blue-700 dark:text-blue-400 leading-relaxed">
                            {debate.recommendation}
                        </p>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

function EvidenceCard({ evidence, type }: { evidence: Evidence; type: 'pro' | 'con' }) {
    const isPro = type === 'pro';
    const bgColor = isPro ? 'bg-green-50 dark:bg-green-900/10' : 'bg-red-50 dark:bg-red-900/10';
    const borderColor = isPro ? 'border-green-100 dark:border-green-800' : 'border-red-100 dark:border-red-800';
    const textColor = isPro ? 'text-green-800 dark:text-green-300' : 'text-red-800 dark:text-red-300';

    return (
        <div className={`p-3 rounded-lg border ${bgColor} ${borderColor}`}>
            <div className="flex justify-between items-start mb-1">
                <span className={`text-xs font-bold uppercase tracking-wider ${textColor} opacity-70`}>
                    {evidence.type}
                </span>
                <span className={`text-xs font-bold ${textColor}`}>
                    +{evidence.strength.toFixed(2)}
                </span>
            </div>
            <p className={`text-sm font-medium ${textColor}`}>
                {evidence.claim}
            </p>
        </div>
    );
}

function EmptyCard() {
    return (
        <div className="p-4 rounded-lg border border-dashed border-gray-300 text-center text-gray-400 text-sm">
            N/A
        </div>
    );
}

function LLMArgumentCard({ title, content, variant }: { title: string; content: string; variant: 'pro' | 'con' }) {
    const isPro = variant === 'pro';
    const bgColor = isPro ? 'bg-green-50/70 dark:bg-green-900/10' : 'bg-red-50/70 dark:bg-red-900/10';
    const borderColor = isPro ? 'border-green-200 dark:border-green-800' : 'border-red-200 dark:border-red-800';
    const titleColor = isPro ? 'text-green-700 dark:text-green-300' : 'text-red-700 dark:text-red-300';
    const icon = isPro ? '📗' : '📕';

    return (
        <div className={`p-4 rounded-xl border ${bgColor} ${borderColor}`}>
            <h4 className={`text-sm font-bold ${titleColor} mb-2 flex items-center gap-1.5`}>
                <span>{icon}</span> {title}
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-line">
                {content}
            </p>
        </div>
    );
}
