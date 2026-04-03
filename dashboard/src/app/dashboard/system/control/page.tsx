"use client";

import { useEffect, useState, useCallback } from "react";
import {
    Gauge, RefreshCw, Loader2, Activity, AlertTriangle, CheckCircle, XCircle,
    Beaker, FileText, CloudCog, ChevronDown, ChevronRight, Zap, Shield, Brain
} from "lucide-react";

// ─── 타입 정의 ───
interface DiagnosticCheck { name: string; status: "OK" | "WARNING" | "CRITICAL"; detail: string; }
interface DiagnosticResult {
    timestamp: string; health_score: number; checks: DiagnosticCheck[];
    warnings: string[]; recommendations: string[];
    total_checks: number; ok_count: number; warning_count: number; critical_count: number;
}
interface DGPTemplate {
    name: string; category: string; treatment: string; outcome: string;
    confounders: string[]; moderators: string[]; n_default: number; effect_type: string;
}
interface QualityMetrics {
    pehe_naive: number; jsd_pi_mean: number; overlap_coefficient: number;
    heterogeneity_ratio: number; quality_grade: string; treatment_rate: number;
    ate_true: number; cate_std: number; sample_size: number;
}
interface STEAMResult { dgp_name: string; n: number; ate_true: number; quality_metrics: QualityMetrics; }
interface PaperDraft {
    title: string; abstract: string;
    sections: Record<string, string>;
    statistics: { total_experiments: number; validated_hypotheses: number; rejected_hypotheses: number; methods_used: string[]; };
    metadata: { generated_at: string; version: string; };
}
interface SaaSReadiness {
    readiness_score: number; category_scores: Record<string, number>;
    strengths: string[]; gaps: string[]; recommendation: string;
}
interface MigrationItem { category: string; current: string; target: string; effort: string; notes: string; }
interface MigrationPlan {
    plan_version: string; total_items: number;
    phases: Record<string, MigrationItem[]>;
    estimated_timeline: Record<string, string>;
}

const API = "http://localhost:4001";

// ─── 유틸 ───
const statusIcon = (s: string) => {
    switch (s) {
        case "OK": return <CheckCircle className="w-5 h-5 text-emerald-500" />;
        case "WARNING": return <AlertTriangle className="w-5 h-5 text-amber-500" />;
        case "CRITICAL": return <XCircle className="w-5 h-5 text-red-500 animate-pulse" />;
        default: return null;
    }
};
const statusBg = (s: string) => {
    switch (s) {
        case "OK": return "bg-emerald-50/80 border-emerald-200";
        case "WARNING": return "bg-amber-50/80 border-amber-200";
        case "CRITICAL": return "bg-red-50/80 border-red-200 ring-1 ring-red-100";
        default: return "bg-slate-50 border-slate-200";
    }
};
const gradeColor = (g: string) => {
    switch (g) {
        case "S": return "bg-gradient-to-r from-yellow-400 to-amber-500 text-white";
        case "A": return "bg-emerald-500 text-white";
        case "B": return "bg-blue-500 text-white";
        case "C": return "bg-slate-400 text-white";
        default: return "bg-red-500 text-white";
    }
};
const scoreColor = (s: number) => {
    if (s >= 80) return "text-emerald-500";
    if (s >= 60) return "text-amber-500";
    return "text-red-500";
};
const effortBadge = (e: string) => {
    switch (e) {
        case "Low": return "bg-emerald-100 text-emerald-700 border-emerald-200";
        case "Medium": return "bg-amber-100 text-amber-700 border-amber-200";
        case "High": return "bg-red-100 text-red-700 border-red-200";
        default: return "bg-slate-100 text-slate-700 border-slate-200";
    }
};
const categoryLabel = (key: string) => {
    const map: Record<string, string> = {
        api_layer: "API Layer", db_abstraction: "DB Abstraction", auth: "Auth",
        multi_tenancy: "Multi-Tenancy", async_execution: "Async Execution",
        monitoring: "Monitoring", deployment: "Deployment", billing: "Billing",
    };
    return map[key] || key;
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 메인 페이지
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export default function SystemControlPage() {
    // 상태
    const [diagnostic, setDiagnostic] = useState<DiagnosticResult | null>(null);
    const [dgps, setDgps] = useState<Record<string, DGPTemplate>>({});
    const [dgpList, setDgpList] = useState<string[]>([]);
    const [selectedDgp, setSelectedDgp] = useState("");
    const [sampleSize, setSampleSize] = useState(3000);
    const [seed, setSeed] = useState(42);
    const [steamResult, setSteamResult] = useState<STEAMResult | null>(null);
    const [paper, setPaper] = useState<PaperDraft | null>(null);
    const [activeTab, setActiveTab] = useState("abstract");
    const [saas, setSaas] = useState<SaaSReadiness | null>(null);
    const [migration, setMigration] = useState<MigrationPlan | null>(null);
    const [expandedPhase, setExpandedPhase] = useState<string | null>(null);
    const [loading, setLoading] = useState<Record<string, boolean>>({});

    // API 호출 헬퍼
    const fetchData = useCallback(async (key: string, url: string, setter: (d: any) => void, method = "GET") => {
        setLoading(prev => ({ ...prev, [key]: true }));
        try {
            const res = await fetch(url, { method });
            if (res.ok) setter(await res.json());
        } catch (e) {
            console.error(`Fetch error [${key}]:`, e);
        } finally {
            setLoading(prev => ({ ...prev, [key]: false }));
        }
    }, []);

    // 초기 로드
    useEffect(() => {
        fetchData("diag", `${API}/system/architect/diagnose`, setDiagnostic);
        fetchData("dgps", `${API}/system/steam/dgps`, (d) => {
            setDgps(d.templates || {});
            setDgpList(d.available_dgps || []);
            if (d.available_dgps?.length) setSelectedDgp(d.available_dgps[0]);
        });
        fetchData("saas", `${API}/system/saas/readiness`, setSaas);
    }, [fetchData]);

    // 핸들러
    const handleRefreshDiag = () => fetchData("diag", `${API}/system/architect/diagnose`, setDiagnostic);
    const handleGenerate = () =>
        fetchData("steam", `${API}/system/steam/generate?dgp_name=${selectedDgp}&n=${sampleSize}&seed=${seed}`, setSteamResult, "POST");
    const handlePaper = () =>
        fetchData("paper", `${API}/system/paper/draft`, setPaper, "POST");
    const handleMigration = () => {
        if (!migration) fetchData("migration", `${API}/system/saas/migration-plan`, setMigration);
    };

    return (
        <div className="space-y-8">
            {/* ─── 헤더 ─── */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500 to-accent-pink flex items-center justify-center shadow-lg shadow-brand-500/20">
                            <Gauge className="w-5 h-5 text-white" />
                        </div>
                        System Control Panel
                    </h1>
                    <p className="text-sm text-slate-400 mt-1">Architect Agent Diagnostics & System Management</p>
                </div>
                {diagnostic && (
                    <div className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold border ${diagnostic.health_score >= 80 ? "bg-emerald-50 text-emerald-700 border-emerald-200" : diagnostic.health_score >= 60 ? "bg-amber-50 text-amber-700 border-amber-200" : "bg-red-50 text-red-700 border-red-200"}`}>
                        <Activity className="w-4 h-4" />
                        Health: {diagnostic.health_score}/100
                    </div>
                )}
            </div>

            {/* ━━━━ Section 1: System Health ━━━━ */}
            <section className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <Shield className="w-5 h-5 text-brand-500" />
                        <h2 className="text-lg font-bold text-slate-900">System Health</h2>
                    </div>
                    <button onClick={handleRefreshDiag} disabled={loading.diag}
                        className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-brand-600 bg-brand-50 rounded-lg hover:bg-brand-100 transition-colors disabled:opacity-50">
                        {loading.diag ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
                        Re-diagnose
                    </button>
                </div>
                <div className="p-6">
                    {loading.diag && !diagnostic ? (
                        <div className="flex items-center justify-center py-12">
                            <Loader2 className="w-8 h-8 animate-spin text-brand-400" />
                        </div>
                    ) : diagnostic ? (
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            {/* 게이지 */}
                            <div className="flex flex-col items-center justify-center">
                                <div className="relative w-40 h-40">
                                    <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
                                        <circle cx="60" cy="60" r="50" fill="none" stroke="#e2e8f0" strokeWidth="10" />
                                        <circle cx="60" cy="60" r="50" fill="none"
                                            stroke={diagnostic.health_score >= 80 ? "#10b981" : diagnostic.health_score >= 60 ? "#f59e0b" : "#ef4444"}
                                            strokeWidth="10" strokeLinecap="round"
                                            strokeDasharray={`${diagnostic.health_score * 3.14} 314`}
                                            className="transition-all duration-1000" />
                                    </svg>
                                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                                        <span className={`text-4xl font-black ${scoreColor(diagnostic.health_score)}`}>
                                            {diagnostic.health_score}
                                        </span>
                                        <span className="text-xs text-slate-400 mt-1">/ 100</span>
                                    </div>
                                </div>
                                <div className="flex gap-4 mt-4 text-xs text-slate-500">
                                    <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500" /> OK {diagnostic.ok_count}</span>
                                    <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-500" /> WARN {diagnostic.warning_count}</span>
                                    <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500" /> CRIT {diagnostic.critical_count}</span>
                                </div>
                            </div>

                            {/* 진단 카드 */}
                            <div className="lg:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-3">
                                {diagnostic.checks.map((c, i) => (
                                    <div key={i} className={`rounded-xl border p-4 ${statusBg(c.status)} transition-all hover:shadow-sm`}>
                                        <div className="flex items-center justify-between mb-2">
                                            <span className="text-sm font-semibold text-slate-800">{c.name}</span>
                                            {statusIcon(c.status)}
                                        </div>
                                        <p className="text-xs text-slate-600 leading-relaxed">{c.detail}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : null}

                    {/* 권장 사항 */}
                    {diagnostic && diagnostic.recommendations.length > 0 && (
                        <div className="mt-6 bg-slate-50 rounded-xl p-4 border border-slate-100">
                            <h3 className="text-sm font-semibold text-slate-700 mb-2">Recommendations</h3>
                            <ul className="space-y-1">
                                {diagnostic.recommendations.map((r, i) => (
                                    <li key={i} className="text-xs text-slate-600">{r}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            </section>

            {/* ━━━━ Section 2: STEAM Generator ━━━━ */}
            <section className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="px-6 py-4 border-b border-slate-100 flex items-center gap-3">
                    <Beaker className="w-5 h-5 text-accent-cyan" />
                    <h2 className="text-lg font-bold text-slate-900">STEAM Synthetic Data Generator</h2>
                </div>
                <div className="p-6">
                    {/* 컨트롤 */}
                    <div className="flex flex-wrap items-end gap-4 mb-6">
                        <div className="flex-1 min-w-[200px]">
                            <label className="block text-xs font-medium text-slate-500 mb-1">DGP Template</label>
                            <select value={selectedDgp} onChange={e => setSelectedDgp(e.target.value)}
                                className="w-full px-3 py-2.5 rounded-lg border border-slate-200 bg-white text-sm text-slate-800 focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none">
                                {dgpList.map(d => <option key={d} value={d}>{dgps[d]?.name || d}</option>)}
                            </select>
                        </div>
                        <div className="w-28">
                            <label className="block text-xs font-medium text-slate-500 mb-1">Sample Size</label>
                            <input type="number" value={sampleSize} onChange={e => setSampleSize(+e.target.value)}
                                className="w-full px-3 py-2.5 rounded-lg border border-slate-200 text-sm text-slate-800 focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none" />
                        </div>
                        <div className="w-20">
                            <label className="block text-xs font-medium text-slate-500 mb-1">Seed</label>
                            <input type="number" value={seed} onChange={e => setSeed(+e.target.value)}
                                className="w-full px-3 py-2.5 rounded-lg border border-slate-200 text-sm text-slate-800 focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none" />
                        </div>
                        <button onClick={handleGenerate} disabled={loading.steam || !selectedDgp}
                            className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-brand-600 to-brand-500 text-white text-sm font-semibold hover:from-brand-700 hover:to-brand-600 shadow-lg shadow-brand-500/20 disabled:opacity-50 transition-all flex items-center gap-2">
                            {loading.steam ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                            Generate
                        </button>
                    </div>

                    {/* DGP 상세 카드 */}
                    {selectedDgp && dgps[selectedDgp] && (
                        <div className="bg-slate-50 rounded-xl p-4 border border-slate-100 mb-6">
                            <div className="flex items-center gap-3 mb-3">
                                <span className="text-base font-bold text-slate-800">{dgps[selectedDgp].name}</span>
                                <span className="px-2 py-0.5 text-xs font-mono rounded-full bg-brand-100 text-brand-700 border border-brand-200">
                                    {dgps[selectedDgp].category}
                                </span>
                                <span className="px-2 py-0.5 text-xs font-mono rounded-full bg-slate-100 text-slate-600 border border-slate-200">
                                    {dgps[selectedDgp].effect_type}
                                </span>
                            </div>
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                                <div><span className="text-slate-400">Treatment:</span> <span className="font-mono text-slate-700">{dgps[selectedDgp].treatment}</span></div>
                                <div><span className="text-slate-400">Outcome:</span> <span className="font-mono text-slate-700">{dgps[selectedDgp].outcome}</span></div>
                                <div><span className="text-slate-400">Confounders:</span> <span className="font-mono text-slate-700">{dgps[selectedDgp].confounders.join(", ")}</span></div>
                                <div><span className="text-slate-400">Default n:</span> <span className="font-mono text-slate-700">{dgps[selectedDgp].n_default.toLocaleString()}</span></div>
                            </div>
                        </div>
                    )}

                    {/* 생성 결과 */}
                    {steamResult && (
                        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                            <MetricCard label="ATE (True)" value={steamResult.quality_metrics.ate_true.toFixed(3)} />
                            <MetricCard label="PEHE" value={steamResult.quality_metrics.pehe_naive.toFixed(4)} />
                            <MetricCard label="JSD π" value={steamResult.quality_metrics.jsd_pi_mean.toFixed(4)} />
                            <MetricCard label="Overlap" value={steamResult.quality_metrics.overlap_coefficient.toFixed(3)} />
                            <MetricCard label="Treatment Rate" value={`${(steamResult.quality_metrics.treatment_rate * 100).toFixed(1)}%`} />
                            <div className="bg-white rounded-xl border border-slate-200 p-4 flex flex-col items-center justify-center">
                                <span className="text-xs text-slate-400 mb-2">Quality Grade</span>
                                <span className={`text-2xl font-black px-4 py-1 rounded-lg ${gradeColor(steamResult.quality_metrics.quality_grade)}`}>
                                    {steamResult.quality_metrics.quality_grade}
                                </span>
                            </div>
                        </div>
                    )}
                </div>
            </section>

            {/* ━━━━ Section 3: Paper & Research ━━━━ */}
            <section className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <FileText className="w-5 h-5 text-accent-pink" />
                        <h2 className="text-lg font-bold text-slate-900">Paper & Research</h2>
                    </div>
                    <button onClick={handlePaper} disabled={loading.paper}
                        className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-accent-pink to-brand-500 rounded-lg hover:opacity-90 transition-all shadow-lg shadow-accent-pink/20 disabled:opacity-50">
                        {loading.paper ? <Loader2 className="w-4 h-4 animate-spin" /> : <Brain className="w-4 h-4" />}
                        Generate Paper Draft
                    </button>
                </div>
                <div className="p-6">
                    {!paper ? (
                        <div className="flex flex-col items-center justify-center py-12 text-slate-400">
                            <FileText className="w-12 h-12 mb-3 opacity-30" />
                            <p className="text-sm">&quot;Generate Paper Draft&quot; button will auto-generate an IMRAD-structured paper.</p>
                        </div>
                    ) : (
                        <div>
                            <div className="mb-4">
                                <h3 className="text-lg font-bold text-slate-900">{paper.title}</h3>
                                <p className="text-xs text-slate-400 mt-1">
                                    {paper.metadata.version} • Generated at {new Date(paper.metadata.generated_at).toLocaleString()}
                                </p>
                            </div>

                            {/* 통계 바 */}
                            <div className="grid grid-cols-3 gap-3 mb-5">
                                <div className="bg-brand-50 rounded-lg p-3 text-center border border-brand-100">
                                    <div className="text-xl font-bold text-brand-700">{paper.statistics.total_experiments}</div>
                                    <div className="text-xs text-brand-500">Total Experiments</div>
                                </div>
                                <div className="bg-emerald-50 rounded-lg p-3 text-center border border-emerald-100">
                                    <div className="text-xl font-bold text-emerald-700">{paper.statistics.validated_hypotheses}</div>
                                    <div className="text-xs text-emerald-500">Validated Hypotheses</div>
                                </div>
                                <div className="bg-slate-50 rounded-lg p-3 text-center border border-slate-100">
                                    <div className="text-xl font-bold text-slate-700">{paper.statistics.methods_used.length}</div>
                                    <div className="text-xs text-slate-500">Methods Used</div>
                                </div>
                            </div>

                            {/* 탭 */}
                            <div className="flex gap-1 mb-4 bg-slate-100 rounded-lg p-1">
                                {["abstract", "introduction", "methodology", "results", "discussion", "conclusion"].map(tab => (
                                    <button key={tab} onClick={() => setActiveTab(tab)}
                                        className={`flex-1 py-2 text-xs font-medium rounded-md transition-all capitalize ${activeTab === tab ? "bg-white text-brand-700 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}>
                                        {tab === "abstract" ? "Abstract" : tab.charAt(0).toUpperCase() + tab.slice(1)}
                                    </button>
                                ))}
                            </div>

                            {/* 콘텐츠 */}
                            <div className="bg-slate-50 rounded-xl p-5 border border-slate-100 min-h-[200px]">
                                <pre className="text-sm text-slate-700 whitespace-pre-wrap font-sans leading-relaxed">
                                    {activeTab === "abstract" ? paper.abstract : paper.sections[activeTab] || "No section data available."}
                                </pre>
                            </div>
                        </div>
                    )}
                </div>
            </section>

            {/* ━━━━ Section 4: SaaS Readiness ━━━━ */}
            <section className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="px-6 py-4 border-b border-slate-100 flex items-center gap-3">
                    <CloudCog className="w-5 h-5 text-blue-500" />
                    <h2 className="text-lg font-bold text-slate-900">SaaS Readiness</h2>
                </div>
                <div className="p-6">
                    {!saas ? (
                        <div className="flex items-center justify-center py-12">
                            <Loader2 className="w-8 h-8 animate-spin text-brand-400" />
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            {/* 전체 점수 */}
                            <div className="flex flex-col items-center justify-center">
                                <div className="relative w-40 h-40">
                                    <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
                                        <circle cx="60" cy="60" r="50" fill="none" stroke="#e2e8f0" strokeWidth="10" />
                                        <circle cx="60" cy="60" r="50" fill="none" stroke="#3b82f6"
                                            strokeWidth="10" strokeLinecap="round"
                                            strokeDasharray={`${saas.readiness_score * 3.14} 314`}
                                            className="transition-all duration-1000" />
                                    </svg>
                                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                                        <span className="text-4xl font-black text-blue-500">{saas.readiness_score}</span>
                                        <span className="text-xs text-slate-400 mt-1">%</span>
                                    </div>
                                </div>
                                <p className="text-xs text-slate-500 mt-3 text-center max-w-[220px]">{saas.recommendation}</p>
                            </div>

                            {/* 카테고리별 진행률 */}
                            <div className="lg:col-span-2 space-y-3">
                                {Object.entries(saas.category_scores)
                                    .sort(([, a], [, b]) => b - a)
                                    .map(([key, val]) => (
                                        <div key={key}>
                                            <div className="flex items-center justify-between mb-1">
                                                <span className="text-xs font-medium text-slate-600">{categoryLabel(key)}</span>
                                                <span className={`text-xs font-bold ${val >= 80 ? "text-emerald-600" : val >= 40 ? "text-amber-600" : val > 0 ? "text-red-600" : "text-slate-400"}`}>
                                                    {val}%
                                                </span>
                                            </div>
                                            <div className="bg-slate-100 rounded-full h-2.5 overflow-hidden">
                                                <div className={`h-full rounded-full transition-all duration-1000 ${val >= 80 ? "bg-emerald-500" : val >= 40 ? "bg-amber-400" : val > 0 ? "bg-red-400" : "bg-slate-200"}`}
                                                    style={{ width: `${val}%` }} />
                                            </div>
                                        </div>
                                    ))}
                            </div>
                        </div>
                    )}

                    {/* 강점/갭 */}
                    {saas && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                            <div className="bg-emerald-50/50 rounded-xl p-4 border border-emerald-100">
                                <h3 className="text-sm font-semibold text-emerald-700 mb-2">✅ Strengths</h3>
                                <ul className="space-y-1">{saas.strengths.map((s, i) => <li key={i} className="text-xs text-emerald-600">{s}</li>)}</ul>
                            </div>
                            <div className="bg-red-50/50 rounded-xl p-4 border border-red-100">
                                <h3 className="text-sm font-semibold text-red-700 mb-2">⚠️ Needs Improvement</h3>
                                <ul className="space-y-1">{saas.gaps.map((g, i) => <li key={i} className="text-xs text-red-600">{g}</li>)}</ul>
                            </div>
                        </div>
                    )}

                    {/* 마이그레이션 계획 (accordion) */}
                    <div className="mt-6">
                        <button onClick={handleMigration}
                            className="text-sm font-medium text-blue-600 hover:text-blue-700 flex items-center gap-1 mb-4">
                            {loading.migration ? <Loader2 className="w-4 h-4 animate-spin" /> : <ChevronDown className="w-4 h-4" />}
                            View Migration Plan
                        </button>

                        {migration && (
                            <div className="space-y-2">
                                {Object.entries(migration.phases).map(([phase, items]) => (
                                    <div key={phase} className="border border-slate-200 rounded-xl overflow-hidden">
                                        <button onClick={() => setExpandedPhase(expandedPhase === phase ? null : phase)}
                                            className="w-full flex items-center justify-between px-4 py-3 bg-slate-50 hover:bg-slate-100 transition-colors">
                                            <div className="flex items-center gap-3">
                                                {expandedPhase === phase ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronRight className="w-4 h-4 text-slate-400" />}
                                                <span className="text-sm font-semibold text-slate-800">{phase}</span>
                                                <span className="text-xs text-slate-400">{items.length} items</span>
                                            </div>
                                            <span className="text-xs text-blue-500 font-medium">{migration.estimated_timeline[phase]}</span>
                                        </button>
                                        {expandedPhase === phase && (
                                            <div className="p-4 space-y-3">
                                                {items.map((item, i) => (
                                                    <div key={i} className="flex items-start gap-4 text-xs">
                                                        <span className="font-semibold text-slate-700 min-w-[100px]">{item.category}</span>
                                                        <div className="flex-1">
                                                            <div className="flex items-center gap-2 mb-1">
                                                                <span className="text-slate-400 line-through">{item.current}</span>
                                                                <span className="text-slate-300">→</span>
                                                                <span className="text-slate-800 font-medium">{item.target}</span>
                                                                <span className={`px-1.5 py-0.5 text-[10px] rounded border ${effortBadge(item.effort)}`}>{item.effort}</span>
                                                            </div>
                                                            <p className="text-slate-400">{item.notes}</p>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </section>
        </div>
    );
}

// ─── 서브 컴포넌트 ───
function MetricCard({ label, value }: { label: string; value: string }) {
    return (
        <div className="bg-white rounded-xl border border-slate-200 p-4 text-center hover:shadow-md transition-shadow">
            <div className="text-xs text-slate-400 mb-2">{label}</div>
            <div className="text-lg font-bold text-slate-800 font-mono">{value}</div>
        </div>
    );
}
