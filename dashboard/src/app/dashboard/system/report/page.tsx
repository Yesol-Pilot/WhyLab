"use client";

import { useEffect, useState, useCallback } from "react";
import { FileText, RefreshCw, BookOpen, Download } from "lucide-react";

interface Section {
    id: string;
    title: string;
    content: string;
}

interface ReportStats {
    total_concepts: number;
    total_edges: number;
    total_hypotheses: number;
    total_agents: number;
    causal_relations: number;
}

interface Report {
    meta: {
        title: string;
        subtitle: string;
        generated_at: string;
        version: string;
    };
    sections: Section[];
    stats: ReportStats;
}

export default function ReportPage() {
    const [report, setReport] = useState<Report | null>(null);
    const [activeSection, setActiveSection] = useState("abstract");
    const [loading, setLoading] = useState(false);

    const fetchReport = useCallback(async () => {
        setLoading(true);
        try {
            const res = await fetch("http://localhost:4001/system/report");
            const data = await res.json();
            setReport(data);
        } catch (error) {
            console.error("Report fetch error:", error);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { fetchReport(); }, [fetchReport]);

    if (!report) {
        return (
            <div className="flex items-center justify-center h-screen bg-slate-50">
                <div className="text-center text-slate-400">
                    <FileText className="w-12 h-12 mx-auto mb-4 opacity-30 animate-pulse" />
                    <p>Generating report...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="flex h-screen bg-slate-50 overflow-hidden">
            {/* 좌측: 목차 네비게이션 */}
            <aside className="w-64 bg-white border-r p-6 flex flex-col">
                <div className="flex items-center gap-2 mb-6">
                    <BookOpen className="w-5 h-5 text-indigo-600" />
                    <span className="font-bold text-sm text-slate-900">Contents</span>
                </div>

                <nav className="space-y-1 flex-1">
                    {report.sections.map((sec) => (
                        <button
                            key={sec.id}
                            onClick={() => setActiveSection(sec.id)}
                            className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${activeSection === sec.id
                                ? "bg-indigo-50 text-indigo-700 font-semibold"
                                : "text-slate-600 hover:bg-slate-50"
                                }`}
                        >
                            {sec.title}
                        </button>
                    ))}
                </nav>

                {/* 통계 */}
                <div className="mt-auto pt-4 border-t space-y-2 text-xs text-slate-500">
                    <div className="flex justify-between">
                        <span>Concept Nodes</span>
                        <span className="font-bold text-slate-700">{report.stats.total_concepts}</span>
                    </div>
                    <div className="flex justify-between">
                        <span>Causal Relations</span>
                        <span className="font-bold text-slate-700">{report.stats.causal_relations}</span>
                    </div>
                    <div className="flex justify-between">
                        <span>Agents</span>
                        <span className="font-bold text-slate-700">{report.stats.total_agents}</span>
                    </div>
                    <div className="flex justify-between">
                        <span>Hypotheses</span>
                        <span className="font-bold text-slate-700">{report.stats.total_hypotheses}</span>
                    </div>
                    <div className="text-center text-[10px] text-slate-300 mt-3">
                        {report.meta.version}
                    </div>
                </div>
            </aside>

            {/* 우측: 보고서 본문 */}
            <main className="flex-1 overflow-y-auto">
                {/* 헤더 바 */}
                <div className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b px-8 py-4 flex items-center justify-between">
                    <div>
                        <h1 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                            <FileText className="w-5 h-5 text-indigo-600" />
                            {report.meta.title}
                        </h1>
                        <p className="text-xs text-slate-400">
                            {report.meta.subtitle} · Generated: {new Date(report.meta.generated_at).toLocaleString("en-US")}
                        </p>
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={fetchReport}
                            disabled={loading}
                            className="p-2 border rounded-lg hover:bg-slate-50 disabled:opacity-50"
                        >
                            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
                        </button>
                    </div>
                </div>

                {/* 논문 본문 */}
                <div className="max-w-3xl mx-auto px-8 py-10">
                    {report.sections.map((sec) => (
                        <section
                            key={sec.id}
                            id={sec.id}
                            className={`mb-12 transition-opacity duration-300 ${activeSection === sec.id ? "opacity-100" : "opacity-60"
                                }`}
                            onClick={() => setActiveSection(sec.id)}
                        >
                            <h2 className={`text-xl font-bold mb-4 pb-2 border-b-2 ${sec.id === "abstract"
                                ? "text-indigo-700 border-indigo-200"
                                : sec.id === "references"
                                    ? "text-slate-500 border-slate-200"
                                    : "text-slate-900 border-slate-100"
                                }`}>
                                {sec.title}
                            </h2>
                            <div className={`leading-7 text-slate-700 whitespace-pre-wrap ${sec.id === "abstract"
                                ? "bg-indigo-50 rounded-lg p-6 text-sm italic border-l-4 border-indigo-400"
                                : sec.id === "references"
                                    ? "text-sm text-slate-500"
                                    : ""
                                }`}>
                                {renderContent(sec.content)}
                            </div>
                        </section>
                    ))}
                </div>
            </main>
        </div>
    );
}

function renderContent(content: string) {
    // 간단한 마크다운 렌더링: **bold**, ### 소제목, - 리스트
    const lines = content.split("\n");
    return lines.map((line, i) => {
        // 소제목
        if (line.startsWith("### ")) {
            return (
                <h3 key={i} className="text-lg font-semibold text-slate-800 mt-6 mb-3 not-italic">
                    {line.replace("### ", "")}
                </h3>
            );
        }
        // 리스트
        if (line.startsWith("- ")) {
            const text = line.slice(2);
            return (
                <div key={i} className="flex gap-2 ml-4 my-1 not-italic">
                    <span className="text-indigo-400 mt-1">•</span>
                    <span>{renderBold(text)}</span>
                </div>
            );
        }
        // 번호 리스트
        if (/^\d+\.\s/.test(line)) {
            const num = line.match(/^(\d+)\./)?.[1];
            const text = line.replace(/^\d+\.\s/, "");
            return (
                <div key={i} className="flex gap-2 ml-4 my-1 not-italic">
                    <span className="text-indigo-500 font-semibold w-5 shrink-0">{num}.</span>
                    <span>{renderBold(text)}</span>
                </div>
            );
        }
        // 빈줄
        if (line.trim() === "") {
            return <div key={i} className="h-3" />;
        }
        // 일반 텍스트
        return <p key={i} className="my-1">{renderBold(line)}</p>;
    });
}

function renderBold(text: string) {
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**")) {
            return <strong key={i} className="text-slate-900 font-semibold">{part.slice(2, -2)}</strong>;
        }
        return <span key={i}>{part}</span>;
    });
}
