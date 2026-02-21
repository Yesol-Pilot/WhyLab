"use client";

import { useEffect, useState } from "react";
import DoseResponseChart from "@/components/DoseResponseChart";
import { AlertCircle, Loader2 } from "lucide-react";
import Link from "next/link";

export default function DoseResponsePage() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [data, setData] = useState<any>(null);
    const [config, setConfig] = useState({ treatment: "", outcome: "", confounders: [] as string[] });

    useEffect(() => {
        const sessionId = localStorage.getItem("whylab_session_id");
        if (!sessionId) return;

        // 컬럼 정보 가져오기 (from localStorage or API)
        const columnsStr = localStorage.getItem("whylab_columns");
        if (columnsStr) {
            const columns = JSON.parse(columnsStr);
            // 간단한 휴리스틱으로 변수 자동 선택 (데모용)
            // 실제로는 UI에서 선택하게 해야 함
            const treat = columns.find((c: string) => ["treatment", "treat", "t"].includes(c.toLowerCase())) || columns[0];
            const outcome = columns.find((c: string) => ["outcome", "y", "re78", "target"].includes(c.toLowerCase())) || columns[columns.length - 1];
            const confs = columns.filter((c: string) => c !== treat && c !== outcome);

            const newConfig = { treatment: treat, outcome: outcome, confounders: confs };
            setConfig(newConfig);
            fetchAnalysis(sessionId, newConfig);
        }
    }, []);

    const fetchAnalysis = async (sessionId: string, cfg: any) => {
        setLoading(true);
        setError(null);
        try {
            const res = await fetch("/api/analysis/dose-response", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: sessionId,
                    treatment: cfg.treatment,
                    outcome: cfg.outcome,
                    confounders: cfg.confounders
                })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Analysis failed");
            }

            const json = await res.json();
            setData(json.result);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    if (!data && !loading && !error) {
        return (
            <div className="flex flex-col items-center justify-center h-[50vh] text-slate-400">
                <AlertCircle className="w-12 h-12 mb-4 text-slate-600" />
                <p className="text-lg">No data available.</p>
                <Link href="/dashboard/upload" className="text-brand-400 hover:underline mt-2">
                    Upload data to start analysis
                </Link>
            </div>
        );
    }

    return (
        <div className="space-y-8 h-[calc(100vh-100px)]">
            <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent mb-2">
                    Dose-Response Analysis
                </h1>
                <p className="text-slate-400">
                    Target: {config.treatment} → {config.outcome}
                </p>
            </div>

            {error && (
                <div className="bg-red-900/20 border border-red-800 text-red-300 p-4 rounded-lg flex items-center gap-3">
                    <AlertCircle className="w-5 h-5" />
                    {error}
                </div>
            )}

            {loading ? (
                <div className="flex flex-col items-center justify-center h-[400px]">
                    <Loader2 className="w-8 h-8 text-brand-500 animate-spin mb-4" />
                    <p className="text-slate-400">Estimating Dose-Response Curve...</p>
                    <p className="text-xs text-slate-500 mt-2">Running GPS estimation & Kernel Regression</p>
                </div>
            ) : data ? (
                <div className="h-[500px]">
                    <DoseResponseChart
                        data={data}
                        treatmentName={config.treatment}
                        outcomeName={config.outcome}
                    />
                </div>
            ) : null}
        </div>
    );
}
