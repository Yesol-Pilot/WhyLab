"use client";

import { useEffect, useState } from "react";
import FairnessPanel from "@/components/FairnessPanel";
import { AlertCircle, Loader2 } from "lucide-react";
import Link from "next/link";

export default function FairnessPage() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [results, setResults] = useState<any[]>([]);

    useEffect(() => {
        const sessionId = localStorage.getItem("whylab_session_id");
        if (!sessionId) return;

        const columnsStr = localStorage.getItem("whylab_columns");
        if (columnsStr) {
            const columns = JSON.parse(columnsStr);
            const treat = columns.find((c: string) => ["treatment", "treat", "t"].includes(c.toLowerCase())) || columns[0];
            const outcome = columns.find((c: string) => ["outcome", "y", "re78", "target"].includes(c.toLowerCase())) || columns[columns.length - 1];

            // 민감 속성 자동 감지
            const sensitive = columns.filter((c: string) =>
                ["age", "sex", "gender", "race", "black", "hispanic", "married", "nodegree"].includes(c.toLowerCase())
            );

            const confs = columns.filter((c: string) => c !== treat && c !== outcome && !sensitive.includes(c));

            if (sensitive.length === 0) {
                setError("No sensitive attributes (race, gender, etc.) found in data.");
                return;
            }

            fetchAnalysis(sessionId, {
                treatment: treat,
                outcome: outcome,
                sensitive_attrs: sensitive,
                confounders: confs
            });
        }
    }, []);

    const fetchAnalysis = async (sessionId: string, cfg: any) => {
        setLoading(true);
        setError(null);
        try {
            const res = await fetch("/api/analysis/fairness", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: sessionId,
                    treatment: cfg.treatment,
                    outcome: cfg.outcome,
                    sensitive_attrs: cfg.sensitive_attrs,
                    confounders: cfg.confounders
                })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Analysis failed");
            }

            const json = await res.json();
            setResults(json.result);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    if (results.length === 0 && !loading && !error) {
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
        <div className="space-y-8 pb-10">
            <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent mb-2">
                    Fairness Audit
                </h1>
                <p className="text-slate-400">
                    CATE 기반의 알고리즘 공정성 감사 결과입니다.
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
                    <p className="text-slate-400">Auditing Fairness...</p>
                    <p className="text-xs text-slate-500 mt-2">Calculating Causal Parity & Disparate Impact</p>
                </div>
            ) : (
                <div className="space-y-6">
                    {results.map((res: any, idx: number) => (
                        <div key={idx}>
                            <h2 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
                                <span className="bg-slate-700 text-xs px-2 py-1 rounded">Attribute</span>
                                {res.attribute}
                            </h2>
                            <FairnessPanel data={res} />
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
