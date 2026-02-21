"use client";

import { useEffect, useState } from "react";
import CausalDiscovery from "@/components/CausalDiscovery";
import { DAGNode, DAGEdge } from "@/types";
import { AlertCircle, Loader2 } from "lucide-react";
import Link from "next/link";

export default function CausalGraphPage() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [data, setData] = useState<{ nodes: DAGNode[], edges: DAGEdge[], stabilityScores: any } | null>(null);

    useEffect(() => {
        const sessionId = localStorage.getItem("whylab_session_id");
        if (!sessionId) return;

        fetchDiscovery(sessionId);
    }, []);

    const fetchDiscovery = async (sessionId: string) => {
        setLoading(true);
        setError(null);
        try {
            const res = await fetch("/api/analysis/discovery", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId })
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
                    Causal Structure Discovery
                </h1>
                <p className="text-slate-400">
                    MAC(Multi-Agent Causal) Discovery를 통해 발견된 인과 그래프입니다.
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
                    <p className="text-slate-400">Dicovering causal structure...</p>
                    <p className="text-xs text-slate-500 mt-2">Running PC, GES, LiNGAM algorithms</p>
                </div>
            ) : data ? (
                <CausalDiscovery
                    nodes={data.nodes}
                    edges={data.edges}
                    stabilityScores={data.stabilityScores}
                />
            ) : null}
        </div>
    );
}
