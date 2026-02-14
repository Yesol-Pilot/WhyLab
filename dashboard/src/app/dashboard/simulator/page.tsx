"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import PolicySimulator from "@/components/PolicySimulator";
import { BarChart3 } from "lucide-react";

function SimulatorContent() {
    const searchParams = useSearchParams();
    const scenario = searchParams.get("scenario") || "A";

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                    <BarChart3 className="w-7 h-7 text-brand-400" />
                    정책 시뮬레이터
                </h1>
                <p className="text-sm text-slate-400 mt-1">
                    정책 강도를 조절하여 예상 수익과 리스크를 실시간으로 시뮬레이션합니다.
                </p>
            </div>

            <div className="h-[700px] w-full">
                <PolicySimulator
                    baseLimit={scenario === "A" ? 1000 : 50}
                    baseDefaultRate={scenario === "A" ? 0.02 : 0.4}
                />
            </div>
        </div>
    );
}

export default function SimulatorPage() {
    return (
        <Suspense fallback={
            <div className="flex items-center justify-center h-[60vh]">
                <div className="w-10 h-10 border-4 border-brand-500 border-t-transparent rounded-full animate-spin" />
            </div>
        }>
            <SimulatorContent />
        </Suspense>
    );
}
