"use client";

import { useState } from "react";
import { UploadCloud, CheckCircle, AlertCircle, FileText, Database } from "lucide-react";
import { useRouter } from "next/navigation";

export default function UploadPage() {
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [sessionData, setSessionData] = useState<any>(null);
    const router = useRouter();

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setError(null);
            setSessionData(null);
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setUploading(true);
        setError(null);

        const formData = new FormData();
        formData.append("file", file);

        try {
            // Next.js API Proxy -> FastAPI
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Upload failed");
            }

            const data = await res.json();
            setSessionData(data);

            // 세션 저장 (localStorage)
            localStorage.setItem("whylab_session_id", data.session_id);
            localStorage.setItem("whylab_columns", JSON.stringify(data.columns));

        } catch (err: any) {
            setError(err.message);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="space-y-8 h-full max-w-4xl mx-auto">
            <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent mb-2">
                    Data Upload
                </h1>
                <p className="text-slate-400">
                    Upload CSV data for analysis. Data is retained only for the duration of the session.
                </p>
            </div>

            <div className="glass-card p-8 flex flex-col items-center justify-center min-h-[300px] border-dashed border-2 border-slate-700 hover:border-brand-500 transition-colors">
                {!sessionData ? (
                    <>
                        <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mb-4">
                            <UploadCloud className="w-8 h-8 text-brand-400" />
                        </div>
                        <input
                            type="file"
                            accept=".csv"
                            onChange={handleFileChange}
                            className="hidden"
                            id="file-upload"
                        />
                        <label
                            htmlFor="file-upload"
                            className="cursor-pointer bg-brand-600 hover:bg-brand-500 text-white px-6 py-2 rounded-lg font-medium transition-colors mb-2"
                        >
                            Select CSV File
                        </label>
                        <p className="text-sm text-slate-500">
                            {file ? file.name : "or drag and drop here"}
                        </p>
                        {error && (
                            <div className="mt-4 flex items-center gap-2 text-red-400 text-sm bg-red-900/20 px-4 py-2 rounded">
                                <AlertCircle className="w-4 h-4" />
                                {error}
                            </div>
                        )}
                        {file && (
                            <button
                                onClick={handleUpload}
                                disabled={uploading}
                                className="mt-6 w-full max-w-xs bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg disabled:opacity-50"
                            >
                                {uploading ? "Uploading..." : "Start Upload"}
                            </button>
                        )}
                    </>
                ) : (
                    <div className="w-full">
                        <div className="flex items-center gap-3 text-green-400 mb-6 justify-center">
                            <CheckCircle className="w-6 h-6" />
                            <span className="text-lg font-bold">Upload Successful!</span>
                        </div>

                        <div className="grid grid-cols-2 gap-4 mb-6 text-sm">
                            <div className="bg-slate-800/50 p-4 rounded-lg flex items-center gap-3">
                                <FileText className="w-5 h-5 text-slate-400" />
                                <div>
                                    <div className="text-slate-500">Filename</div>
                                    <div className="text-white font-mono">{sessionData.filename}</div>
                                </div>
                            </div>
                            <div className="bg-slate-800/50 p-4 rounded-lg flex items-center gap-3">
                                <Database className="w-5 h-5 text-slate-400" />
                                <div>
                                    <div className="text-slate-500">Rows</div>
                                    <div className="text-white font-mono">{sessionData.rows.toLocaleString()}</div>
                                </div>
                            </div>
                        </div>

                        <div className="bg-slate-900/50 p-4 rounded-lg mb-6 overflow-hidden">
                            <div className="text-xs text-slate-500 mb-2 uppercase font-bold">Data Preview</div>
                            <div className="overflow-x-auto">
                                <table className="w-full text-xs text-left text-slate-300">
                                    <thead className="text-slate-500 bg-slate-800 uppercase">
                                        <tr>
                                            {sessionData.columns.slice(0, 5).map((col: string) => (
                                                <th key={col} className="px-3 py-2">{col}</th>
                                            ))}
                                            {sessionData.columns.length > 5 && <th className="px-3 py-2">...</th>}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {sessionData.preview.map((row: any, i: number) => (
                                            <tr key={i} className="border-b border-slate-800 hover:bg-slate-800/30">
                                                {sessionData.columns.slice(0, 5).map((col: string) => (
                                                    <td key={col} className="px-3 py-2">{String(row[col]).slice(0, 20)}</td>
                                                ))}
                                                {sessionData.columns.length > 5 && <td className="px-3 py-2">...</td>}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <div className="flex justify-center gap-4">
                            <button
                                onClick={() => router.push('/dashboard/causal-graph')}
                                className="bg-brand-600 hover:bg-brand-500 text-white px-6 py-2 rounded-lg"
                            >
                                Go to Discovery
                            </button>
                            <button
                                onClick={() => router.push('/dashboard/dose-response')}
                                className="bg-slate-700 hover:bg-slate-600 text-white px-6 py-2 rounded-lg"
                            >
                                Go to Analysis
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
