"use client";

import { useEffect, useRef, useState } from "react";
import { Terminal, Clock, Activity } from "lucide-react";

interface LogMessage {
    id: string;
    agent_id: string;
    message: string;
    timestamp: string;
}

interface LiveLogFeedProps {
    logs: LogMessage[];
}

export default function LiveLogFeed({ logs }: LiveLogFeedProps) {
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    const getAgentColor = (agent: string) => {
        switch (agent) {
            case "Theorist": return "text-brand-400";
            case "Engineer": return "text-accent-cyan";
            case "Critic": return "text-accent-pink";
            case "System": return "text-slate-400";
            default: return "text-white";
        }
    };

    return (
        <div className="flex flex-col h-[500px] bg-slate-950/80 rounded-2xl border border-white/10 overflow-hidden backdrop-blur-md">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-white/5 bg-slate-900/50">
                <div className="flex items-center gap-2 text-slate-300 font-mono text-sm">
                    <Terminal className="w-4 h-4" />
                    <span>Research Log Stream</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-500">
                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                    LIVE
                </div>
            </div>

            {/* Log Stream */}
            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto p-4 space-y-2 font-mono text-sm scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent"
            >
                {logs.length === 0 ? (
                    <div className="text-slate-600 italic text-center mt-20">Waiting for agent activity...</div>
                ) : (
                    logs.map((log) => (
                        <div key={log.id} className="flex gap-3 hover:bg-white/5 p-1 rounded transition-colors group">
                            <div className="text-slate-600 whitespace-nowrap opacity-50 text-xs mt-0.5">
                                {new Date(log.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                            </div>
                            <div className="flex-1 break-words">
                                <span className={`font-bold mr-2 ${getAgentColor(log.agent_id)}`}>
                                    [{log.agent_id}]
                                </span>
                                <span className="text-slate-300 group-hover:text-white transition-colors">
                                    {log.message}
                                </span>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
