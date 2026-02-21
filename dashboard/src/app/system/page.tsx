"use client";

import { useEffect, useState } from 'react';
import { Activity, Terminal, Shield, Brain, Hammer, Gavel, Cpu } from "lucide-react";

interface Agent {
    id: string;
    name: string;
    role: string;
    status: string;
    config: any;
    created_at: string;
}

interface Log {
    id: number;
    agent_id: string;
    level: string;
    message: string;
    created_at: string;
}

export default function ControlRoomPage() {
    const [agents, setAgents] = useState<Agent[]>([]);
    const [logs, setLogs] = useState<Log[]>([]);
    const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

    const fetchSystemStatus = async () => {
        try {
            const agentRes = await fetch('http://localhost:8000/system/agents');
            const agentData = await agentRes.json();
            setAgents(agentData);

            const logRes = await fetch('http://localhost:8000/system/logs?limit=100');
            const logData = await logRes.json();
            setLogs(logData);

            setLastUpdated(new Date());
        } catch (error) {
            console.error("System fetch error:", error);
        }
    };

    useEffect(() => {
        fetchSystemStatus();
        const interval = setInterval(fetchSystemStatus, 3000); // Poll every 3s
        return () => clearInterval(interval);
    }, []);

    const getIcon = (role: string) => {
        switch (role) {
            case "Theorist": return <Brain className="w-6 h-6 text-purple-500" />;
            case "Engineer": return <Hammer className="w-6 h-6 text-blue-500" />;
            case "Critic": return <Gavel className="w-6 h-6 text-red-500" />;
            case "Coordinator": return <Shield className="w-6 h-6 text-yellow-500" />;
            default: return <Cpu className="w-6 h-6 text-gray-500" />;
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case "WORKING": return "bg-green-100 text-green-700 border-green-200";
            case "ERROR": return "bg-red-100 text-red-700 border-red-200";
            default: return "bg-gray-100 text-gray-700 border-gray-200";
        }
    };

    return (
        <div className="p-8 space-y-6 bg-slate-50 min-h-screen">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight text-slate-900">Control Room</h1>
                    <p className="text-slate-500 text-sm mt-1">Observer Mode Active • Last Updated: {lastUpdated.toLocaleTimeString()}</p>
                </div>
                <div className="px-4 py-2 text-sm bg-black text-white rounded-full flex items-center gap-2 border border-slate-800">
                    <Activity className="w-4 h-4 text-green-400" /> System Online
                </div>
            </div>

            {/* Agent Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {agents.map((agent) => (
                    <div key={agent.id} className="bg-white rounded-lg border border-slate-200 shadow-sm p-4 hover:shadow-md transition-shadow">
                        <div className="flex items-center justify-between mb-4">
                            <span className="text-sm font-medium text-slate-500">{agent.role}</span>
                            {getIcon(agent.role)}
                        </div>
                        <div className="text-2xl font-bold text-slate-900">{agent.name}</div>
                        <div className="mt-4 flex items-center justify-between">
                            <span className="text-xs text-slate-400 font-mono">{agent.id}</span>
                            <span className={`px-2 py-1 text-xs rounded-full border ${getStatusColor(agent.status)}`}>
                                {agent.status}
                            </span>
                        </div>
                    </div>
                ))}
            </div>

            {/* System Logs */}
            <div className="bg-white rounded-lg border border-slate-200 shadow-sm">
                <div className="p-4 border-b border-slate-100 flex items-center gap-2">
                    <Terminal className="w-5 h-5 text-slate-500" />
                    <h2 className="font-semibold text-slate-900">System Activity Log</h2>
                </div>
                <div className="p-4 bg-slate-900 rounded-b-lg h-[400px] overflow-y-auto font-mono text-sm">
                    <div className="space-y-2">
                        {logs.length === 0 && <span className="text-slate-500">Waiting for system events...</span>}
                        {logs.map((log) => (
                            <div key={log.id} className="flex gap-4 text-slate-300">
                                <span className="text-slate-500 min-w-[150px] shrink-0">
                                    {new Date(log.created_at).toLocaleString()}
                                </span>
                                <span className={`min-w-[80px] shrink-0 font-bold ${log.level === 'ERROR' ? 'text-red-400' :
                                        log.level === 'WARNING' ? 'text-yellow-400' : 'text-blue-400'
                                    }`}>
                                    [{log.level}]
                                </span>
                                <span className="text-slate-400 min-w-[120px] shrink-0">
                                    {log.agent_id || 'SYSTEM'}
                                </span>
                                <span className="text-slate-100 break-all">
                                    {log.message}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
