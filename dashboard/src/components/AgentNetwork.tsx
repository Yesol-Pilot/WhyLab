"use client";

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BrainCircuit, Wrench, Eye, Zap, Activity, Microscope, Library, FileText, Share2, ShieldCheck, Database } from "lucide-react";

interface LogMessage {
    id: string;
    agent_id: string;
    message: string;
    timestamp: string;
}

interface AgentNetworkProps {
    logs: LogMessage[];
}

// Define the comprehensive agent map for "The Grand Lab"
const AGENT_CONFIG: any = {
    // 1. Strategic Core (Center)
    "Architect": { icon: BrainCircuit, color: "text-purple-400", bg: "bg-purple-500/20", border: "border-purple-500/50", x: 50, y: 35, scale: 1.2 },
    "Director": { icon: Share2, color: "text-indigo-400", bg: "bg-indigo-500/20", x: 50, y: 15, scale: 1.1 },

    // 2. Research Team (Left)
    "HypothesisGen": { icon: Zap, color: "text-yellow-400", bg: "bg-yellow-500/20", x: 20, y: 30 },
    "LiteratureReview": { icon: Library, color: "text-blue-400", bg: "bg-blue-500/20", x: 15, y: 50 },
    "DataMiner": { icon: Database, color: "text-cyan-400", bg: "bg-cyan-500/20", x: 25, y: 65 },

    // 3. The Lab (Right) - Execution
    "ExperimentRunner": { icon: Microscope, color: "text-green-400", bg: "bg-green-500/20", x: 80, y: 30 },
    "DataAnalyst": { icon: Activity, color: "text-emerald-400", bg: "bg-emerald-500/20", x: 85, y: 50 },
    "FeatureEngineer": { icon: Wrench, color: "text-teal-400", bg: "bg-teal-500/20", x: 75, y: 65 },

    // 4. Review Board (Bottom)
    "Critic": { icon: Eye, color: "text-pink-400", bg: "bg-pink-500/20", x: 40, y: 85 },
    "EthicsBoard": { icon: ShieldCheck, color: "text-red-400", bg: "bg-red-500/20", x: 60, y: 85 },
    "Publisher": { icon: FileText, color: "text-slate-300", bg: "bg-slate-500/20", x: 50, y: 95 },

    // Legacy support mapping
    "Theorist": "HypothesisGen",
    "Engineer": "ExperimentRunner",
};

export default function AgentNetwork({ logs }: AgentNetworkProps) {
    const [activeAgents, setActiveAgents] = useState<Set<string>>(new Set());
    const [particles, setParticles] = useState<{ id: string; from: string; to: string }[]>([]);
    const lastLogIdRef = useRef<string | null>(null);

    // Simulation: Generate background chatter to make it look busy
    useEffect(() => {
        const interval = setInterval(() => {
            if (Math.random() > 0.3) {
                const agents = Object.keys(AGENT_CONFIG).filter(k => !["Theorist", "Engineer"].includes(k));
                const randomAgent = agents[Math.floor(Math.random() * agents.length)];

                // Randomly activate
                setActiveAgents(prev => {
                    const next = new Set(prev);
                    next.add(randomAgent);
                    setTimeout(() => setActiveAgents(curr => {
                        const up = new Set(curr);
                        up.delete(randomAgent);
                        return up;
                    }), 1500);
                    return next;
                });

                // Random particle
                const target = agents[Math.floor(Math.random() * agents.length)];
                if (randomAgent !== target) {
                    setParticles(prev => [...prev, { id: `rnd-${Date.now()}-${Math.random()}`, from: randomAgent, to: target }]);
                }
            }
        }, 800);
        return () => clearInterval(interval);
    }, []);

    // Real Log Handling
    useEffect(() => {
        if (logs.length === 0) return;

        const latestLog = logs[logs.length - 1];
        if (latestLog.id === lastLogIdRef.current) return;
        lastLogIdRef.current = latestLog.id;

        // Map legacy names to new scale names
        let agent = latestLog.agent_id;
        if (AGENT_CONFIG[agent] === undefined && AGENT_CONFIG["Theorist"] && agent === "Theorist") agent = "HypothesisGen";
        if (AGENT_CONFIG[agent] === undefined && AGENT_CONFIG["Engineer"] && agent === "Engineer") agent = "ExperimentRunner";

        setActiveAgents((prev) => {
            const next = new Set(prev);
            next.add(agent);
            setTimeout(() => {
                setActiveAgents((current) => {
                    const updated = new Set(current);
                    updated.delete(agent);
                    return updated;
                });
            }, 2000);
            return next;
        });

        // Spawn Logic Particle
        let target = "";
        if (agent === "HypothesisGen") target = "LiteratureReview";
        else if (agent === "LiteratureReview") target = "ExperimentRunner";
        else if (agent === "ExperimentRunner") target = "DataAnalyst";
        else if (agent === "DataAnalyst") target = "Critic";
        else if (agent === "Critic") target = "EthicsBoard";
        else if (agent === "EthicsBoard") target = "Architect";
        else if (agent === "Architect") target = "Director";

        if (target) {
            setParticles((prev) => [
                ...prev,
                { id: `p-${Date.now()}`, from: agent, to: target },
            ]);
        }
    }, [logs]);

    return (
        <div className="w-full h-[600px] relative bg-slate-900/50 rounded-2xl border border-white/5 backdrop-blur-sm overflow-hidden flex items-center justify-center">
            {/* Background Grid */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:40px_40px]" />

            {/* Central Hub Glow */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] bg-brand-500/5 rounded-full blur-3xl animate-pulse" />

            {/* Agents Container */}
            <div className="relative w-full h-full max-w-[800px] mx-auto">
                {Object.entries(AGENT_CONFIG).map(([name, config]: any) => {
                    if (typeof config === 'string') return null; // Skip mapping entries
                    const Icon = config.icon;
                    return (
                        <AgentNode
                            key={name}
                            type={name}
                            icon={<Icon />}
                            x={`${config.x}%`}
                            y={`${config.y}%`}
                            active={activeAgents.has(name)}
                            color={config.color}
                            bgColor={config.bg}
                            border={config.border}
                            scale={config.scale}
                        />
                    );
                })}

                {/* Interaction Particles */}
                <AnimatePresence>
                    {particles.map((p) => (
                        <Particle key={p.id} from={p.from} to={p.to} onComplete={() => setParticles(prev => prev.filter(item => item.id !== p.id))} />
                    ))}
                </AnimatePresence>
            </div>

            {/* Status Overlay */}
            <div className="absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/80 border border-white/10 text-xs text-slate-400 z-50">
                <Activity className="w-3.5 h-3.5 text-green-400 animate-pulse" />
                <span>HIVE MIND: EXPANDED PROTOCOL (11 NODE)</span>
            </div>
        </div>
    );
}

function AgentNode({ type, icon, x, y, active, color, bgColor, border, scale = 1 }: any) {
    return (
        <motion.div
            className={`absolute flex flex-col items-center gap-2 ${active ? 'z-30' : 'z-20'} transition-all duration-300`}
            style={{ top: y, left: x, translateX: "-50%", translateY: "-50%", scale: active ? scale * 1.1 : scale }}
        >
            <div className={`w-16 h-16 rounded-xl flex items-center justify-center border ${active ? 'border-white shadow-[0_0_20px_rgba(255,255,255,0.2)]' : (border || 'border-white/10')} ${bgColor} backdrop-blur-md transition-all`}>
                <div className={`w-8 h-8 ${color}`}>{icon}</div>
            </div>
            <div className="text-center bg-black/40 px-2 py-0.5 rounded backdrop-blur-sm">
                <div className={`font-bold text-[10px] uppercase tracking-wider ${active ? 'text-white' : 'text-slate-500'}`}>{type}</div>
            </div>
        </motion.div>
    );
}

function Particle({ from, to, onComplete }: any) {
    if (!AGENT_CONFIG[from] || !AGENT_CONFIG[to]) return null;

    // Convert % coordinates to approximate pixels for 800x600 container
    // This is rough estimation for visualisation
    const fromX = (AGENT_CONFIG[from].x / 100) * 800;
    const fromY = (AGENT_CONFIG[from].y / 100) * 600;
    const toX = (AGENT_CONFIG[to].x / 100) * 800;
    const toY = (AGENT_CONFIG[to].y / 100) * 600;

    return (
        <motion.div
            initial={{ left: fromX, top: fromY, opacity: 1, scale: 0.5 }}
            animate={{ left: toX, top: toY, opacity: 0, scale: 1.5 }}
            transition={{ duration: 1.2, ease: "linear" }}
            onAnimationComplete={onComplete}
            className="absolute w-2 h-2 rounded-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)] z-10 pointer-events-none"
        />
    );
}
