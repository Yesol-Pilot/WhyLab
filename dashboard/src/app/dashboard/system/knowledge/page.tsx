"use client";

import { useEffect, useState, useRef, useCallback } from 'react';
import { RefreshCw, ZoomIn, ZoomOut, Maximize2 } from "lucide-react";

interface GraphNode {
    id: string;
    label: string;
    group: string;
    x?: number;
    y?: number;
    vx?: number;
    vy?: number;
}

interface GraphLink {
    source: string;
    target: string;
    label: string;
}

interface GraphData {
    nodes: GraphNode[];
    links: GraphLink[];
}

const GROUP_COLORS: Record<string, string> = {
    Treatment: "#8b5cf6",   // 보라
    Outcome: "#10b981",     // 초록
    Confounder: "#f59e0b",  // 노랑
    Unknown: "#6b7280",     // 회색
};

const RELATION_STYLES: Record<string, { color: string; dash: number[] }> = {
    increases: { color: "#10b981", dash: [] },
    affects: { color: "#3b82f6", dash: [] },
    correlates: { color: "#f59e0b", dash: [6, 3] },
    hypothesis: { color: "#ef4444", dash: [4, 4] },
    moderates: { color: "#8b5cf6", dash: [8, 4] },
};

export default function KnowledgeGraphPage() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [graphData, setGraphData] = useState<GraphData | null>(null);
    const [nodes, setNodes] = useState<GraphNode[]>([]);
    const [hoveredNode, setHoveredNode] = useState<string | null>(null);
    const [stats, setStats] = useState({ nodes: 0, edges: 0 });
    const [zoom, setZoom] = useState(1);
    const animFrameRef = useRef<number>(0);

    const fetchGraph = useCallback(async () => {
        try {
            const res = await fetch('http://localhost:4001/system/graph');
            const data: GraphData = await res.json();
            setGraphData(data);
            setStats({ nodes: data.nodes.length, edges: data.links.length });

            // Force-Directed 초기 위치 설정
            const centerX = 400;
            const centerY = 300;
            const initialized = data.nodes.map((node, i) => ({
                ...node,
                x: centerX + Math.cos((2 * Math.PI * i) / data.nodes.length) * 180,
                y: centerY + Math.sin((2 * Math.PI * i) / data.nodes.length) * 140,
                vx: 0,
                vy: 0,
            }));
            setNodes(initialized);
        } catch (error) {
            console.error("Graph fetch error:", error);
        }
    }, []);

    useEffect(() => {
        fetchGraph();
    }, [fetchGraph]);

    // Force-Directed 시뮬레이션
    useEffect(() => {
        if (!nodes.length || !graphData) return;

        let iteration = 0;
        const maxIterations = 150;

        const simulate = () => {
            if (iteration >= maxIterations) return;

            const updated = [...nodes];
            const damping = 0.9;
            const repulsion = 3000;
            const attraction = 0.005;
            const centerForce = 0.01;
            const centerX = 400;
            const centerY = 300;

            // 반발력 (모든 노드 쌍)
            for (let i = 0; i < updated.length; i++) {
                for (let j = i + 1; j < updated.length; j++) {
                    const dx = (updated[i].x || 0) - (updated[j].x || 0);
                    const dy = (updated[i].y || 0) - (updated[j].y || 0);
                    const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
                    const force = repulsion / (dist * dist);
                    const fx = (dx / dist) * force;
                    const fy = (dy / dist) * force;
                    updated[i].vx = (updated[i].vx || 0) + fx;
                    updated[i].vy = (updated[i].vy || 0) + fy;
                    updated[j].vx = (updated[j].vx || 0) - fx;
                    updated[j].vy = (updated[j].vy || 0) - fy;
                }
            }

            // 인력 (연결된 노드)
            for (const link of graphData.links) {
                const srcNode = updated.find(n => n.id === link.source);
                const tgtNode = updated.find(n => n.id === link.target);
                if (!srcNode || !tgtNode) continue;
                const dx = (tgtNode.x || 0) - (srcNode.x || 0);
                const dy = (tgtNode.y || 0) - (srcNode.y || 0);
                const dist = Math.sqrt(dx * dx + dy * dy);
                const force = dist * attraction;
                srcNode.vx = (srcNode.vx || 0) + (dx / dist) * force;
                srcNode.vy = (srcNode.vy || 0) + (dy / dist) * force;
                tgtNode.vx = (tgtNode.vx || 0) - (dx / dist) * force;
                tgtNode.vy = (tgtNode.vy || 0) - (dy / dist) * force;
            }

            // 중심 인력 + 속도 적용
            for (const node of updated) {
                node.vx = ((node.vx || 0) + (centerX - (node.x || 0)) * centerForce) * damping;
                node.vy = ((node.vy || 0) + (centerY - (node.y || 0)) * centerForce) * damping;
                node.x = (node.x || 0) + (node.vx || 0);
                node.y = (node.y || 0) + (node.vy || 0);
                // 경계 제한
                node.x = Math.max(60, Math.min(740, node.x));
                node.y = Math.max(60, Math.min(540, node.y));
            }

            setNodes([...updated]);
            iteration++;
            animFrameRef.current = requestAnimationFrame(simulate);
        };

        animFrameRef.current = requestAnimationFrame(simulate);
        return () => cancelAnimationFrame(animFrameRef.current);
    }, [graphData]);

    // Canvas 렌더링
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !graphData || !nodes.length) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        canvas.width = 800 * dpr;
        canvas.height = 600 * dpr;
        ctx.scale(dpr * zoom, dpr * zoom);

        // 배경
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, 800 / zoom, 600 / zoom);

        // 그리드
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 0.5;
        for (let x = 0; x < 800; x += 40) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, 600);
            ctx.stroke();
        }
        for (let y = 0; y < 600; y += 40) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(800, y);
            ctx.stroke();
        }

        // 엣지 그리기
        for (const link of graphData.links) {
            const src = nodes.find(n => n.id === link.source);
            const tgt = nodes.find(n => n.id === link.target);
            if (!src || !tgt) continue;

            const style = RELATION_STYLES[link.label] || RELATION_STYLES.correlates;
            ctx.strokeStyle = style.color;
            ctx.lineWidth = 1.5;
            ctx.setLineDash(style.dash);

            ctx.beginPath();
            ctx.moveTo(src.x || 0, src.y || 0);
            ctx.lineTo(tgt.x || 0, tgt.y || 0);
            ctx.stroke();
            ctx.setLineDash([]);

            // 화살표
            const angle = Math.atan2((tgt.y || 0) - (src.y || 0), (tgt.x || 0) - (src.x || 0));
            const arrowLen = 10;
            const arrowX = (tgt.x || 0) - Math.cos(angle) * 28;
            const arrowY = (tgt.y || 0) - Math.sin(angle) * 28;
            ctx.fillStyle = style.color;
            ctx.beginPath();
            ctx.moveTo(arrowX, arrowY);
            ctx.lineTo(arrowX - arrowLen * Math.cos(angle - 0.4), arrowY - arrowLen * Math.sin(angle - 0.4));
            ctx.lineTo(arrowX - arrowLen * Math.cos(angle + 0.4), arrowY - arrowLen * Math.sin(angle + 0.4));
            ctx.closePath();
            ctx.fill();

            // 엣지 라벨
            const midX = ((src.x || 0) + (tgt.x || 0)) / 2;
            const midY = ((src.y || 0) + (tgt.y || 0)) / 2;
            ctx.fillStyle = '#94a3b8';
            ctx.font = '9px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(link.label, midX, midY - 5);
        }

        // 노드 그리기
        for (const node of nodes) {
            const color = GROUP_COLORS[node.group] || GROUP_COLORS.Unknown;
            const isHovered = hoveredNode === node.id;
            const radius = isHovered ? 24 : 20;

            // 글로우 효과
            if (isHovered) {
                ctx.shadowColor = color;
                ctx.shadowBlur = 20;
            }

            // 노드 원
            ctx.beginPath();
            ctx.arc(node.x || 0, node.y || 0, radius, 0, Math.PI * 2);
            ctx.fillStyle = color + '30';
            ctx.fill();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;

            // 노드 라벨
            ctx.fillStyle = '#e2e8f0';
            ctx.font = `${isHovered ? 'bold ' : ''}11px sans-serif`;
            ctx.textAlign = 'center';

            // 긴 텍스트 줄바꿈
            const words = node.label.split(' ');
            if (words.length > 2) {
                const line1 = words.slice(0, Math.ceil(words.length / 2)).join(' ');
                const line2 = words.slice(Math.ceil(words.length / 2)).join(' ');
                ctx.fillText(line1, node.x || 0, (node.y || 0) + radius + 14);
                ctx.fillText(line2, node.x || 0, (node.y || 0) + radius + 26);
            } else {
                ctx.fillText(node.label, node.x || 0, (node.y || 0) + radius + 14);
            }

            // 카테고리 태그
            ctx.fillStyle = color;
            ctx.font = 'bold 8px monospace';
            ctx.fillText(node.group.toUpperCase(), node.x || 0, (node.y || 0) - radius - 6);
        }
    }, [nodes, graphData, hoveredNode, zoom]);

    // 마우스 호버
    const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (800 / rect.width);
        const y = (e.clientY - rect.top) * (600 / rect.height);

        const found = nodes.find(n => {
            const dx = (n.x || 0) - x;
            const dy = (n.y || 0) - y;
            return Math.sqrt(dx * dx + dy * dy) < 25;
        });
        setHoveredNode(found?.id || null);
    }, [nodes]);

    return (
        <div className="p-8 space-y-6 bg-slate-50 min-h-screen">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight text-slate-900">Knowledge Graph</h1>
                    <p className="text-slate-500 text-sm mt-1">Causal knowledge network accumulated by agents</p>
                </div>
                <div className="flex items-center gap-2">
                    <button onClick={() => setZoom(z => Math.max(0.5, z - 0.1))} className="p-2 bg-white border rounded-lg hover:bg-slate-50">
                        <ZoomOut className="w-4 h-4" />
                    </button>
                    <span className="text-sm text-slate-500 font-mono w-12 text-center">{Math.round(zoom * 100)}%</span>
                    <button onClick={() => setZoom(z => Math.min(2, z + 0.1))} className="p-2 bg-white border rounded-lg hover:bg-slate-50">
                        <ZoomIn className="w-4 h-4" />
                    </button>
                    <button onClick={() => setZoom(1)} className="p-2 bg-white border rounded-lg hover:bg-slate-50">
                        <Maximize2 className="w-4 h-4" />
                    </button>
                    <button onClick={fetchGraph} className="px-4 py-2 bg-gradient-to-r from-purple-600 to-blue-600 text-white text-sm rounded-lg flex items-center gap-2 hover:from-purple-700 hover:to-blue-700">
                        <RefreshCw className="w-4 h-4" /> Refresh
                    </button>
                </div>
            </div>

            {/* 범례 + 통계 */}
            <div className="flex gap-6 items-center">
                <div className="flex gap-4">
                    {Object.entries(GROUP_COLORS).filter(([k]) => k !== 'Unknown').map(([group, color]) => (
                        <div key={group} className="flex items-center gap-1.5">
                            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                            <span className="text-xs text-slate-600">{group}</span>
                        </div>
                    ))}
                </div>
                <div className="border-l pl-4 flex gap-4">
                    {Object.entries(RELATION_STYLES).map(([rel, style]) => (
                        <div key={rel} className="flex items-center gap-1.5">
                            <div className="w-6 h-0.5" style={{
                                backgroundColor: style.color,
                                borderTop: style.dash.length ? `2px dashed ${style.color}` : `2px solid ${style.color}`,
                            }} />
                            <span className="text-xs text-slate-600">{rel}</span>
                        </div>
                    ))}
                </div>
                <div className="ml-auto flex gap-3 text-xs text-slate-500 font-mono">
                    <span>{stats.nodes} nodes</span>
                    <span>{stats.edges} edges</span>
                </div>
            </div>

            {/* Canvas */}
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <canvas
                    ref={canvasRef}
                    width={800}
                    height={600}
                    style={{ width: '100%', height: '600px', cursor: hoveredNode ? 'pointer' : 'default' }}
                    onMouseMove={handleMouseMove}
                />
            </div>

            {/* 호버 정보 */}
            {hoveredNode && (
                <div className="fixed bottom-8 right-8 bg-slate-900 text-white px-4 py-3 rounded-lg shadow-xl text-sm">
                    <div className="font-bold text-purple-400">{hoveredNode}</div>
                    <div className="text-slate-400 text-xs mt-1">
                        Connections: {graphData?.links.filter(l => l.source === hoveredNode || l.target === hoveredNode).length || 0} edges
                    </div>
                </div>
            )}
        </div>
    );
}
