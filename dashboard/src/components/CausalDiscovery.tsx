"use client";

import React, { useMemo } from 'react';
import ReactFlow, {
    Handle,
    Position,
    type Node,
    type Edge,
    Background,
    Controls,
    MarkerType
} from 'reactflow';
import 'reactflow/dist/style.css';
import { clsx } from 'clsx';
import { DAGNode, DAGEdge } from '@/types';

// 커스텀 노드 스타일
const CustomNode = ({ data }: { data: { label: string; role: string } }) => {
    const roleColors = {
        treatment: "bg-brand-500 border-brand-300 shadow-[0_0_15px_rgba(139,92,246,0.5)]",
        outcome: "bg-accent-cyan border-cyan-300 shadow-[0_0_15px_rgba(34,211,238,0.5)]",
        confounder: "bg-slate-700 border-slate-500",
        mediator: "bg-slate-600 border-slate-400",
        other: "bg-slate-800 border-slate-600"
    };

    const colorClass = roleColors[data.role as keyof typeof roleColors] || roleColors.other;

    return (
        <div className={clsx(
            "px-4 py-2 rounded-lg border-2 text-white font-bold text-sm min-w-[100px] text-center transition-all hover:scale-105",
            colorClass
        )}>
            <Handle type="target" position={Position.Top} className="!bg-white !w-3 !h-3" />
            {data.label}
            <Handle type="source" position={Position.Bottom} className="!bg-white !w-3 !h-3" />
        </div>
    );
};

const nodeTypes = { custom: CustomNode };

interface CausalDiscoveryProps {
    nodes: DAGNode[];
    edges: DAGEdge[];
    stabilityScores?: Record<string, number>; // "source-target": score (0~1)
}

export default function CausalDiscovery({ nodes, edges, stabilityScores = {} }: CausalDiscoveryProps) {
    // 간단한 계층형 레이아웃 (역할 기반)
    const flowNodes: Node[] = useMemo(() => {
        const roleY = {
            confounder: 0,
            treatment: 150,
            mediator: 300,
            outcome: 450,
            other: 0
        };

        const roleCounts = { confounder: 0, treatment: 0, mediator: 0, outcome: 0, other: 0 };
        const roleWidths = { confounder: 600, treatment: 400, mediator: 400, outcome: 200, other: 800 };

        return nodes.map((node) => {
            const role = (node.role as keyof typeof roleY) || 'other';
            const count = roleCounts[role];
            roleCounts[role]++;

            // 중앙 정렬을 위한 X 좌표 계산
            // (간단히 순서대로 배치하되, 전체 폭을 고려)
            const xOffset = (count * 180) - (roleWidths[role] / 2) + 300;

            return {
                id: node.id,
                type: 'custom',
                position: { x: xOffset, y: roleY[role] },
                data: { label: node.label, role: role }
            };
        });
    }, [nodes]);

    const flowEdges: Edge[] = useMemo(() => {
        return edges.map((edge) => {
            const edgeId = `${edge.source}-${edge.target}`;
            const stability = stabilityScores[edgeId] ?? 1.0;

            // 안정성이 낮으면 투명하게, 높으면 진하게
            const opacity = 0.3 + (stability * 0.7);
            const width = 1 + (stability * 3);

            return {
                id: edgeId,
                source: edge.source,
                target: edge.target,
                type: 'smoothstep',
                animated: true,
                style: {
                    stroke: '#94a3b8',
                    strokeWidth: width,
                    opacity: opacity
                },
                label: stability < 1.0 ? `${(stability * 100).toFixed(0)}%` : undefined,
                labelStyle: { fill: '#cbd5e1', fontSize: 10 },
                labelBgStyle: { fill: '#1e293b', opacity: 0.8 },
                markerEnd: {
                    type: MarkerType.ArrowClosed,
                    color: '#94a3b8',
                },
            };
        });
    }, [edges, stabilityScores]);

    return (
        <div className="w-full h-[600px] glass-card !p-0 overflow-hidden relative">
            <div className="absolute top-4 left-4 z-10 bg-black/50 px-3 py-1 rounded text-xs text-slate-300">
                MAC Discovery Result
            </div>
            <div className="absolute bottom-4 left-4 z-10 flex flex-col gap-2 text-xs text-slate-400 bg-black/50 p-2 rounded">
                <div className="flex items-center gap-2">
                    <span className="w-3 h-3 bg-brand-500 rounded-sm"></span> Treatment
                </div>
                <div className="flex items-center gap-2">
                    <span className="w-3 h-3 bg-accent-cyan rounded-sm"></span> Outcome
                </div>
                <div className="flex items-center gap-2">
                    <span className="w-3 h-3 bg-slate-700 rounded-sm"></span> Confounder
                </div>
                <div className="mt-1 border-t border-slate-600 pt-1">
                    Edge thickness = Stability
                </div>
            </div>
            <ReactFlow
                nodes={flowNodes}
                edges={flowEdges}
                nodeTypes={nodeTypes}
                fitView
                proOptions={{ hideAttribution: true }}
            >
                <Background color="#334155" gap={20} size={1} />
                <Controls className="!bg-slate-800 !border-slate-700 !text-white" />
            </ReactFlow>
        </div>
    );
}
