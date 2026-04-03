"use client";

import React, { useMemo } from 'react';
import ReactFlow, {
    Handle,
    Position,
    type Node,
    type Edge,
    Background,
    Controls
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

interface CausalGraphProps {
    nodes: DAGNode[];
    edges: DAGEdge[];
}

export default function CausalGraph({ nodes, edges }: CausalGraphProps) {
    // DAG 데이터를 React Flow 형식으로 변환 및 자동 배치 (간략화)
    // 실제로는 dagre 등의 라이브러리로 자동 레이아웃을 잡아야 하지만, 
    // 여기서는 역할별로 Y축을 고정하여 간단히 배치합니다.

    const flowNodes: Node[] = useMemo(() => {
        const roleY = {
            confounder: 0,
            treatment: 150,
            mediator: 150,
            outcome: 300,
            other: 0
        };

        const roleCounts = { confounder: 0, treatment: 0, mediator: 0, outcome: 0, other: 0 };

        return nodes.map((node) => {
            const role = node.role as keyof typeof roleY;
            const count = roleCounts[role];
            roleCounts[role]++;

            // X축: 중앙을 기준으로 좌우로 펼침
            return {
                id: node.id,
                type: 'custom',
                position: { x: count * 150 + (role === 'treatment' ? 100 : 0), y: roleY[role] },
                data: { label: node.label, role: node.role }
            };
        });
    }, [nodes]);

    const flowEdges: Edge[] = useMemo(() => {
        return edges.map((edge) => ({
            id: `${edge.source}-${edge.target}`,
            source: edge.source,
            target: edge.target,
            animated: true,
            style: { stroke: '#94a3b8', strokeWidth: 2 },
        }));
    }, [edges]);

    return (
        <div className="w-full h-[400px] glass-card !p-0 overflow-hidden relative">
            <div className="absolute top-4 left-4 z-10 bg-black/50 px-3 py-1 rounded text-xs text-slate-300">
                Causal DAG
            </div>
            <ReactFlow
                nodes={flowNodes}
                edges={flowEdges}
                nodeTypes={nodeTypes}
                fitView
                proOptions={{ hideAttribution: true }}
            >
                <Background color="#aaa" gap={16} size={1} />
                <Controls className="!bg-slate-800 !border-slate-700 !text-white" />
            </ReactFlow>
        </div>
    );
}
