"use client";

import "@xyflow/react/dist/style.css";

import { useEffect, useMemo, useRef } from "react";
import { motion } from "framer-motion";
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import type { VisualNode, VisualPayload } from "@/lib/types";
import { buildMindMapLayout } from "@/lib/visuals";

interface MindMapProps {
  animationKey: string;
  visual: VisualPayload;
  onSelectNode: (node: VisualNode | null) => void;
}

type TutorNodeData = Record<string, unknown> & {
  color?: string;
  description?: string;
  label: string;
};

function TutorNode({ data, selected }: NodeProps<Node<TutorNodeData>>) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.88 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`min-w-[12rem] max-w-[15rem] rounded-[1.35rem] border px-4 py-3 shadow-lg ${
        selected
          ? "border-primary bg-primary/10"
          : "border-border bg-background/90 dark:bg-background/80"
      }`}
      style={{
        borderColor: selected ? "var(--primary)" : data.color,
      }}
    >
      <Handle type="target" position={Position.Left} className="!bg-primary" />
      <p className="text-sm font-semibold">{data.label}</p>
      {data.description ? (
        <p className="mt-1 text-xs text-muted-foreground">{data.description}</p>
      ) : null}
      <Handle type="source" position={Position.Right} className="!bg-primary" />
    </motion.div>
  );
}

const nodeTypes = { tutor: TutorNode };

export function MindMap({ animationKey, visual, onSelectNode }: MindMapProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<TutorNodeData>>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const timeoutsRef = useRef<number[]>([]);
  const flowRef = useRef<{
    fitView: (options?: { duration?: number; padding?: number }) => void;
  } | null>(null);

  const laidOutNodes = useMemo(() => buildMindMapLayout(visual), [visual]);

  useEffect(() => {
    setNodes([]);
    setEdges([]);
    onSelectNode(null);

    timeoutsRef.current.forEach((timeout) => window.clearTimeout(timeout));
    timeoutsRef.current = [];

    const mappedNodes = laidOutNodes.map<Node<TutorNodeData>>((node) => ({
      id: node.id,
      type: "tutor",
      position: node.position ?? { x: 0, y: 0 },
      data: {
        label: node.label,
        description: node.description,
        color: node.color,
      },
      sourcePosition: (node.position?.x ?? 0) >= 0 ? Position.Right : Position.Left,
      targetPosition: (node.position?.x ?? 0) >= 0 ? Position.Left : Position.Right,
    }));

    const mappedEdges = visual.edges.map<Edge>((edge, index) => ({
      id: edge.id ?? `edge-${index}`,
      source: edge.source,
      target: edge.target,
      label: edge.label,
      animated: edge.animated ?? true,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: "#0f766e",
      },
      style: {
        stroke: "#0f766e",
        strokeWidth: 2,
      },
    }));

    mappedNodes.forEach((node, index) => {
      const timeout = window.setTimeout(() => {
        setNodes((current) => [...current, node]);
      }, index * 220);
      timeoutsRef.current.push(timeout);
    });

    mappedEdges.forEach((edge, index) => {
      const timeout = window.setTimeout(() => {
        setEdges((current) => [...current, edge]);
        flowRef.current?.fitView({ padding: 0.24, duration: 500 });
      }, mappedNodes.length * 220 + index * 140);
      timeoutsRef.current.push(timeout);
    });

    return () => {
      timeoutsRef.current.forEach((timeout) => window.clearTimeout(timeout));
      timeoutsRef.current = [];
    };
  }, [animationKey, laidOutNodes, onSelectNode, setEdges, setNodes, visual.edges]);

  useEffect(() => {
    setNodes((current) =>
      current.map((node) => {
        const updatedNode = laidOutNodes.find((entry) => entry.id === node.id);
        if (!updatedNode) {
          return node;
        }

        return {
          ...node,
          data: {
            label: updatedNode.label,
            description: updatedNode.description,
            color: updatedNode.color,
          },
        };
      }),
    );
  }, [laidOutNodes, setNodes]);

  return (
    <div className="h-full min-h-[22rem] overflow-hidden rounded-[1.7rem] border border-border bg-background/60">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={(_, node) => {
          const selected = laidOutNodes.find((entry) => entry.id === node.id) ?? null;
          onSelectNode(selected);
        }}
        fitView
        onInit={(instance) => {
          flowRef.current = instance;
          instance.fitView({ padding: 0.24, duration: 500 });
        }}
        nodeTypes={nodeTypes}
        panOnScroll
        selectionOnDrag={false}
      >
        <Background color="rgba(128,128,128,0.14)" gap={24} />
        <Controls position="bottom-right" />
        <MiniMap
          pannable
          zoomable
          className="!rounded-2xl !border !border-border !bg-background/90"
          nodeColor={() => "#0f766e"}
        />
      </ReactFlow>
    </div>
  );
}
