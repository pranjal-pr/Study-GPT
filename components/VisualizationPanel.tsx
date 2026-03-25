"use client";

import { useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Download, RefreshCw, Share2, Sparkles } from "lucide-react";
import { toPng, toSvg } from "html-to-image";
import { AGENT_DISPLAY_NAMES } from "@/lib/constants";
import type { AgentRuntimeState, VisualNode, VisualPayload } from "@/lib/types";
import { buildMermaidFromVisual } from "@/lib/visuals";
import { downloadDataUrl } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ChartView } from "@/components/ChartView";
import { MermaidFallback } from "@/components/MermaidFallback";
import { MindMap } from "@/components/MindMap";

interface VisualizationPanelProps {
  onExplainNode: (node: VisualNode) => void;
  onRegenerateVisualization: () => void;
  visualAgentState: AgentRuntimeState;
  visual: VisualPayload | null;
}

export function VisualizationPanel({
  onExplainNode,
  onRegenerateVisualization,
  visualAgentState,
  visual,
}: VisualizationPanelProps) {
  const [selectedNode, setSelectedNode] = useState<VisualNode | null>(null);
  const [showMermaid, setShowMermaid] = useState(false);
  const [workingVisual, setWorkingVisual] = useState<VisualPayload | null>(visual);
  const exportRef = useRef<HTMLDivElement>(null);

  const mermaidChart = useMemo(
    () => (workingVisual ? buildMermaidFromVisual(workingVisual) : ""),
    [workingVisual],
  );

  const exportVisual = async (format: "png" | "svg") => {
    if (!exportRef.current || !workingVisual) {
      return;
    }

    const slug = workingVisual.title.toLowerCase().replace(/[^a-z0-9]+/g, "-");
    const dataUrl =
      format === "png" ? await toPng(exportRef.current) : await toSvg(exportRef.current);
    downloadDataUrl(dataUrl, `${slug || "studygpt-visual"}.${format}`);
  };

  const updateSelectedNodeLabel = (nextLabel: string) => {
    if (!selectedNode) {
      return;
    }

    setSelectedNode((current) => (current ? { ...current, label: nextLabel } : current));
    setWorkingVisual((current) => {
      if (!current) {
        return current;
      }

      return {
        ...current,
        nodes: current.nodes.map((node) =>
          node.id === selectedNode.id ? { ...node, label: nextLabel } : node,
        ),
      };
    });
  };

  return (
    <Card className="flex h-full min-h-0 flex-col overflow-hidden">
      <CardHeader className="pb-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <CardTitle className="text-lg">
                {workingVisual?.title ?? "Live visualization"}
              </CardTitle>
              {workingVisual ? (
                <Badge>{workingVisual.type === "mindmap" ? "Mind map" : "Chart"}</Badge>
              ) : null}
              <Badge variant="secondary" className="normal-case tracking-normal">
                {AGENT_DISPLAY_NAMES.visual}: {visualAgentState.status}
              </Badge>
            </div>
            <p className="mt-2 text-sm text-muted-foreground">
              StudyGPT draws concepts here in real time. Click nodes to dive deeper or
              switch to Mermaid when you want a compact fallback view.
            </p>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            <Button type="button" variant="secondary" size="sm" onClick={onRegenerateVisualization}>
              <RefreshCw className="h-4 w-4" />
              Regenerate
            </Button>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={() => setShowMermaid((current) => !current)}
              disabled={!workingVisual}
            >
              <Share2 className="h-4 w-4" />
              {showMermaid ? "Interactive view" : "Mermaid fallback"}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col gap-4 overflow-hidden">
        {!workingVisual ? (
          <div className="grid min-h-0 flex-1 place-items-center rounded-[1.7rem] border border-dashed border-border bg-muted/25 p-8 text-center">
            <div>
              <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary/12 text-primary">
                <Sparkles className="h-7 w-7" />
              </div>
              <h3 className="font-heading text-xl font-semibold">Waiting for a visual</h3>
              <p className="mt-3 max-w-md text-sm text-muted-foreground">
                Ask for a mind map, flowchart, or chart. StudyGPT will open it here
                automatically once the model returns structured visual JSON.
              </p>
              <p className="mt-3 text-xs uppercase tracking-[0.16em] text-muted-foreground">
                {visualAgentState.message}
              </p>
            </div>
          </div>
        ) : (
          <>
            <div className="flex min-h-0 flex-1 flex-col gap-4">
              <div ref={exportRef} className="min-h-0 flex-1">
                {showMermaid ? (
                  <MermaidFallback chart={mermaidChart} />
                ) : workingVisual.type === "mindmap" ? (
                  <MindMap
                    animationKey={`${workingVisual.title}-${workingVisual.nodes.length}-${workingVisual.edges.length}`}
                    visual={workingVisual}
                    onSelectNode={setSelectedNode}
                  />
                ) : (
                  <ChartView visual={workingVisual} />
                )}
              </div>

              <div className="flex flex-wrap gap-2">
                <Button type="button" variant="secondary" size="sm" onClick={() => exportVisual("png")}>
                  <Download className="h-4 w-4" />
                  Export PNG
                </Button>
                <Button type="button" variant="secondary" size="sm" onClick={() => exportVisual("svg")}>
                  <Download className="h-4 w-4" />
                  Export SVG
                </Button>
              </div>
            </div>

            {workingVisual.type === "mindmap" ? (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-[1.5rem] border border-border bg-muted/35 p-4"
              >
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold">Selected node</p>
                    <p className="text-xs text-muted-foreground">
                      Click a node, rename it locally, or ask StudyGPT to explain it more.
                    </p>
                  </div>
                  <Button
                    type="button"
                    size="sm"
                    onClick={() => selectedNode && onExplainNode(selectedNode)}
                    disabled={!selectedNode}
                  >
                    Explain this node
                  </Button>
                </div>
                <Input
                  className="mt-4"
                  value={selectedNode?.label ?? ""}
                  onChange={(event) => updateSelectedNodeLabel(event.target.value)}
                  placeholder="Select a node to edit its label"
                  disabled={!selectedNode}
                />
                {selectedNode?.description ? (
                  <p className="mt-3 text-sm text-muted-foreground">
                    {selectedNode.description}
                  </p>
                ) : null}
              </motion.div>
            ) : null}
          </>
        )}
      </CardContent>
    </Card>
  );
}
