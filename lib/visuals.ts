import type { ChartDatum, VisualEdge, VisualNode, VisualPayload } from "@/lib/types";

const VISUAL_BLOCK_REGEX = /```visual\s*([\s\S]*?)```/i;

function safeJsonParse(value: string) {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function normalizeNode(node: unknown, index: number): VisualNode | null {
  if (!node || typeof node !== "object") {
    return null;
  }

  const record = node as Record<string, unknown>;
  const id = String(record.id ?? `node-${index + 1}`);
  const label = String(record.label ?? record.title ?? id);

  return {
    id,
    label,
    description:
      typeof record.description === "string" ? record.description : undefined,
    color: typeof record.color === "string" ? record.color : undefined,
    group: typeof record.group === "string" ? record.group : undefined,
    level:
      typeof record.level === "number" && Number.isFinite(record.level)
        ? record.level
        : undefined,
    position:
      record.position &&
      typeof record.position === "object" &&
      typeof (record.position as { x?: unknown }).x === "number" &&
      typeof (record.position as { y?: unknown }).y === "number"
        ? {
            x: (record.position as { x: number }).x,
            y: (record.position as { y: number }).y,
          }
        : undefined,
  };
}

function normalizeEdge(edge: unknown, index: number): VisualEdge | null {
  if (!edge || typeof edge !== "object") {
    return null;
  }

  const record = edge as Record<string, unknown>;
  if (typeof record.source !== "string" || typeof record.target !== "string") {
    return null;
  }

  return {
    id:
      typeof record.id === "string"
        ? record.id
        : `edge-${record.source}-${record.target}-${index}`,
    source: record.source,
    target: record.target,
    label: typeof record.label === "string" ? record.label : undefined,
    animated: typeof record.animated === "boolean" ? record.animated : undefined,
  };
}

function normalizeChartData(value: unknown): ChartDatum[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .filter((entry) => entry && typeof entry === "object")
    .map((entry) => {
      const result: ChartDatum = {};
      for (const [key, raw] of Object.entries(entry as Record<string, unknown>)) {
        if (typeof raw === "number" || typeof raw === "string" || raw === null) {
          result[key] = raw;
        }
      }
      return result;
    });
}

export function parseVisualBlock(content: string) {
  const match = content.match(VISUAL_BLOCK_REGEX);
  const cleanedContent = content.replace(VISUAL_BLOCK_REGEX, "").trim();

  if (!match) {
    return {
      cleanedContent,
      visual: null,
      rawVisual: null,
    };
  }

  const rawVisual = match[1]?.trim() ?? "";
  const parsed = safeJsonParse(rawVisual);
  const visual = normalizeVisual(parsed);

  return {
    cleanedContent,
    visual,
    rawVisual,
  };
}

export function stripPartialVisual(content: string) {
  const visualIndex = content.toLowerCase().indexOf("```visual");
  if (visualIndex === -1) {
    return content;
  }

  return content.slice(0, visualIndex).trimEnd();
}

export function parseVisualAgentResponse(content: string) {
  const trimmed = content.trim();
  if (!trimmed) {
    return null;
  }

  const direct = normalizeVisual(safeJsonParse(trimmed));
  if (direct) {
    return direct;
  }

  const fenced = trimmed.match(/```(?:json|visual)?\s*([\s\S]*?)```/i);
  if (fenced?.[1]) {
    return normalizeVisual(safeJsonParse(fenced[1].trim()));
  }

  return null;
}

export function normalizeVisual(input: unknown): VisualPayload | null {
  if (!input || typeof input !== "object") {
    return null;
  }

  const record = input as Record<string, unknown>;
  const type = record.type;
  const title =
    typeof record.title === "string" && record.title.trim()
      ? record.title.trim()
      : "Live visualization";

  if (type !== "mindmap" && type !== "chart") {
    return null;
  }

  const nodes = Array.isArray(record.nodes)
    ? record.nodes.map(normalizeNode).filter(Boolean)
    : [];
  const edges = Array.isArray(record.edges)
    ? record.edges.map(normalizeEdge).filter(Boolean)
    : [];
  const chartData = normalizeChartData(record.chartData);

  return {
    type,
    title,
    nodes: nodes as VisualNode[],
    edges: edges as VisualEdge[],
    chartData,
    chartType:
      record.chartType === "bar" ||
      record.chartType === "line" ||
      record.chartType === "pie" ||
      record.chartType === "area"
        ? record.chartType
        : undefined,
    xKey: typeof record.xKey === "string" ? record.xKey : undefined,
    seriesKeys: Array.isArray(record.seriesKeys)
      ? record.seriesKeys.filter((value): value is string => typeof value === "string")
      : undefined,
    description:
      typeof record.description === "string" ? record.description : undefined,
  };
}

export function buildMindMapLayout(visual: VisualPayload) {
  const nodes = visual.nodes;
  const edges = visual.edges;

  if (!nodes.length) {
    return [];
  }

  const incoming = new Map<string, number>();
  const children = new Map<string, string[]>();
  const depthByNode = new Map<string, number>();
  const sideByNode = new Map<string, number>();

  for (const node of nodes) {
    incoming.set(node.id, 0);
    children.set(node.id, []);
  }

  for (const edge of edges) {
    incoming.set(edge.target, (incoming.get(edge.target) ?? 0) + 1);
    children.set(edge.source, [...(children.get(edge.source) ?? []), edge.target]);
  }

  const root = nodes.find((node) => (incoming.get(node.id) ?? 0) === 0) ?? nodes[0];
  depthByNode.set(root.id, 0);
  sideByNode.set(root.id, 0);

  const queue = [root.id];
  (children.get(root.id) ?? []).forEach((childId, index) => {
    sideByNode.set(childId, index % 2 === 0 ? 1 : -1);
  });

  while (queue.length > 0) {
    const current = queue.shift()!;
    const currentDepth = depthByNode.get(current) ?? 0;
    const currentSide = sideByNode.get(current) ?? 1;

    for (const childId of children.get(current) ?? []) {
      if (!depthByNode.has(childId)) {
        depthByNode.set(childId, currentDepth + 1);
        sideByNode.set(
          childId,
          current === root.id ? sideByNode.get(childId) ?? currentSide : currentSide,
        );
        queue.push(childId);
      }
    }
  }

  for (const node of nodes) {
    if (!depthByNode.has(node.id)) {
      depthByNode.set(node.id, 1);
      sideByNode.set(node.id, 1);
    }
  }

  const buckets = new Map<string, string[]>();
  for (const node of nodes) {
    const depth = depthByNode.get(node.id) ?? 0;
    const side = sideByNode.get(node.id) ?? 1;
    const key = `${depth}:${side}`;
    buckets.set(key, [...(buckets.get(key) ?? []), node.id]);
  }

  return nodes.map((node) => {
    if (node.position) {
      return node;
    }

    const depth = depthByNode.get(node.id) ?? 0;
    const side = sideByNode.get(node.id) ?? 1;
    const bucket = buckets.get(`${depth}:${side}`) ?? [node.id];
    const index = bucket.indexOf(node.id);

    return {
      ...node,
      position: {
        x: depth === 0 ? 0 : side * depth * 260,
        y: (index - (bucket.length - 1) / 2) * 160,
      },
    };
  });
}

export function inferChartConfig(visual: VisualPayload) {
  const sample = visual.chartData[0];
  if (!sample) {
    return null;
  }

  const keys = Object.keys(sample);
  const xKey =
    visual.xKey ??
    keys.find((key) => typeof sample[key] === "string") ??
    keys[0];

  const seriesKeys =
    visual.seriesKeys?.length
      ? visual.seriesKeys
      : keys.filter((key) => key !== xKey && typeof sample[key] === "number");

  const chartType =
    visual.chartType ??
    (seriesKeys.length === 1 && visual.chartData.length <= 6 ? "bar" : "line");

  return {
    chartType,
    xKey,
    seriesKeys,
  };
}

export function buildMermaidFromVisual(visual: VisualPayload) {
  if (visual.type === "chart") {
    return `graph TD
  chart["${visual.title}"]
  data["${visual.chartData.length} data points"]
  chart --> data`;
  }

  const lines = ["graph TD"];

  for (const node of visual.nodes) {
    lines.push(`  ${sanitizeMermaidId(node.id)}["${escapeMermaidText(node.label)}"]`);
  }

  for (const edge of visual.edges) {
    const label = edge.label ? `|${escapeMermaidText(edge.label)}|` : "";
    lines.push(
      `  ${sanitizeMermaidId(edge.source)} -->${label} ${sanitizeMermaidId(edge.target)}`,
    );
  }

  return lines.join("\n");
}

function sanitizeMermaidId(id: string) {
  return id.replace(/[^a-zA-Z0-9_]/g, "_");
}

function escapeMermaidText(text: string) {
  return text.replace(/"/g, '\\"');
}
