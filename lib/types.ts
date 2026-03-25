export type TutorRole = "user" | "assistant";

export type ProviderId = "groq" | "openai" | "anthropic" | "gemini";
export type AgentKind = "text" | "visual" | "voice";
export type AgentRunStatus = "idle" | "running" | "done" | "error";

export interface ModelOption {
  id: string;
  label: string;
  description: string;
  provider: ProviderId;
}

export interface ProviderKeys {
  groq: string;
  openai: string;
  anthropic: string;
  gemini: string;
}

export interface VisualNode {
  id: string;
  label: string;
  description?: string;
  color?: string;
  level?: number;
  group?: string;
  position?: {
    x: number;
    y: number;
  };
}

export interface VisualEdge {
  id?: string;
  source: string;
  target: string;
  label?: string;
  animated?: boolean;
}

export type ChartDatum = Record<string, number | string | null>;

export interface VisualPayload {
  type: "mindmap" | "chart";
  title: string;
  nodes: VisualNode[];
  edges: VisualEdge[];
  chartData: ChartDatum[];
  chartType?: "bar" | "line" | "pie" | "area";
  xKey?: string;
  seriesKeys?: string[];
  description?: string;
}

export interface ChatMessage {
  id: string;
  role: TutorRole;
  content: string;
  createdAt: string;
  model?: string;
  provider?: ProviderId;
  visual?: VisualPayload | null;
  voiceScript?: string | null;
  status?: "streaming" | "done" | "error";
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  provider: ProviderId;
  model: string;
  messages: ChatMessage[];
  saved: boolean;
}

export interface VoiceSettings {
  enabled: boolean;
  mode: "continuous" | "push";
  rate: number;
  pitch: number;
  autoSpeak: boolean;
  voiceURI: string;
}

export interface ChatRequestBody {
  provider: ProviderId;
  model: string;
  messages: Pick<ChatMessage, "role" | "content">[];
  apiKeys?: Partial<ProviderKeys>;
  regenerateVisualization?: boolean;
  voiceMode?: boolean;
}

export interface AgentRuntimeState {
  status: AgentRunStatus;
  message: string;
}

export type AgentRuntimeMap = Record<AgentKind, AgentRuntimeState>;

export type ChatStreamEvent =
  | {
      type: "agent";
      agent: AgentKind;
      status: AgentRunStatus;
      message: string;
    }
  | {
      type: "text-delta";
      delta: string;
    }
  | {
      type: "visual";
      visual: VisualPayload | null;
    }
  | {
      type: "voice";
      voiceScript: string;
    }
  | {
      type: "error";
      message: string;
    }
  | {
      type: "done";
    };
