import type {
  AgentKind,
  AgentRuntimeMap,
  ModelOption,
  ProviderId,
  VoiceSettings,
} from "@/lib/types";

export const MODEL_OPTIONS: ModelOption[] = [
  {
    id: "llama-3.3-70b-versatile",
    label: "Groq Llama 3.3 70B",
    description: "Fast, rich tutoring by default.",
    provider: "groq",
  },
  {
    id: "llama-3.1-8b-instant",
    label: "Groq Llama 3.1 8B Instant",
    description: "Lowest latency for quick follow-ups.",
    provider: "groq",
  },
  {
    id: "mistral-saba-24b",
    label: "Groq Mistral Saba 24B",
    description: "Balanced multilingual teaching.",
    provider: "groq",
  },
  {
    id: "gpt-4o",
    label: "OpenAI GPT-4o",
    description: "High quality explanation and synthesis.",
    provider: "openai",
  },
  {
    id: "gpt-4o-mini",
    label: "OpenAI GPT-4o Mini",
    description: "Fast budget-friendly tutoring.",
    provider: "openai",
  },
  {
    id: "claude-sonnet-4-6",
    label: "Claude Sonnet 4.6",
    description: "Deep reasoning and nuanced coaching.",
    provider: "anthropic",
  },
  {
    id: "claude-haiku-4-5",
    label: "Claude Haiku 4.5",
    description: "Fast Anthropic responses.",
    provider: "anthropic",
  },
  {
    id: "gemini-2.5-flash",
    label: "Gemini 2.5 Flash",
    description: "Fast multimodal explanations.",
    provider: "gemini",
  },
  {
    id: "gemini-2.5-pro",
    label: "Gemini 2.5 Pro",
    description: "Longer-form analytical tutoring.",
    provider: "gemini",
  },
];

export const PROVIDER_LABELS: Record<ProviderId, string> = {
  groq: "Groq",
  openai: "OpenAI",
  anthropic: "Anthropic",
  gemini: "Gemini",
};

export const DEFAULT_MODEL = MODEL_OPTIONS[0];

export const DEFAULT_VOICE_SETTINGS: VoiceSettings = {
  enabled: false,
  mode: "continuous",
  rate: 1,
  pitch: 1,
  autoSpeak: true,
  voiceURI: "",
};

export const SUGGESTED_TOPICS = [
  "Explain quantum mechanics simply",
  "Teach me calculus from first principles",
  "Walk me through World War II causes",
  "Break down recursion with visuals",
  "How does photosynthesis actually work?",
  "Teach me SQL joins with a chart",
  "Explain market cycles like a tutor",
  "Map out the human immune system",
];

export const STUDYGPT_STORAGE_KEY = "studygpt-app-state";

export const VOICE_SHORTCUT = "Cmd/Ctrl + Shift + M";
export const NEW_CHAT_SHORTCUT = "Cmd/Ctrl + K";

export const VISUAL_REGENERATION_PROMPT =
  "Regenerate the best visualization for the most recent explanation. Keep the prose to two short sentences and output a strong visual block at the end.";

export const AGENT_DISPLAY_NAMES: Record<AgentKind, string> = {
  text: "Claude",
  visual: "Codex",
  voice: "Echo",
};

export const DEFAULT_AGENT_RUNTIME_STATE: AgentRuntimeMap = {
  text: {
    status: "idle",
    message: `${AGENT_DISPLAY_NAMES.text} is ready`,
  },
  visual: {
    status: "idle",
    message: `${AGENT_DISPLAY_NAMES.visual} is ready`,
  },
  voice: {
    status: "idle",
    message: `${AGENT_DISPLAY_NAMES.voice} agent is ready`,
  },
};
