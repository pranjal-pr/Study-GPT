import type { ProviderKeys } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

interface TranscriptionBackend {
  apiKey: string;
  endpoint: string;
  model: string;
  provider: "groq" | "openai";
}

function jsonError(message: string, status = 400) {
  return Response.json({ message }, { status });
}

function resolveKey(name: keyof ProviderKeys, apiKeys?: Partial<ProviderKeys>) {
  const localKey = apiKeys?.[name]?.trim();
  if (localKey) {
    return localKey;
  }

  switch (name) {
    case "groq":
      return process.env.GROQ_API_KEY?.trim() ?? "";
    case "openai":
      return process.env.OPENAI_API_KEY?.trim() ?? "";
    default:
      return "";
  }
}

function resolveTranscriptionBackend(
  apiKeys?: Partial<ProviderKeys>,
): TranscriptionBackend | null {
  const groqKey = resolveKey("groq", apiKeys);
  if (groqKey) {
    return {
      apiKey: groqKey,
      endpoint: "https://api.groq.com/openai/v1/audio/transcriptions",
      model: "whisper-large-v3-turbo",
      provider: "groq",
    };
  }

  const openAiKey = resolveKey("openai", apiKeys);
  if (openAiKey) {
    return {
      apiKey: openAiKey,
      endpoint: "https://api.openai.com/v1/audio/transcriptions",
      model: "gpt-4o-mini-transcribe",
      provider: "openai",
    };
  }

  return null;
}

function parseApiKeys(raw: FormDataEntryValue | null) {
  if (typeof raw !== "string" || !raw.trim()) {
    return {};
  }

  try {
    return JSON.parse(raw) as Partial<ProviderKeys>;
  } catch {
    return {};
  }
}

export async function POST(request: Request) {
  let formData: FormData;

  try {
    formData = await request.formData();
  } catch {
    return jsonError("Invalid form data.");
  }

  const file = formData.get("file");
  if (!(file instanceof File) || file.size === 0) {
    return jsonError("An audio recording is required.");
  }

  const backend = resolveTranscriptionBackend(parseApiKeys(formData.get("apiKeys")));
  if (!backend) {
    return jsonError(
      "No transcription API key is available. Add a Groq or OpenAI API key in the UI or server environment.",
      401,
    );
  }

  const upstreamBody = new FormData();
  upstreamBody.append("file", file, file.name || "voice.webm");
  upstreamBody.append("model", backend.model);
  upstreamBody.append("response_format", "text");

  const language = formData.get("language");
  if (typeof language === "string" && language.trim()) {
    upstreamBody.append("language", language.trim());
  }

  const response = await fetch(backend.endpoint, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${backend.apiKey}`,
    },
    body: upstreamBody,
  });

  const text = (await response.text()).trim();
  if (!response.ok) {
    return jsonError(text || "Audio transcription failed.", response.status);
  }

  if (!text) {
    return jsonError("StudyGPT could not transcribe the recording.", 422);
  }

  return Response.json({
    provider: backend.provider,
    text,
  });
}
