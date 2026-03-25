import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenAI } from "@google/genai";
import Groq from "groq-sdk";
import OpenAI from "openai";
import { formatChatStreamEvent } from "@/lib/chat-stream";
import { AGENT_DISPLAY_NAMES } from "@/lib/constants";
import {
  buildExplanationAgentPrompt,
  buildVisualAgentPrompt,
  buildVoiceAgentPrompt,
} from "@/lib/system-prompt";
import type { ChatRequestBody, ProviderId, ProviderKeys } from "@/lib/types";
import { parseVisualAgentResponse } from "@/lib/visuals";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

type ProviderMessage = {
  role: "user" | "assistant";
  content: string;
};

interface ProviderCallParams {
  apiKey: string;
  messages: ProviderMessage[];
  model: string;
  systemPrompt: string;
}

function jsonError(message: string, status = 400) {
  return Response.json({ message }, { status });
}

function resolveApiKey(provider: ProviderId, apiKeys?: Partial<ProviderKeys>) {
  const localKey = apiKeys?.[provider]?.trim();
  if (localKey) {
    return localKey;
  }

  switch (provider) {
    case "groq":
      return process.env.GROQ_API_KEY?.trim() ?? "";
    case "openai":
      return process.env.OPENAI_API_KEY?.trim() ?? "";
    case "anthropic":
      return process.env.ANTHROPIC_API_KEY?.trim() ?? "";
    case "gemini":
      return process.env.GEMINI_API_KEY?.trim() ?? "";
    default:
      return "";
  }
}

function normalizeMessages(messages: ChatRequestBody["messages"]) {
  return messages
    .filter(
      (message): message is ProviderMessage =>
        (message.role === "user" || message.role === "assistant") &&
        typeof message.content === "string" &&
        message.content.trim().length > 0,
    )
    .map((message) => ({
      role: message.role,
      content: message.content.trim(),
    }));
}

async function* streamGroqCompletion(params: ProviderCallParams) {
  const client = new Groq({ apiKey: params.apiKey });
  const stream = await client.chat.completions.create({
    model: params.model,
    temperature: 0.55,
    stream: true,
    messages: [
      { role: "system", content: params.systemPrompt },
      ...params.messages,
    ],
  });

  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content;
    if (content) {
      yield content;
    }
  }
}

async function* streamOpenAICompletion(params: ProviderCallParams) {
  const client = new OpenAI({ apiKey: params.apiKey });
  const stream = await client.chat.completions.create({
    model: params.model,
    temperature: 0.55,
    stream: true,
    messages: [
      { role: "system", content: params.systemPrompt },
      ...params.messages,
    ],
  });

  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content;
    if (content) {
      yield content;
    }
  }
}

async function* streamAnthropicCompletion(params: ProviderCallParams) {
  const client = new Anthropic({ apiKey: params.apiKey });
  const stream = client.messages.stream({
    model: params.model,
    system: params.systemPrompt,
    max_tokens: 4096,
    temperature: 0.55,
    messages: params.messages,
  });

  for await (const event of stream) {
    if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
      yield event.delta.text;
    }
  }
}

async function* streamGeminiCompletion(params: ProviderCallParams) {
  const client = new GoogleGenAI({ apiKey: params.apiKey });
  const stream = await client.models.generateContentStream({
    model: params.model,
    contents: params.messages.map((message) => ({
      role: message.role === "assistant" ? "model" : "user",
      parts: [{ text: message.content }],
    })),
    config: {
      temperature: 0.55,
      systemInstruction: params.systemPrompt,
    },
  });

  for await (const chunk of stream) {
    if (chunk.text) {
      yield chunk.text;
    }
  }
}

async function completeGroqCompletion(params: ProviderCallParams) {
  const client = new Groq({ apiKey: params.apiKey });
  const response = await client.chat.completions.create({
    model: params.model,
    temperature: 0.45,
    messages: [
      { role: "system", content: params.systemPrompt },
      ...params.messages,
    ],
  });

  return response.choices[0]?.message?.content?.trim() ?? "";
}

async function completeOpenAICompletion(params: ProviderCallParams) {
  const client = new OpenAI({ apiKey: params.apiKey });
  const response = await client.chat.completions.create({
    model: params.model,
    temperature: 0.45,
    messages: [
      { role: "system", content: params.systemPrompt },
      ...params.messages,
    ],
  });

  return response.choices[0]?.message?.content?.trim() ?? "";
}

async function completeAnthropicCompletion(params: ProviderCallParams) {
  const client = new Anthropic({ apiKey: params.apiKey });
  const response = await client.messages.create({
    model: params.model,
    stream: false,
    system: params.systemPrompt,
    max_tokens: 2048,
    temperature: 0.45,
    messages: params.messages,
  });

  return response.content
    .filter((block): block is Extract<typeof block, { type: "text" }> => block.type === "text")
    .map((block) => block.text)
    .join("\n")
    .trim();
}

async function completeGeminiCompletion(params: ProviderCallParams) {
  const client = new GoogleGenAI({ apiKey: params.apiKey });
  const response = await client.models.generateContent({
    model: params.model,
    contents: params.messages.map((message) => ({
      role: message.role === "assistant" ? "model" : "user",
      parts: [{ text: message.content }],
    })),
    config: {
      temperature: 0.45,
      systemInstruction: params.systemPrompt,
    },
  });

  return response.text?.trim() ?? "";
}

function createProviderStream(params: ProviderCallParams & { provider: ProviderId }) {
  switch (params.provider) {
    case "groq":
      return streamGroqCompletion(params);
    case "openai":
      return streamOpenAICompletion(params);
    case "anthropic":
      return streamAnthropicCompletion(params);
    case "gemini":
      return streamGeminiCompletion(params);
    default:
      throw new Error("Unsupported provider");
  }
}

async function createProviderCompletion(
  params: ProviderCallParams & { provider: ProviderId },
) {
  switch (params.provider) {
    case "groq":
      return completeGroqCompletion(params);
    case "openai":
      return completeOpenAICompletion(params);
    case "anthropic":
      return completeAnthropicCompletion(params);
    case "gemini":
      return completeGeminiCompletion(params);
    default:
      throw new Error("Unsupported provider");
  }
}

export async function POST(request: Request) {
  let body: ChatRequestBody;

  try {
    body = (await request.json()) as ChatRequestBody;
  } catch {
    return jsonError("Invalid JSON payload.");
  }

  if (!body?.provider || !body?.model || !Array.isArray(body?.messages)) {
    return jsonError("Missing provider, model, or messages.");
  }

  const messages = normalizeMessages(body.messages);
  if (messages.length === 0) {
    return jsonError("At least one user or assistant message is required.");
  }

  const apiKey = resolveApiKey(body.provider, body.apiKeys);
  if (!apiKey) {
    return jsonError(
      `No API key available for ${body.provider}. Add one in the UI or set the matching environment variable.`,
      401,
    );
  }

  const encoder = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const emit = (payload: Parameters<typeof formatChatStreamEvent>[0]) => {
        controller.enqueue(encoder.encode(formatChatStreamEvent(payload)));
      };

      const backgroundTasks: Promise<void>[] = [];

      emit({
        type: "agent",
        agent: "text",
        status: "running",
        message: `${AGENT_DISPLAY_NAMES.text} is teaching`,
      });
      emit({
        type: "agent",
        agent: "visual",
        status: "running",
        message: `${AGENT_DISPLAY_NAMES.visual} is mapping the topic`,
      });
      emit({
        type: "agent",
        agent: "voice",
        status: body.voiceMode ? "running" : "idle",
        message: body.voiceMode
          ? `${AGENT_DISPLAY_NAMES.voice} agent is preparing the spoken reply`
          : "Voice mode is off",
      });

      backgroundTasks.push(
        (async () => {
          try {
            const rawVisual = await createProviderCompletion({
              apiKey,
              messages,
              model: body.model,
              provider: body.provider,
              systemPrompt: buildVisualAgentPrompt(body.regenerateVisualization),
            });
            const visual = parseVisualAgentResponse(rawVisual);

            emit({
              type: "visual",
              visual,
            });
            emit({
              type: "agent",
              agent: "visual",
              status: "done",
              message: visual
                ? `${AGENT_DISPLAY_NAMES.visual} finished the interactive diagram`
                : `${AGENT_DISPLAY_NAMES.visual} decided no visual was needed`,
            });
          } catch (error) {
            emit({
              type: "agent",
              agent: "visual",
              status: "error",
              message:
                error instanceof Error
                  ? error.message
                  : `${AGENT_DISPLAY_NAMES.visual} failed to finish`,
            });
          }
        })(),
      );

      if (body.voiceMode) {
        backgroundTasks.push(
          (async () => {
            try {
              const voiceScript = await createProviderCompletion({
                apiKey,
                messages,
                model: body.model,
                provider: body.provider,
                systemPrompt: buildVoiceAgentPrompt(),
              });

              if (voiceScript) {
                emit({
                  type: "voice",
                  voiceScript,
                });
              }
              emit({
                type: "agent",
                agent: "voice",
                status: "done",
                message: voiceScript
                  ? `${AGENT_DISPLAY_NAMES.voice} agent prepared the spoken reply`
                  : `${AGENT_DISPLAY_NAMES.voice} agent returned an empty spoken reply`,
              });
            } catch (error) {
              emit({
                type: "agent",
                agent: "voice",
                status: "error",
                message:
                  error instanceof Error
                    ? error.message
                    : `${AGENT_DISPLAY_NAMES.voice} agent failed to finish`,
              });
            }
          })(),
        );
      }

      try {
        const providerStream = createProviderStream({
          apiKey,
          messages,
          model: body.model,
          provider: body.provider,
          systemPrompt: buildExplanationAgentPrompt(body.regenerateVisualization),
        });

        for await (const chunk of providerStream) {
          emit({
            type: "text-delta",
            delta: chunk,
          });
        }

        emit({
          type: "agent",
          agent: "text",
          status: "done",
          message: `${AGENT_DISPLAY_NAMES.text} finished the explanation`,
        });
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : "StudyGPT could not complete the explanation.";

        emit({
          type: "agent",
          agent: "text",
          status: "error",
          message,
        });
        emit({
          type: "error",
          message,
        });
      } finally {
        await Promise.allSettled(backgroundTasks);
        emit({
          type: "done",
        });
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
