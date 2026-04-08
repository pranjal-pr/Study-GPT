"use client";

import { useEffect, useMemo, useState, useTransition } from "react";
import { PanelLeftOpen, PanelRightOpen, Plus, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { useKeyboardShortcuts } from "@/hooks/use-keyboard-shortcuts";
import { useVoiceTutor } from "@/hooks/use-voice-tutor";
import {
  AGENT_DISPLAY_NAMES,
  DEFAULT_AGENT_RUNTIME_STATE,
  DEFAULT_MODEL,
  MODEL_OPTIONS,
  NEW_CHAT_SHORTCUT,
  VISUAL_REGENERATION_PROMPT,
  VOICE_SHORTCUT,
} from "@/lib/constants";
import type { ChatMessage, VisualNode } from "@/lib/types";
import { parseChatStreamBuffer } from "@/lib/chat-stream";
import { useStudyGPTStore } from "@/lib/use-studygpt-store";
import { createId, nowIso } from "@/lib/utils";
import { ApiKeysDialog } from "@/components/ApiKeysDialog";
import { Chat } from "@/components/Chat";
import { ModelSelector } from "@/components/ModelSelector";
import { Sidebar } from "@/components/Sidebar";
import { ThemeToggle } from "@/components/ThemeToggle";
import { VisualizationPanel } from "@/components/VisualizationPanel";
import { VoiceToggle } from "@/components/VoiceToggle";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";

export function StudyGPTShell() {
  const [input, setInput] = useState("");
  const [agentRuntime, setAgentRuntime] = useState(DEFAULT_AGENT_RUNTIME_STATE);
  const [isLoading, setIsLoading] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [mobileVisualOpen, setMobileVisualOpen] = useState(false);
  const [directAppHref, setDirectAppHref] = useState<string | undefined>(undefined);
  const [, startTransition] = useTransition();

  const {
    activeSessionId,
    apiKeys,
    appendMessage,
    createSession,
    deleteSession,
    hasHydrated,
    sessions,
    selectedModel,
    selectedProvider,
    setActiveSession,
    setVisualizationOpen,
    toggleSaved,
    updateApiKey,
    updateMessage,
    updateSelectedModel,
    updateVoiceSettings,
    visualizationOpen,
    voiceSettings,
  } = useStudyGPTStore();

  const sortedSessions = useMemo(
    () =>
      [...sessions].sort(
        (left, right) =>
          new Date(right.updatedAt).getTime() - new Date(left.updatedAt).getTime(),
      ),
    [sessions],
  );

  const activeSession =
    sortedSessions.find((session) => session.id === activeSessionId) ?? sortedSessions[0] ?? null;

  useEffect(() => {
    if (!hasHydrated) {
      return;
    }

    if (!activeSession && sortedSessions.length === 0) {
      const nextId = createSession();
      setActiveSession(nextId);
      return;
    }

    if (!activeSession && sortedSessions[0]) {
      setActiveSession(sortedSessions[0].id);
    }
  }, [activeSession, createSession, hasHydrated, setActiveSession, sortedSessions]);

  const activeModel =
    MODEL_OPTIONS.find((model) => model.id === selectedModel) ?? DEFAULT_MODEL;
  const latestAssistantMessage =
    [...(activeSession?.messages ?? [])]
      .reverse()
      .find((message) => message.role === "assistant") ?? null;
  const latestVisual = latestAssistantMessage?.visual ?? null;
  const visualizationKey = latestVisual
    ? `${activeSession?.id ?? "session"}-${latestVisual.title}-${latestVisual.type}-${latestVisual.nodes.length}-${latestVisual.edges.length}-${latestVisual.chartData.length}`
    : `${activeSession?.id ?? "session"}-empty`;

  useEffect(() => {
    setAgentRuntime(DEFAULT_AGENT_RUNTIME_STATE);
  }, [activeSession?.id]);

  const {
    availableVoices,
    isEmbeddedFrame,
    isListening,
    isSpeaking,
    isSupported,
    speak,
    startListening,
    stopListening,
    stopSpeaking,
    toggleListening,
    transcriptPreview,
  } = useVoiceTutor({
    settings: voiceSettings,
    onError: (message) => {
      toast.error(message);
    },
    onTranscript: (value) => {
      setInput(value);
      void sendPrompt(value);
    },
  });

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    setDirectAppHref(window.location.href);
  }, []);

  useEffect(() => {
    if (!voiceSettings.enabled) {
      stopListening();
      stopSpeaking();
    }
  }, [stopListening, stopSpeaking, voiceSettings.enabled]);

  const toggleVoiceMode = () => {
    if (!isSupported) {
      toast.error("Speech recognition is not available in this browser.");
      return;
    }

    const nextEnabled = !voiceSettings.enabled;
    updateVoiceSettings({ enabled: nextEnabled });

    if (nextEnabled) {
      startListening();
    }
  };

  const handleNewChat = () => {
    startTransition(() => {
      setInput("");
      const nextId = createSession();
      setActiveSession(nextId);
    });
  };

  useKeyboardShortcuts({
    onNewChat: handleNewChat,
    onToggleVoice: toggleVoiceMode,
  });

  const sendPrompt = async (
    prompt: string,
    options?: {
      messagesSnapshot?: ChatMessage[];
      regenerateVisualization?: boolean;
      sessionId?: string;
    },
  ) => {
    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt) {
      return;
    }

    const sessionId = options?.sessionId ?? activeSession?.id ?? createSession(trimmedPrompt);
    const historySnapshot = options?.messagesSnapshot ?? activeSession?.messages ?? [];

    const userMessage: ChatMessage = {
      id: createId(),
      role: "user",
      content: trimmedPrompt,
      createdAt: nowIso(),
      status: "done",
    };

    const assistantMessageId = createId();
    appendMessage(sessionId, userMessage);
    appendMessage(sessionId, {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      createdAt: nowIso(),
      provider: selectedProvider,
      model: selectedModel,
      visual: null,
      voiceScript: null,
      status: "streaming",
    });
    setActiveSession(sessionId);
    setInput("");
    setIsLoading(true);
    setAgentRuntime({
      text: {
        status: "running",
        message: `${AGENT_DISPLAY_NAMES.text} is teaching`,
      },
      visual: {
        status: "running",
        message: `${AGENT_DISPLAY_NAMES.visual} is mapping the topic`,
      },
      voice: {
        status: voiceSettings.enabled ? "running" : "idle",
        message: voiceSettings.enabled
          ? `${AGENT_DISPLAY_NAMES.voice} agent is preparing the spoken reply`
          : "Voice mode is off",
      },
    });

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          provider: selectedProvider,
          model: selectedModel,
          apiKeys,
          regenerateVisualization: options?.regenerateVisualization ?? false,
          voiceMode: voiceSettings.enabled,
          messages: [...historySnapshot, userMessage].map((message) => ({
            role: message.role,
            content: message.content,
          })),
        }),
      });

      if (!response.ok || !response.body) {
        const message = await response.text();
        throw new Error(message || "StudyGPT could not reach the selected provider.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let streamBuffer = "";
      let assistantText = "";
      let voiceScript = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        streamBuffer += decoder.decode(value, { stream: true });
        const { events, remaining } = parseChatStreamBuffer(streamBuffer);
        streamBuffer = remaining;

        for (const event of events) {
          if (event.type === "text-delta") {
            assistantText += event.delta;
            updateMessage(sessionId, assistantMessageId, {
              content: assistantText,
              status: "streaming",
            });
            continue;
          }

          if (event.type === "visual") {
            updateMessage(sessionId, assistantMessageId, {
              visual: event.visual,
            });
            if (event.visual) {
              setVisualizationOpen(true);
            }
            continue;
          }

          if (event.type === "voice") {
            voiceScript = event.voiceScript;
            updateMessage(sessionId, assistantMessageId, {
              voiceScript: event.voiceScript,
            });
            continue;
          }

          if (event.type === "agent") {
            setAgentRuntime((current) => ({
              ...current,
              [event.agent]: {
                status: event.status,
                message: event.message,
              },
            }));
            continue;
          }

          if (event.type === "error") {
            throw new Error(event.message);
          }
        }
      }

      if (streamBuffer.trim()) {
        const { events } = parseChatStreamBuffer(`${streamBuffer}\n\n`);
        for (const event of events) {
          if (event.type === "visual") {
            updateMessage(sessionId, assistantMessageId, {
              visual: event.visual,
            });
            if (event.visual) {
              setVisualizationOpen(true);
            }
          } else if (event.type === "voice") {
            voiceScript = event.voiceScript;
            updateMessage(sessionId, assistantMessageId, {
              voiceScript: event.voiceScript,
            });
          } else if (event.type === "agent") {
            setAgentRuntime((current) => ({
              ...current,
              [event.agent]: {
                status: event.status,
                message: event.message,
              },
            }));
          } else if (event.type === "error") {
            throw new Error(event.message);
          } else if (event.type === "text-delta") {
            assistantText += event.delta;
          }
        }
      }

      updateMessage(sessionId, assistantMessageId, {
        content: assistantText.trim(),
        status: "done",
      });
      await speak(voiceScript || assistantText);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "StudyGPT hit an unexpected provider error.";
      updateMessage(sessionId, assistantMessageId, {
        content: message,
        status: "error",
      });
      toast.error(message);
    } finally {
      setAgentRuntime((current) => ({
        text:
          current.text.status === "running"
            ? {
                status: "done",
                message: `${AGENT_DISPLAY_NAMES.text} finished`,
              }
            : current.text,
        visual:
          current.visual.status === "running"
            ? {
                status: "done",
                message: `${AGENT_DISPLAY_NAMES.visual} finished`,
              }
            : current.visual,
        voice:
          current.voice.status === "running"
            ? {
                status: "done",
                message: `${AGENT_DISPLAY_NAMES.voice} agent finished`,
              }
            : current.voice,
      }));
      setIsLoading(false);
    }
  };

  const handleSubmit = async () => {
    await sendPrompt(input);
  };

  const handleTopicClick = (topic: string) => {
    const nextId = createSession(topic);
    setActiveSession(nextId);
    setMobileSidebarOpen(false);
    void sendPrompt(topic, {
      sessionId: nextId,
      messagesSnapshot: [],
    });
  };

  const handleExplainNode = (node: VisualNode) => {
    void sendPrompt(
      `Explain the "${node.label}" node in simpler language, then extend the current visual with a deeper branch focused on this idea.`,
      {
        regenerateVisualization: true,
      },
    );
  };

  const handleRegenerateVisualization = () => {
    void sendPrompt(VISUAL_REGENERATION_PROMPT, {
      regenerateVisualization: true,
    });
  };

  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="grid-noise pointer-events-none absolute inset-0 opacity-[0.04]" />
      <div className="relative flex min-h-screen flex-col">
        <header className="sticky top-0 z-30 border-b border-border/70 bg-background/75 backdrop-blur-xl">
          <div className="mx-auto flex w-full max-w-[1800px] items-center justify-between gap-3 px-4 py-3">
            <div className="flex items-center gap-3">
              <Button
                type="button"
                variant="secondary"
                size="icon"
                className="lg:hidden"
                onClick={() => setMobileSidebarOpen(true)}
              >
                <PanelLeftOpen className="h-4 w-4" />
              </Button>
              <div>
                <div className="flex items-center gap-2">
                  <div className="animated-orb flex h-10 w-10 items-center justify-center rounded-full bg-primary text-primary-foreground">
                    <Sparkles className="h-5 w-5" />
                  </div>
                  <div>
                    <p className="font-heading text-lg font-semibold">StudyGPT</p>
                    <p className="text-xs text-muted-foreground">
                      Live tutor with chat, voice, and visuals
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="hidden flex-1 items-center justify-center gap-3 lg:flex">
              <ModelSelector
                value={selectedModel}
                onChange={(model, provider) => updateSelectedModel(model, provider)}
              />
              <Badge variant="secondary">{activeModel.description}</Badge>
            </div>

            <div className="flex items-center gap-2">
              <VoiceToggle
                availableVoices={availableVoices}
                directAppHref={directAppHref}
                isEmbeddedFrame={isEmbeddedFrame}
                isListening={isListening}
                isSpeaking={isSpeaking}
                isSupported={isSupported}
                settings={voiceSettings}
                onToggleEnabled={toggleVoiceMode}
                onToggleListening={toggleListening}
                onStopSpeaking={stopSpeaking}
                onUpdateSettings={updateVoiceSettings}
              />
              <div className="hidden md:block">
                <ApiKeysDialog apiKeys={apiKeys} onChange={updateApiKey} />
              </div>
              <Button type="button" variant="secondary" onClick={handleNewChat}>
                <Plus className="h-4 w-4" />
                New chat
              </Button>
              <Button
                type="button"
                variant="secondary"
                size="icon"
                className="xl:hidden"
                onClick={() => setMobileVisualOpen(true)}
              >
                <PanelRightOpen className="h-4 w-4" />
              </Button>
              <Button
                type="button"
                variant="secondary"
                size="icon"
                className="hidden xl:inline-flex"
                onClick={() => setVisualizationOpen(!visualizationOpen)}
              >
                <PanelRightOpen className="h-4 w-4" />
              </Button>
              <ThemeToggle />
            </div>
          </div>
          <div className="mx-auto hidden w-full max-w-[1800px] items-center gap-3 px-4 pb-3 lg:flex xl:hidden">
            <ModelSelector
              value={selectedModel}
              onChange={(model, provider) => updateSelectedModel(model, provider)}
            />
            <Badge variant="secondary">{activeModel.description}</Badge>
          </div>
        </header>

        <div
          className={`mx-auto grid w-full max-w-[1800px] flex-1 gap-4 p-4 lg:min-h-0 lg:grid-cols-[280px_minmax(0,1fr)] lg:overflow-hidden ${
            visualizationOpen
              ? "xl:grid-cols-[280px_minmax(0,1fr)_minmax(0,1fr)]"
              : "xl:grid-cols-[280px_minmax(0,1fr)]"
          }`}
        >
          <div className="hidden h-full min-h-0 lg:block">
            <Sidebar
              sessions={sortedSessions}
              activeSessionId={activeSession?.id ?? null}
              onNewChat={handleNewChat}
              onSelectSession={(id) => {
                startTransition(() => setActiveSession(id));
              }}
              onDeleteSession={deleteSession}
              onTopicClick={handleTopicClick}
            />
          </div>

          <div className="min-w-0 lg:h-full lg:min-h-0">
            <Chat
              activeModelLabel={activeModel.label}
              agentRuntime={agentRuntime}
              input={input}
              isListening={isListening}
              isLoading={isLoading}
              isSpeaking={isSpeaking}
              transcriptPreview={transcriptPreview}
              session={activeSession}
              onInputChange={setInput}
              onMicClick={toggleListening}
              onSubmit={handleSubmit}
              onToggleSaved={() => activeSession && toggleSaved(activeSession.id)}
            />
          </div>

          {visualizationOpen ? (
            <div className="hidden min-w-0 xl:block xl:h-full xl:min-h-0">
              <VisualizationPanel
                key={visualizationKey}
                visual={latestVisual}
                visualAgentState={agentRuntime.visual}
                onExplainNode={handleExplainNode}
                onRegenerateVisualization={handleRegenerateVisualization}
              />
            </div>
          ) : null}
        </div>

        <div className="pointer-events-none fixed bottom-4 left-4 z-20 hidden gap-2 md:flex">
          <Badge variant="secondary">{NEW_CHAT_SHORTCUT}</Badge>
          <Badge variant="secondary">{VOICE_SHORTCUT}</Badge>
        </div>

        <Sheet open={mobileSidebarOpen} onOpenChange={setMobileSidebarOpen}>
          <SheetContent side="left" className="p-0">
            <div className="flex h-full flex-col p-4">
              <SheetHeader>
                <SheetTitle>StudyGPT history</SheetTitle>
                <SheetDescription>
                  Switch chats, pin good lessons, or launch a suggested topic.
                </SheetDescription>
              </SheetHeader>
              <div className="mb-4 md:hidden">
                <ApiKeysDialog apiKeys={apiKeys} onChange={updateApiKey} />
              </div>
              <Sidebar
                sessions={sortedSessions}
                activeSessionId={activeSession?.id ?? null}
                onNewChat={handleNewChat}
                onSelectSession={(id) => {
                  setActiveSession(id);
                  setMobileSidebarOpen(false);
                }}
                onDeleteSession={deleteSession}
                onTopicClick={handleTopicClick}
              />
            </div>
          </SheetContent>
        </Sheet>

        <Sheet open={mobileVisualOpen} onOpenChange={setMobileVisualOpen}>
          <SheetContent side="right" className="w-[min(96vw,34rem)] p-4">
            <SheetHeader>
              <SheetTitle>Visualization</SheetTitle>
              <SheetDescription>
                Interactive mind maps and charts follow the active conversation.
              </SheetDescription>
            </SheetHeader>
            <VisualizationPanel
              key={`${visualizationKey}-mobile`}
              visual={latestVisual}
              visualAgentState={agentRuntime.visual}
              onExplainNode={handleExplainNode}
              onRegenerateVisualization={handleRegenerateVisualization}
            />
          </SheetContent>
        </Sheet>
      </div>
    </div>
  );
}
