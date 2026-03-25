"use client";

import { useEffect, useMemo, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Bot,
  MessagesSquare,
  Mic,
  SendHorizonal,
  Sparkles,
  Star,
  Volume2,
  Waypoints,
} from "lucide-react";
import { AGENT_DISPLAY_NAMES } from "@/lib/constants";
import type { AgentRuntimeMap, ChatSession } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";

interface ChatProps {
  activeModelLabel: string;
  input: string;
  agentRuntime: AgentRuntimeMap;
  isListening: boolean;
  isLoading: boolean;
  isSpeaking: boolean;
  transcriptPreview: string;
  session: ChatSession | null;
  onInputChange: (value: string) => void;
  onMicClick: () => void;
  onSubmit: () => void;
  onToggleSaved: () => void;
}

export function Chat({
  activeModelLabel,
  agentRuntime,
  input,
  isListening,
  isLoading,
  isSpeaking,
  transcriptPreview,
  session,
  onInputChange,
  onMicClick,
  onSubmit,
  onToggleSaved,
}: ChatProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const latestAssistantId = useMemo(
    () =>
      [...(session?.messages ?? [])]
        .reverse()
        .find((message) => message.role === "assistant")?.id ?? null,
    [session?.messages],
  );

  useEffect(() => {
    if (!scrollRef.current) {
      return;
    }

    scrollRef.current.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [session?.messages, transcriptPreview]);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSubmit();
    }
  };

  return (
    <Card className="flex h-full min-h-0 flex-col overflow-hidden">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between gap-4">
          <div className="min-w-0">
            <CardTitle className="truncate text-xl">
              {session?.title ?? "StudyGPT"}
            </CardTitle>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Badge>{activeModelLabel}</Badge>
              <Badge variant="accent">{isSpeaking ? "Speaking" : "Ready"}</Badge>
              <Badge variant="secondary">3-agent mode</Badge>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <AgentBadge
                icon={<MessagesSquare className="h-3.5 w-3.5" />}
                label={AGENT_DISPLAY_NAMES.text}
                status={agentRuntime.text.status}
              />
              <AgentBadge
                icon={<Waypoints className="h-3.5 w-3.5" />}
                label={AGENT_DISPLAY_NAMES.visual}
                status={agentRuntime.visual.status}
              />
              <AgentBadge
                icon={<Volume2 className="h-3.5 w-3.5" />}
                label={AGENT_DISPLAY_NAMES.voice}
                status={agentRuntime.voice.status}
              />
            </div>
          </div>
          <Button type="button" variant="secondary" onClick={onToggleSaved}>
            <Star className="h-4 w-4" />
            {session?.saved ? "Saved" : "Save topic"}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col gap-4">
        <div
          ref={scrollRef}
          className="grid-noise min-h-0 flex-1 overflow-auto rounded-[1.6rem] border border-border bg-background/45 p-4"
        >
          {session?.messages.length ? (
            <AnimatePresence initial={false}>
              <div className="space-y-4">
                {session.messages.map((message) => {
                  const isAssistant = message.role === "assistant";
                  const isLatestSpeaking = isAssistant && latestAssistantId === message.id && isSpeaking;
                  return (
                    <motion.div
                      key={message.id}
                      layout
                      initial={{ opacity: 0, y: 12 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -8 }}
                      className={`flex gap-3 ${
                        isAssistant ? "items-start" : "justify-end"
                      }`}
                    >
                      {isAssistant ? (
                        <div
                          className={`animated-orb flex h-11 w-11 shrink-0 items-center justify-center rounded-full ${
                            isLatestSpeaking
                              ? "bg-primary text-primary-foreground"
                              : "bg-secondary text-secondary-foreground"
                          }`}
                        >
                          <Bot className="h-5 w-5" />
                        </div>
                      ) : null}
                      <div
                        className={`max-w-[min(100%,52rem)] rounded-[1.7rem] border px-4 py-3 shadow-sm ${
                          isAssistant
                            ? "border-border bg-card text-card-foreground"
                            : "border-primary/20 bg-primary text-primary-foreground"
                        }`}
                      >
                        {message.status === "streaming" && !message.content ? (
                          <ThinkingDots />
                        ) : (
                          <p className="whitespace-pre-wrap text-sm leading-7">
                            {message.content}
                          </p>
                        )}
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </AnimatePresence>
          ) : (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <div className="animated-orb mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary text-primary-foreground">
                <Sparkles className="h-7 w-7" />
              </div>
              <h2 className="font-heading text-2xl font-semibold">Live virtual tutor</h2>
              <p className="mt-3 max-w-xl text-sm text-muted-foreground">
                Ask StudyGPT to teach any topic, then let the right panel build a mind
                map or chart as the explanation unfolds.
              </p>
            </div>
          )}
        </div>

        {transcriptPreview ? (
          <div className="rounded-[1.4rem] border border-primary/20 bg-primary/8 px-4 py-3 text-sm text-primary">
            Listening: {transcriptPreview}
          </div>
        ) : null}

        {isLoading ? (
          <div className="rounded-[1.4rem] border border-border bg-muted/35 px-4 py-3 text-sm text-muted-foreground">
            {agentRuntime.text.message}. {agentRuntime.visual.message}. {agentRuntime.voice.message}.
          </div>
        ) : null}

        <Separator />

        <div className="rounded-[1.7rem] border border-border bg-card p-3">
          <Textarea
            value={input}
            onChange={(event) => onInputChange(event.target.value)}
            onKeyDown={handleKeyDown}
            rows={4}
            className="min-h-[8rem] border-0 bg-transparent px-2 shadow-none focus-visible:ring-0"
            placeholder="Ask anything. Try: 'Explain neural networks like I'm 12 and show a mind map.'"
          />
          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs text-muted-foreground">
              Press Enter to send, Shift + Enter for a new line.
            </p>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="secondary"
                size="icon"
                onClick={onMicClick}
                className={isListening ? "bg-primary text-primary-foreground" : ""}
              >
                <Mic className="h-4 w-4" />
              </Button>
              <Button type="button" onClick={onSubmit} disabled={isLoading || !input.trim()}>
                <SendHorizonal className="h-4 w-4" />
                Send
              </Button>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function AgentBadge({
  icon,
  label,
  status,
}: {
  icon: React.ReactNode;
  label: string;
  status: AgentRuntimeMap["text"]["status"];
}) {
  const variant =
    status === "running"
      ? "default"
      : status === "done"
        ? "accent"
        : status === "error"
          ? "outline"
          : "secondary";

  return (
    <Badge variant={variant} className="gap-1.5 normal-case tracking-normal">
      {icon}
      {label}: {status}
    </Badge>
  );
}

function ThinkingDots() {
  return (
    <div className="flex items-center gap-1.5">
      {[0, 1, 2].map((index) => (
        <motion.span
          key={index}
          className="h-2.5 w-2.5 rounded-full bg-primary"
          animate={{ opacity: [0.25, 1, 0.25], y: [0, -3, 0] }}
          transition={{ repeat: Infinity, duration: 1.2, delay: index * 0.12 }}
        />
      ))}
    </div>
  );
}
