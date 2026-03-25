"use client";

import { useDeferredValue, useMemo, useState } from "react";
import { BookOpenText, Clock3, Plus, Star, Trash2 } from "lucide-react";
import { SUGGESTED_TOPICS } from "@/lib/constants";
import type { ChatSession } from "@/lib/types";
import { formatRelativeTime } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

interface SidebarProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onNewChat: () => void;
  onSelectSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onTopicClick: (prompt: string) => void;
}

export function Sidebar({
  sessions,
  activeSessionId,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  onTopicClick,
}: SidebarProps) {
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query);

  const filteredSessions = useMemo(() => {
    const normalized = deferredQuery.trim().toLowerCase();
    if (!normalized) {
      return sessions;
    }

    return sessions.filter((session) =>
      session.title.toLowerCase().includes(normalized),
    );
  }, [deferredQuery, sessions]);

  const savedTopics = sessions.filter((session) => session.saved);

  return (
    <Card className="flex h-full min-h-0 flex-col overflow-hidden">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between gap-3">
          <CardTitle className="text-base">Workspace</CardTitle>
          <Button type="button" size="sm" onClick={onNewChat}>
            <Plus className="h-4 w-4" />
            New chat
          </Button>
        </div>
        <Input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search history"
        />
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col gap-5 overflow-hidden">
        <section className="min-h-0 flex-1 overflow-auto pr-1">
          <div className="mb-3 flex items-center gap-2">
            <Clock3 className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-semibold">Chat history</h3>
          </div>
          <div className="space-y-2">
            {filteredSessions.map((session) => (
              <div
                key={session.id}
                className={`group rounded-[1.35rem] border p-3 transition-colors ${
                  session.id === activeSessionId
                    ? "border-primary/25 bg-primary/10"
                    : "border-border bg-muted/35 hover:bg-muted/55"
                }`}
              >
                <button
                  type="button"
                  onClick={() => onSelectSession(session.id)}
                  className="w-full text-left"
                >
                  <div className="flex items-center justify-between gap-2">
                    <p className="line-clamp-2 text-sm font-medium">{session.title}</p>
                    {session.saved ? <Star className="h-3.5 w-3.5 text-accent" /> : null}
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    {formatRelativeTime(session.updatedAt)}
                  </p>
                </button>
                <div className="mt-3 flex justify-end">
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => onDeleteSession(session.id)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section>
          <div className="mb-3 flex items-center gap-2">
            <Star className="h-4 w-4 text-accent" />
            <h3 className="text-sm font-semibold">Saved topics</h3>
          </div>
          <div className="flex flex-wrap gap-2">
            {savedTopics.length > 0 ? (
              savedTopics.map((session) => (
                <Badge
                  key={session.id}
                  variant="secondary"
                  className="cursor-pointer normal-case tracking-normal"
                  onClick={() => onSelectSession(session.id)}
                >
                  {session.title}
                </Badge>
              ))
            ) : (
              <p className="text-xs text-muted-foreground">
                Star a conversation to pin it here.
              </p>
            )}
          </div>
        </section>

        <section>
          <div className="mb-3 flex items-center gap-2">
            <BookOpenText className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-semibold">Suggested topics</h3>
          </div>
          <div className="flex flex-wrap gap-2">
            {SUGGESTED_TOPICS.map((topic) => (
              <button
                key={topic}
                type="button"
                onClick={() => onTopicClick(topic)}
                className="rounded-full border border-border bg-muted/45 px-3 py-2 text-left text-xs font-medium text-foreground transition-colors hover:bg-muted"
              >
                {topic}
              </button>
            ))}
          </div>
        </section>
      </CardContent>
    </Card>
  );
}
