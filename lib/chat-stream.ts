import type { ChatStreamEvent } from "@/lib/types";

export function formatChatStreamEvent(event: ChatStreamEvent) {
  return `data: ${JSON.stringify(event)}\n\n`;
}

export function parseChatStreamBuffer(buffer: string) {
  const blocks = buffer.split("\n\n");
  const remaining = blocks.pop() ?? "";
  const events: ChatStreamEvent[] = [];

  for (const block of blocks) {
    const dataLine = block
      .split("\n")
      .find((line) => line.startsWith("data: "));

    if (!dataLine) {
      continue;
    }

    try {
      const parsed = JSON.parse(dataLine.slice(6)) as ChatStreamEvent;
      events.push(parsed);
    } catch {
      continue;
    }
  }

  return {
    events,
    remaining,
  };
}
