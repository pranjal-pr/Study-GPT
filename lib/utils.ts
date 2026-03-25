import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function createId() {
  return crypto.randomUUID();
}

export function nowIso() {
  return new Date().toISOString();
}

export function truncate(text: string, maxLength = 48) {
  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength - 1).trim()}…`;
}

export function titleFromPrompt(prompt: string) {
  return truncate(prompt.replace(/\s+/g, " ").trim() || "New chat");
}

export function formatRelativeTime(value: string) {
  const now = Date.now();
  const target = new Date(value).getTime();
  const diffMinutes = Math.round((target - now) / 60000);
  const formatter = new Intl.RelativeTimeFormat(undefined, {
    numeric: "auto",
  });

  if (Math.abs(diffMinutes) < 60) {
    return formatter.format(diffMinutes, "minute");
  }

  const diffHours = Math.round(diffMinutes / 60);
  if (Math.abs(diffHours) < 24) {
    return formatter.format(diffHours, "hour");
  }

  const diffDays = Math.round(diffHours / 24);
  return formatter.format(diffDays, "day");
}

export function sanitizeAssistantContent(content: string) {
  return content.replace(/```visual[\s\S]*$/i, "").trim();
}

export function downloadDataUrl(dataUrl: string, fileName: string) {
  const anchor = document.createElement("a");
  anchor.href = dataUrl;
  anchor.download = fileName;
  anchor.click();
}
