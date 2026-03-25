"use client";

import { useEffect } from "react";

interface KeyboardShortcutOptions {
  onNewChat: () => void;
  onToggleVoice: () => void;
}

export function useKeyboardShortcuts({
  onNewChat,
  onToggleVoice,
}: KeyboardShortcutOptions) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const modifierPressed = event.metaKey || event.ctrlKey;
      if (!modifierPressed) {
        return;
      }

      const key = event.key.toLowerCase();
      if (key === "k" && !event.shiftKey) {
        event.preventDefault();
        onNewChat();
        return;
      }

      if (key === "m" && event.shiftKey) {
        event.preventDefault();
        onToggleVoice();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onNewChat, onToggleVoice]);
}
