"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  DEFAULT_MODEL,
  DEFAULT_VOICE_SETTINGS,
  STUDYGPT_STORAGE_KEY,
} from "@/lib/constants";
import type {
  ChatMessage,
  ChatSession,
  ProviderId,
  ProviderKeys,
  VoiceSettings,
} from "@/lib/types";
import { createId, nowIso, titleFromPrompt } from "@/lib/utils";

interface StudyGPTState {
  sessions: ChatSession[];
  activeSessionId: string | null;
  selectedProvider: ProviderId;
  selectedModel: string;
  apiKeys: ProviderKeys;
  voiceSettings: VoiceSettings;
  visualizationOpen: boolean;
  hasHydrated: boolean;
  createSession: (seedPrompt?: string) => string;
  setActiveSession: (id: string) => void;
  setHasHydrated: (value: boolean) => void;
  updateSelectedModel: (model: string, provider: ProviderId) => void;
  updateApiKey: (provider: ProviderId, value: string) => void;
  updateVoiceSettings: (settings: Partial<VoiceSettings>) => void;
  setVisualizationOpen: (value: boolean) => void;
  appendMessage: (sessionId: string, message: ChatMessage) => void;
  updateMessage: (
    sessionId: string,
    messageId: string,
    updates: Partial<ChatMessage>,
  ) => void;
  toggleSaved: (sessionId: string) => void;
  deleteSession: (sessionId: string) => void;
}

function createSessionRecord(seedPrompt?: string): ChatSession {
  const timestamp = nowIso();
  return {
    id: createId(),
    title: seedPrompt ? titleFromPrompt(seedPrompt) : "New chat",
    createdAt: timestamp,
    updatedAt: timestamp,
    provider: DEFAULT_MODEL.provider,
    model: DEFAULT_MODEL.id,
    messages: [],
    saved: false,
  };
}

export const useStudyGPTStore = create<StudyGPTState>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,
      selectedProvider: DEFAULT_MODEL.provider,
      selectedModel: DEFAULT_MODEL.id,
      apiKeys: {
        groq: "",
        openai: "",
        anthropic: "",
        gemini: "",
      },
      voiceSettings: DEFAULT_VOICE_SETTINGS,
      visualizationOpen: true,
      hasHydrated: false,
      createSession: (seedPrompt) => {
        const session = createSessionRecord(seedPrompt);
        set((state) => ({
          sessions: [session, ...state.sessions],
          activeSessionId: session.id,
        }));
        return session.id;
      },
      setActiveSession: (id) => {
        set({ activeSessionId: id });
      },
      setHasHydrated: (value) => {
        set({ hasHydrated: value });
      },
      updateSelectedModel: (model, provider) => {
        set({ selectedModel: model, selectedProvider: provider });
      },
      updateApiKey: (provider, value) => {
        set((state) => ({
          apiKeys: {
            ...state.apiKeys,
            [provider]: value,
          },
        }));
      },
      updateVoiceSettings: (settings) => {
        set((state) => ({
          voiceSettings: {
            ...state.voiceSettings,
            ...settings,
          },
        }));
      },
      setVisualizationOpen: (value) => {
        set({ visualizationOpen: value });
      },
      appendMessage: (sessionId, message) => {
        set((state) => ({
          sessions: state.sessions.map((session) => {
            if (session.id !== sessionId) {
              return session;
            }

            const messages = [...session.messages, message];
            const firstUserMessage = messages.find((entry) => entry.role === "user");

            return {
              ...session,
              title:
                session.title === "New chat" && firstUserMessage
                  ? titleFromPrompt(firstUserMessage.content)
                  : session.title,
              updatedAt: nowIso(),
              provider: get().selectedProvider,
              model: get().selectedModel,
              messages,
            };
          }),
        }));
      },
      updateMessage: (sessionId, messageId, updates) => {
        set((state) => ({
          sessions: state.sessions.map((session) => {
            if (session.id !== sessionId) {
              return session;
            }

            return {
              ...session,
              updatedAt: nowIso(),
              messages: session.messages.map((message) =>
                message.id === messageId ? { ...message, ...updates } : message,
              ),
            };
          }),
        }));
      },
      toggleSaved: (sessionId) => {
        set((state) => ({
          sessions: state.sessions.map((session) =>
            session.id === sessionId
              ? { ...session, saved: !session.saved, updatedAt: nowIso() }
              : session,
          ),
        }));
      },
      deleteSession: (sessionId) => {
        set((state) => {
          const remaining = state.sessions.filter((session) => session.id !== sessionId);
          if (remaining.length === 0) {
            const nextSession = createSessionRecord();
            return {
              sessions: [nextSession],
              activeSessionId: nextSession.id,
            };
          }

          return {
            sessions: remaining,
            activeSessionId:
              state.activeSessionId === sessionId
                ? remaining[0]?.id ?? null
                : state.activeSessionId,
          };
        });
      },
    }),
    {
      name: STUDYGPT_STORAGE_KEY,
      version: 1,
      onRehydrateStorage: () => (state) => {
        state?.setHasHydrated(true);
      },
      partialize: (state) => ({
        sessions: state.sessions,
        activeSessionId: state.activeSessionId,
        selectedProvider: state.selectedProvider,
        selectedModel: state.selectedModel,
        apiKeys: state.apiKeys,
        voiceSettings: state.voiceSettings,
        visualizationOpen: state.visualizationOpen,
      }),
    },
  ),
);
