"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { VoiceSettings } from "@/lib/types";
import { sanitizeAssistantContent } from "@/lib/utils";

interface UseVoiceTutorOptions {
  settings: VoiceSettings;
  onTranscript: (value: string) => void;
}

interface StopListeningOptions {
  auto?: boolean;
  preserveTranscript?: boolean;
}

export function useVoiceTutor({
  settings,
  onTranscript,
}: UseVoiceTutorOptions) {
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const transcriptRef = useRef("");
  const autoRestartRef = useRef(false);
  const settingsRef = useRef(settings);
  const listeningRef = useRef(false);
  const speakingRef = useRef(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [transcriptPreview, setTranscriptPreview] = useState("");
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const isSupported =
    typeof window !== "undefined" &&
    !!(window.SpeechRecognition ?? window.webkitSpeechRecognition);

  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);

  useEffect(() => {
    listeningRef.current = isListening;
  }, [isListening]);

  useEffect(() => {
    speakingRef.current = isSpeaking;
  }, [isSpeaking]);

  const commitTranscript = useCallback(() => {
    const finalTranscript = transcriptRef.current.trim();
    transcriptRef.current = "";
    setTranscriptPreview("");

    if (finalTranscript) {
      onTranscript(finalTranscript);
    }
  }, [onTranscript]);

  const stopListening = useCallback(
    ({ auto = false, preserveTranscript = false }: StopListeningOptions = {}) => {
      autoRestartRef.current = auto;
      if (!preserveTranscript) {
        transcriptRef.current = "";
        setTranscriptPreview("");
      }

      if (!recognitionRef.current) {
        setIsListening(false);
        return;
      }

      try {
        recognitionRef.current.stop();
      } catch {
        setIsListening(false);
      }
    },
    [],
  );

  const startListening = useCallback(() => {
    if (!recognitionRef.current || speakingRef.current) {
      return;
    }

    transcriptRef.current = "";
    setTranscriptPreview("");
    autoRestartRef.current = false;

    try {
      recognitionRef.current.continuous = settingsRef.current.mode === "continuous";
      recognitionRef.current.interimResults = true;
      recognitionRef.current.start();
    } catch {
      // Ignore duplicate browser start errors.
    }
  }, []);

  const toggleListening = useCallback(() => {
    if (listeningRef.current) {
      stopListening();
    } else {
      startListening();
    }
  }, [startListening, stopListening]);

  useEffect(() => {
    const RecognitionConstructor =
      window.SpeechRecognition ?? window.webkitSpeechRecognition;

    if (!RecognitionConstructor) {
      return;
    }

    const recognition = new RecognitionConstructor();
    recognition.lang = "en-US";
    recognition.maxAlternatives = 1;
    recognition.continuous = settings.mode === "continuous";
    recognition.interimResults = true;

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onresult = (event) => {
      let interim = "";
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        const transcript = result[0]?.transcript ?? "";

        if (result.isFinal) {
          transcriptRef.current = `${transcriptRef.current} ${transcript}`.trim();
        } else {
          interim += transcript;
        }
      }

      setTranscriptPreview(`${transcriptRef.current} ${interim}`.trim());

      if (settingsRef.current.mode === "push") {
        const latestResult = event.results[event.results.length - 1];
        if (latestResult?.isFinal) {
          commitTranscript();
          stopListening();
        }
      }
    };

    recognition.onerror = () => {
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);

      if (settingsRef.current.mode === "push") {
        return;
      }

      if (autoRestartRef.current) {
        autoRestartRef.current = false;
        return;
      }

      if (transcriptRef.current.trim()) {
        commitTranscript();
      }

      if (
        settingsRef.current.enabled &&
        settingsRef.current.mode === "continuous" &&
        !speakingRef.current
      ) {
        startListening();
      }
    };

    recognitionRef.current = recognition;

    return () => {
      recognition.onstart = null;
      recognition.onresult = null;
      recognition.onerror = null;
      recognition.onend = null;
      try {
        recognition.stop();
      } catch {
        // Ignore teardown errors.
      }
      recognitionRef.current = null;
    };
  }, [commitTranscript, settings.mode, startListening, stopListening]);

  useEffect(() => {
    const loadVoices = () => {
      setVoices(window.speechSynthesis.getVoices());
    };

    loadVoices();
    window.speechSynthesis.addEventListener("voiceschanged", loadVoices);

    return () => {
      window.speechSynthesis.removeEventListener("voiceschanged", loadVoices);
    };
  }, []);

  useEffect(() => {
    if (!settings.enabled || settings.mode !== "continuous") {
      return;
    }

    if (!isListening && !isSpeaking) {
      const timeout = window.setTimeout(() => {
        startListening();
      }, 0);

      return () => {
        window.clearTimeout(timeout);
      };
    }
  }, [isListening, isSpeaking, settings.enabled, settings.mode, startListening]);

  const availableVoices = useMemo(
    () =>
      voices.filter((voice) =>
        /en|hi/i.test(voice.lang) || /Google|Microsoft|Apple/i.test(voice.name),
      ),
    [voices],
  );

  const speak = useCallback(async (value: string) => {
    if (!settingsRef.current.enabled || !settingsRef.current.autoSpeak) {
      return;
    }

    const cleanText = sanitizeAssistantContent(value);
    if (!cleanText) {
      return;
    }

    if (listeningRef.current) {
      stopListening({ auto: true, preserveTranscript: true });
    }

    window.speechSynthesis.cancel();

    await new Promise<void>((resolve) => {
      const utterance = new SpeechSynthesisUtterance(cleanText);
      const selectedVoice = voices.find(
        (voice) => voice.voiceURI === settingsRef.current.voiceURI,
      );

      if (selectedVoice) {
        utterance.voice = selectedVoice;
      }

      utterance.rate = settingsRef.current.rate;
      utterance.pitch = settingsRef.current.pitch;
      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => {
        setIsSpeaking(false);
        resolve();

        if (settingsRef.current.enabled && settingsRef.current.mode === "continuous") {
          startListening();
        }
      };
      utterance.onerror = () => {
        setIsSpeaking(false);
        resolve();
      };

      window.speechSynthesis.speak(utterance);
    });
  }, [startListening, stopListening, voices]);

  const stopSpeaking = useCallback(() => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  }, []);

  return {
    availableVoices,
    isListening,
    isSpeaking,
    isSupported,
    transcriptPreview,
    speak,
    startListening,
    stopListening,
    stopSpeaking,
    toggleListening,
  };
}
