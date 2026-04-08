"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { VoiceSettings } from "@/lib/types";
import { sanitizeAssistantContent } from "@/lib/utils";

interface UseVoiceTutorOptions {
  settings: VoiceSettings;
  onTranscript: (value: string) => void;
  onError?: (message: string) => void;
}

interface StopListeningOptions {
  auto?: boolean;
  preserveTranscript?: boolean;
}

export function useVoiceTutor({
  settings,
  onTranscript,
  onError,
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
  const isRecognitionSupported =
    typeof window !== "undefined" &&
    !!(window.SpeechRecognition ?? window.webkitSpeechRecognition);
  const isSpeechSynthesisSupported =
    typeof window !== "undefined" &&
    typeof window.speechSynthesis !== "undefined" &&
    typeof SpeechSynthesisUtterance !== "undefined";
  const isEmbeddedFrame =
    typeof window !== "undefined" && window.self !== window.top;
  const isSupported = isRecognitionSupported || isSpeechSynthesisSupported;

  const emitError = useCallback(
    (message: string) => {
      onError?.(message);
    },
    [onError],
  );

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
    if (!isRecognitionSupported) {
      emitError(
        isEmbeddedFrame
          ? "Voice input is not available in this embedded preview. Open the app directly in a new tab and allow microphone access."
          : "This browser does not support speech recognition.",
      );
      return false;
    }

    if (!recognitionRef.current || speakingRef.current) {
      return false;
    }

    if (typeof window !== "undefined" && !window.isSecureContext) {
      emitError("Voice input requires a secure HTTPS context.");
      return false;
    }

    transcriptRef.current = "";
    setTranscriptPreview("");
    autoRestartRef.current = false;

    try {
      recognitionRef.current.continuous = settingsRef.current.mode === "continuous";
      recognitionRef.current.interimResults = true;
      recognitionRef.current.start();
      return true;
    } catch {
      emitError(
        isEmbeddedFrame
          ? "Microphone access was blocked in the embedded app. Open the direct app tab and allow microphone access."
          : "StudyGPT could not start microphone capture. Check browser microphone permission and try again.",
      );
      return false;
    }
  }, [emitError, isEmbeddedFrame, isRecognitionSupported]);

  const toggleListening = useCallback(() => {
    if (listeningRef.current) {
      stopListening();
    } else {
      startListening();
    }
  }, [startListening, stopListening]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

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

    recognition.onerror = (event) => {
      setIsListening(false);

      const recognitionEvent = event as SpeechRecognitionErrorEvent;
      const errorCode = recognitionEvent.error;
      const message =
        errorCode === "not-allowed" || errorCode === "service-not-allowed"
          ? isEmbeddedFrame
            ? "Microphone permission was blocked in the embedded app. Open the direct app tab and allow microphone access."
            : "Microphone permission was denied. Allow microphone access in your browser and try again."
          : errorCode === "audio-capture"
            ? "No working microphone was detected."
            : errorCode === "language-not-supported"
              ? "This browser does not support the selected speech recognition language."
              : errorCode === "network"
                ? "Speech recognition hit a network error. Try again."
                : errorCode === "no-speech"
                  ? "No speech was detected. Try speaking again."
                  : recognitionEvent.message || "Speech recognition failed.";

      emitError(message);
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
  }, [
    commitTranscript,
    emitError,
    isEmbeddedFrame,
    settings.mode,
    startListening,
    stopListening,
  ]);

  useEffect(() => {
    if (typeof window === "undefined" || !isSpeechSynthesisSupported) {
      return;
    }

    const speechSynthesisApi = window.speechSynthesis;

    const loadVoices = () => {
      setVoices(speechSynthesisApi.getVoices());
    };

    loadVoices();
    speechSynthesisApi.addEventListener("voiceschanged", loadVoices);

    return () => {
      speechSynthesisApi.removeEventListener("voiceschanged", loadVoices);
    };
  }, [isSpeechSynthesisSupported]);

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
    if (
      !settingsRef.current.enabled ||
      !settingsRef.current.autoSpeak ||
      !isSpeechSynthesisSupported
    ) {
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
  }, [isSpeechSynthesisSupported, startListening, stopListening, voices]);

  const stopSpeaking = useCallback(() => {
    if (!isSpeechSynthesisSupported) {
      return;
    }

    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  }, [isSpeechSynthesisSupported]);

  return {
    availableVoices,
    isEmbeddedFrame,
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
