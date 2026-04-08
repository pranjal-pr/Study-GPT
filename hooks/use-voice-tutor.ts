"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { VoiceSettings } from "@/lib/types";
import { sanitizeAssistantContent } from "@/lib/utils";

interface UseVoiceTutorOptions {
  settings: VoiceSettings;
  onTranscript: (value: string) => void;
  onError?: (message: string) => void;
  transcribeAudio?: (blob: Blob) => Promise<string>;
}

interface StopListeningOptions {
  auto?: boolean;
  preserveTranscript?: boolean;
}

export function useVoiceTutor({
  settings,
  onTranscript,
  onError,
  transcribeAudio,
}: UseVoiceTutorOptions) {
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const discardRecordingRef = useRef(false);
  const fallbackNoticeShownRef = useRef(false);
  const frameRef = useRef<number | null>(null);
  const lastSpeechAtRef = useRef(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const recordingStartedAtRef = useRef(0);
  const speechDetectedRef = useRef(false);
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
  const isMediaRecordingSupported =
    typeof window !== "undefined" &&
    typeof MediaRecorder !== "undefined" &&
    typeof navigator !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia;
  const isSpeechSynthesisSupported =
    typeof window !== "undefined" &&
    typeof window.speechSynthesis !== "undefined" &&
    typeof SpeechSynthesisUtterance !== "undefined";
  const isEmbeddedFrame =
    typeof window !== "undefined" && window.self !== window.top;
  const isSupported =
    isRecognitionSupported || isMediaRecordingSupported || isSpeechSynthesisSupported;

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

  const stopMediaTracks = useCallback(() => {
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;
  }, []);

  const stopVolumeMonitoring = useCallback(() => {
    if (frameRef.current !== null) {
      window.cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }

    analyserRef.current = null;

    if (audioContextRef.current) {
      void audioContextRef.current.close().catch(() => {
        // Ignore close errors during teardown.
      });
      audioContextRef.current = null;
    }
  }, []);

  const startVolumeMonitoring = useCallback(
    (stream: MediaStream, recorder: MediaRecorder) => {
      if (typeof window === "undefined") {
        return;
      }

      const AudioContextCtor =
        window.AudioContext ||
        (window as Window & typeof globalThis & { webkitAudioContext?: typeof AudioContext })
          .webkitAudioContext;
      if (!AudioContextCtor) {
        return;
      }

      const context = new AudioContextCtor();
      const source = context.createMediaStreamSource(stream);
      const analyser = context.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);

      audioContextRef.current = context;
      analyserRef.current = analyser;

      const buffer = new Uint8Array(analyser.frequencyBinCount);
      const silenceThreshold = 0.02;
      const initialSilenceMs = 3500;
      const silenceAfterSpeechMs = 1200;
      const minRecordingMs = 700;
      const maxRecordingMs = settingsRef.current.mode === "continuous" ? 15000 : 20000;

      const tick = () => {
        if (mediaRecorderRef.current !== recorder || recorder.state === "inactive") {
          stopVolumeMonitoring();
          return;
        }

        analyser.getByteTimeDomainData(buffer);

        let sum = 0;
        for (const value of buffer) {
          const normalized = (value - 128) / 128;
          sum += normalized * normalized;
        }

        const rms = Math.sqrt(sum / buffer.length);
        const now = performance.now();
        const elapsed = now - recordingStartedAtRef.current;

        if (rms > silenceThreshold) {
          speechDetectedRef.current = true;
          lastSpeechAtRef.current = now;
          setTranscriptPreview("Listening...");
        }

        const shouldStopForSilence =
          speechDetectedRef.current &&
          elapsed >= minRecordingMs &&
          now - lastSpeechAtRef.current >= silenceAfterSpeechMs;
        const shouldStopForNoSpeech =
          !speechDetectedRef.current && elapsed >= initialSilenceMs;
        const shouldStopForMaxLength = elapsed >= maxRecordingMs;

        if (shouldStopForSilence || shouldStopForNoSpeech || shouldStopForMaxLength) {
          recorder.stop();
          return;
        }

        frameRef.current = window.requestAnimationFrame(tick);
      };

      frameRef.current = window.requestAnimationFrame(tick);
    },
    [stopVolumeMonitoring],
  );

  const stopListening = useCallback(
    ({ auto = false, preserveTranscript = false }: StopListeningOptions = {}) => {
      autoRestartRef.current = auto;

      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        discardRecordingRef.current = !preserveTranscript;
        mediaRecorderRef.current.stop();
        if (!preserveTranscript) {
          transcriptRef.current = "";
          setTranscriptPreview("");
        }
        return;
      }

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

  const startAudioRecording = useCallback(async () => {
    if (!isMediaRecordingSupported || !transcribeAudio) {
      emitError("Voice recording is not supported in this browser.");
      return false;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });
      mediaStreamRef.current = stream;

      const mimeType =
        [
          "audio/webm;codecs=opus",
          "audio/webm",
          "audio/mp4",
          "audio/ogg;codecs=opus",
        ].find((candidate) => MediaRecorder.isTypeSupported(candidate)) ?? "";

      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);

      audioChunksRef.current = [];
      discardRecordingRef.current = false;
      fallbackNoticeShownRef.current = true;
      lastSpeechAtRef.current = 0;
      recordingStartedAtRef.current = performance.now();
      speechDetectedRef.current = false;
      recorder.onstart = () => {
        setIsListening(true);
        setTranscriptPreview(
          settingsRef.current.mode === "continuous" ? "Listening..." : "Recording...",
        );
      };
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      recorder.onerror = () => {
        setIsListening(false);
        setTranscriptPreview("");
        stopVolumeMonitoring();
        stopMediaTracks();
        emitError("Audio recording failed. Please try again.");
      };
      recorder.onstop = async () => {
        setIsListening(false);
        stopVolumeMonitoring();

        const chunks = [...audioChunksRef.current];
        audioChunksRef.current = [];
        mediaRecorderRef.current = null;
        stopMediaTracks();

        if (discardRecordingRef.current) {
          discardRecordingRef.current = false;
          setTranscriptPreview("");
          return;
        }

        if (chunks.length === 0) {
          setTranscriptPreview("");
          return;
        }

        setTranscriptPreview("Transcribing...");

        try {
          const blob = new Blob(chunks, {
            type: recorder.mimeType || "audio/webm",
          });
          const transcript = (await transcribeAudio(blob)).trim();

          setTranscriptPreview("");
          if (!transcript) {
            emitError("StudyGPT could not transcribe the recording.");
            return;
          }

          onTranscript(transcript);
        } catch (error) {
          setTranscriptPreview("");
          emitError(
            error instanceof Error
              ? error.message
              : "StudyGPT could not transcribe the recording.",
          );
        }
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      startVolumeMonitoring(stream, recorder);
      return true;
    } catch (error) {
      stopVolumeMonitoring();
      stopMediaTracks();
      emitError(
        error instanceof DOMException &&
        (error.name === "NotAllowedError" || error.name === "SecurityError")
          ? isEmbeddedFrame
            ? "Microphone permission was blocked in the embedded app. Open the direct app tab and allow microphone access."
            : "Microphone permission was denied. Allow microphone access in your browser and try again."
          : error instanceof DOMException && error.name === "NotFoundError"
            ? "No microphone was found on this device."
            : "StudyGPT could not access the microphone.",
      );
      return false;
    }
  }, [
    emitError,
    isEmbeddedFrame,
    isMediaRecordingSupported,
    onTranscript,
    startVolumeMonitoring,
    stopVolumeMonitoring,
    stopMediaTracks,
    transcribeAudio,
  ]);

  const startListening = useCallback(() => {
    if (!isRecognitionSupported) {
      if (
        settingsRef.current.mode === "continuous" &&
        isMediaRecordingSupported &&
        transcribeAudio &&
        !fallbackNoticeShownRef.current
      ) {
        emitError(
          "Browser speech recognition is unavailable here. StudyGPT will record your speech and transcribe it automatically instead.",
        );
      }

      return startAudioRecording();
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
  }, [
    emitError,
    isEmbeddedFrame,
    isMediaRecordingSupported,
    isRecognitionSupported,
    startAudioRecording,
    transcribeAudio,
  ]);

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
      stopVolumeMonitoring();
      stopMediaTracks();
      recognitionRef.current = null;
    };
  }, [
    commitTranscript,
    emitError,
    isEmbeddedFrame,
    settings.mode,
    startListening,
    stopMediaTracks,
    stopVolumeMonitoring,
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
