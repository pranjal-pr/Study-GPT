"use client";

import { Mic, MicOff, Settings2, Volume2, VolumeX } from "lucide-react";
import type { VoiceSettings } from "@/lib/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";

interface VoiceToggleProps {
  availableVoices: SpeechSynthesisVoice[];
  isListening: boolean;
  isSpeaking: boolean;
  isSupported: boolean;
  settings: VoiceSettings;
  onToggleEnabled: () => void;
  onToggleListening: () => void;
  onStopSpeaking: () => void;
  onUpdateSettings: (settings: Partial<VoiceSettings>) => void;
}

export function VoiceToggle({
  availableVoices,
  isListening,
  isSpeaking,
  isSupported,
  settings,
  onToggleEnabled,
  onToggleListening,
  onStopSpeaking,
  onUpdateSettings,
}: VoiceToggleProps) {
  const active = settings.enabled || isListening || isSpeaking;

  return (
    <div className="flex items-center gap-2">
      <Button
        type="button"
        variant={active ? "default" : "secondary"}
        size="icon"
        onClick={onToggleEnabled}
        disabled={!isSupported}
        aria-label="Toggle voice mode"
        className="relative h-12 w-12 rounded-full"
      >
        {active ? (
          <Mic className="h-5 w-5" />
        ) : (
          <MicOff className="h-5 w-5 text-muted-foreground" />
        )}
        {isListening ? <span className="animated-orb absolute inset-0 rounded-full" /> : null}
      </Button>
      <Button
        type="button"
        variant="secondary"
        size="icon"
        onClick={isSpeaking ? onStopSpeaking : onToggleListening}
        disabled={!isSupported}
        aria-label={isSpeaking ? "Stop speaking" : "Toggle listening"}
        className="h-10 w-10"
      >
        {isSpeaking ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
      </Button>
      <Dialog>
        <DialogTrigger asChild>
          <Button type="button" variant="secondary" size="icon" className="h-10 w-10">
            <Settings2 className="h-4 w-4" />
          </Button>
        </DialogTrigger>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Voice mode</DialogTitle>
            <DialogDescription>
              Browser speech APIs keep StudyGPT low-cost while still supporting live
              listening and spoken replies.
            </DialogDescription>
          </DialogHeader>
          <div className="mt-6 space-y-5">
            <div className="flex items-center justify-between rounded-[1.4rem] border border-border bg-muted/55 px-4 py-3">
              <div>
                <p className="text-sm font-medium">Enable voice tutor</p>
                <p className="text-xs text-muted-foreground">
                  Turn on continuous voice conversation.
                </p>
              </div>
              <Switch
                checked={settings.enabled}
                disabled={!isSupported}
                onCheckedChange={(checked) => onUpdateSettings({ enabled: checked })}
              />
            </div>

            <div className="flex items-center justify-between rounded-[1.4rem] border border-border bg-muted/55 px-4 py-3">
              <div>
                <p className="text-sm font-medium">Continuous listening</p>
                <p className="text-xs text-muted-foreground">
                  Off means push-to-talk style capture.
                </p>
              </div>
              <Switch
                checked={settings.mode === "continuous"}
                onCheckedChange={(checked) =>
                  onUpdateSettings({ mode: checked ? "continuous" : "push" })
                }
              />
            </div>

            <div className="flex items-center justify-between rounded-[1.4rem] border border-border bg-muted/55 px-4 py-3">
              <div>
                <p className="text-sm font-medium">Auto speak replies</p>
                <p className="text-xs text-muted-foreground">
                  StudyGPT reads the answer aloud after each response.
                </p>
              </div>
              <Switch
                checked={settings.autoSpeak}
                onCheckedChange={(checked) => onUpdateSettings({ autoSpeak: checked })}
              />
            </div>

            <label className="block space-y-2">
              <span className="text-sm font-medium">Voice URI</span>
              <Input
                value={settings.voiceURI}
                onChange={(event) => onUpdateSettings({ voiceURI: event.target.value })}
                placeholder={
                  availableVoices[0]?.voiceURI || "Leave blank to use the browser default"
                }
              />
              <p className="text-xs text-muted-foreground">
                Available: {availableVoices.slice(0, 3).map((voice) => voice.name).join(", ")}
                {availableVoices.length > 3 ? "..." : ""}
              </p>
            </label>

            <label className="block space-y-2">
              <span className="text-sm font-medium">Speech rate: {settings.rate.toFixed(1)}</span>
              <input
                type="range"
                min="0.7"
                max="1.4"
                step="0.1"
                value={settings.rate}
                onChange={(event) =>
                  onUpdateSettings({ rate: Number.parseFloat(event.target.value) })
                }
                className="w-full accent-[var(--primary)]"
              />
            </label>

            <label className="block space-y-2">
              <span className="text-sm font-medium">
                Speech pitch: {settings.pitch.toFixed(1)}
              </span>
              <input
                type="range"
                min="0.8"
                max="1.3"
                step="0.1"
                value={settings.pitch}
                onChange={(event) =>
                  onUpdateSettings({ pitch: Number.parseFloat(event.target.value) })
                }
                className="w-full accent-[var(--primary)]"
              />
            </label>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
