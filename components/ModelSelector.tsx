"use client";

import { MODEL_OPTIONS, PROVIDER_LABELS } from "@/lib/constants";
import type { ProviderId } from "@/lib/types";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface ModelSelectorProps {
  value: string;
  onChange: (model: string, provider: ProviderId) => void;
}

export function ModelSelector({ value, onChange }: ModelSelectorProps) {
  return (
    <Select
      value={value}
      onValueChange={(nextValue) => {
        const option = MODEL_OPTIONS.find((entry) => entry.id === nextValue);
        if (option) {
          onChange(option.id, option.provider);
        }
      }}
    >
      <SelectTrigger className="min-w-[15rem] md:min-w-[18rem]">
        <SelectValue placeholder="Choose a model" />
      </SelectTrigger>
      <SelectContent>
        {(["groq", "openai", "anthropic", "gemini"] as ProviderId[]).map(
          (provider, index) => (
            <div key={provider}>
              {index > 0 ? <SelectSeparator /> : null}
              <SelectGroup>
                <SelectLabel>{PROVIDER_LABELS[provider]}</SelectLabel>
                {MODEL_OPTIONS.filter((entry) => entry.provider === provider).map((entry) => (
                  <SelectItem key={entry.id} value={entry.id}>
                    {entry.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </div>
          ),
        )}
      </SelectContent>
    </Select>
  );
}
