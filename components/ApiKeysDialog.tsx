"use client";

import type { ProviderId, ProviderKeys } from "@/lib/types";
import { PROVIDER_LABELS } from "@/lib/constants";
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

interface ApiKeysDialogProps {
  apiKeys: ProviderKeys;
  onChange: (provider: ProviderId, value: string) => void;
}

export function ApiKeysDialog({ apiKeys, onChange }: ApiKeysDialogProps) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button type="button" variant="secondary">
          API Keys
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Provider API keys</DialogTitle>
          <DialogDescription>
            Keys entered here stay in this browser via localStorage. If a key is left
            blank, StudyGPT falls back to the matching server environment variable.
          </DialogDescription>
        </DialogHeader>
        <div className="mt-6 grid gap-4 md:grid-cols-2">
          {(Object.keys(apiKeys) as ProviderId[]).map((provider) => (
            <label
              key={provider}
              className="rounded-[1.5rem] border border-border bg-muted/45 p-4"
            >
              <span className="mb-2 block text-sm font-medium">
                {PROVIDER_LABELS[provider]}
              </span>
              <Input
                type="password"
                autoComplete="off"
                placeholder={`${PROVIDER_LABELS[provider]} API key`}
                value={apiKeys[provider]}
                onChange={(event) => onChange(provider, event.target.value)}
              />
            </label>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
