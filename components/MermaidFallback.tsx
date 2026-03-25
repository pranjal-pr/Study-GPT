"use client";

import { useEffect, useId, useState } from "react";
import { useTheme } from "next-themes";

interface MermaidFallbackProps {
  chart: string;
}

export function MermaidFallback({ chart }: MermaidFallbackProps) {
  const [svg, setSvg] = useState("");
  const [error, setError] = useState("");
  const id = useId();
  const { resolvedTheme } = useTheme();

  useEffect(() => {
    let cancelled = false;

    async function render() {
      try {
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "loose",
          theme: resolvedTheme === "dark" ? "dark" : "neutral",
        });

        const result = await mermaid.render(`mermaid-${id}`, chart);
        if (!cancelled) {
          setSvg(result.svg);
          setError("");
        }
      } catch (renderError) {
        if (!cancelled) {
          setError(renderError instanceof Error ? renderError.message : "Unable to render");
        }
      }
    }

    render();
    return () => {
      cancelled = true;
    };
  }, [chart, id, resolvedTheme]);

  if (error) {
    return (
      <div className="rounded-[1.6rem] border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
        Mermaid fallback unavailable: {error}
      </div>
    );
  }

  return (
    <div
      className="h-full min-h-[22rem] overflow-auto rounded-[1.6rem] border border-border bg-background/55 p-4"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
