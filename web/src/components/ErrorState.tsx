import type { ReactNode } from "react";
import { AlertTriangle, RotateCw } from "lucide-react";
import { cn } from "@/lib/utils";

interface ErrorStateProps {
  title?: ReactNode;
  description?: ReactNode;
  onRetry?: () => void;
  className?: string;
}

/**
 * Inline error block — lucide `AlertTriangle` + title + optional retry.
 * Used inside `GlassPanel` sections when a specific query fails so the
 * rest of the page stays usable.
 */
export function ErrorState({
  title = "Something went wrong",
  description,
  onRetry,
  className,
}: ErrorStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center text-center py-10 px-4 gap-3",
        className,
      )}
      role="alert"
    >
      <AlertTriangle
        size={48}
        strokeWidth={1.5}
        style={{ color: "var(--danger)" }}
        aria-hidden="true"
      />
      <div
        style={{
          fontSize: "0.95rem",
          fontWeight: 600,
          color: "var(--ink-primary)",
        }}
      >
        {title}
      </div>
      {description && (
        <div
          style={{
            fontSize: "0.82rem",
            color: "var(--ink-secondary)",
            maxWidth: "32rem",
          }}
        >
          {description}
        </div>
      )}
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex items-center gap-1.5 mt-1"
          style={{
            padding: "0.5rem 1rem",
            background: "var(--bg-card)",
            border: "1px solid var(--rule-soft)",
            borderRadius: "var(--r-btn)",
            fontSize: "0.82rem",
            fontWeight: 600,
            color: "var(--eg-navy)",
            cursor: "pointer",
            fontFamily: "var(--font-sans)",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          <RotateCw size={14} />
          Retry
        </button>
      )}
    </div>
  );
}
