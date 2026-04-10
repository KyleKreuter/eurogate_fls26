import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface LoadingSpinnerProps {
  size?: number;
  label?: string;
  fullscreen?: boolean;
  className?: string;
}

/**
 * Centered spinner using lucide's `Loader2` + Tailwind's `animate-spin`.
 * `fullscreen` covers the viewport for top-level Suspense fallbacks.
 */
export function LoadingSpinner({
  size = 24,
  label,
  fullscreen,
  className,
}: LoadingSpinnerProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3",
        fullscreen ? "fixed inset-0 bg-white/70 z-40" : "py-12",
        className,
      )}
    >
      <Loader2
        size={size}
        className="animate-spin"
        style={{ color: "var(--accent-primary)" }}
        aria-hidden="true"
      />
      {label && (
        <div
          style={{
            fontSize: "0.78rem",
            fontWeight: 500,
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            color: "var(--ink-secondary)",
          }}
        >
          {label}
        </div>
      )}
    </div>
  );
}
