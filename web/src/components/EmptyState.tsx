import type { ReactNode } from "react";
import { Inbox } from "lucide-react";
import { cn } from "@/lib/utils";

interface EmptyStateProps {
  icon?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}

/**
 * Icon + title + description + action layout for "no data yet" or
 * "no results match" states. Defaults to lucide `Inbox`.
 */
export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center text-center py-12 px-4 gap-3",
        className,
      )}
    >
      <div style={{ color: "var(--ink-muted)" }} aria-hidden="true">
        {icon ?? <Inbox size={48} strokeWidth={1.5} />}
      </div>
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
            maxWidth: "28rem",
          }}
        >
          {description}
        </div>
      )}
      {action && <div className="mt-1">{action}</div>}
    </div>
  );
}
