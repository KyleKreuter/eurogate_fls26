import type { ReactNode, HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

interface GlassPanelProps extends Omit<HTMLAttributes<HTMLDivElement>, "title"> {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  variant?: "default" | "flush" | "tight";
  children?: ReactNode;
}

/**
 * The canonical card wrapper. 8px radius, white background, soft shadow,
 * 2.5rem padding. Matches the Python dashboard's `.glass-panel` style.
 *
 * Use `variant="flush"` for tables (zero padding, keeps border + radius).
 * Use `variant="tight"` for dense internal cards (1.5rem padding).
 */
export function GlassPanel({
  title,
  subtitle,
  actions,
  variant = "default",
  className,
  children,
  ...rest
}: GlassPanelProps) {
  const variantClass =
    variant === "flush"
      ? "glass-panel glass-panel--flush"
      : variant === "tight"
        ? "glass-panel glass-panel--tight"
        : "glass-panel";

  const hasHeader = title || subtitle || actions;

  return (
    <div className={cn(variantClass, className)} {...rest}>
      {hasHeader && (
        <div className="flex items-start justify-between mb-4 gap-4">
          <div>
            {title && (
              <h3
                className="font-sans"
                style={{
                  fontSize: "1rem",
                  fontWeight: 600,
                  color: "var(--ink-primary)",
                  margin: 0,
                }}
              >
                {title}
              </h3>
            )}
            {subtitle && (
              <p
                style={{
                  fontSize: "0.78rem",
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  color: "var(--ink-secondary)",
                  marginTop: "0.25rem",
                }}
              >
                {subtitle}
              </p>
            )}
          </div>
          {actions && <div className="shrink-0">{actions}</div>}
        </div>
      )}
      {children}
    </div>
  );
}
