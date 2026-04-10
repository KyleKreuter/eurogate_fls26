import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { EurogateLogo } from "@/components/EurogateLogo";
import { cn } from "@/lib/utils";

interface NavigationProps {
  /** Back arrow link target (e.g. `/` for the landing page) */
  backTo?: string;
  backLabel?: string;
  /** Right-aligned breadcrumb / page name label */
  centerLabel?: ReactNode;
  className?: string;
}

/**
 * Sticky white app navbar — 80px, 1.1rem vertical padding, subtle
 * shadow-header. Present on every page of the dashboard. Logo left,
 * optional back link next to it, optional right-side breadcrumb.
 */
export function Navigation({
  backTo,
  backLabel = "BACK",
  centerLabel,
  className,
}: NavigationProps) {
  return (
    <header className={cn("app-navbar", className)}>
      <div className="flex items-center gap-6">
        <Link to="/" className="inline-flex items-center" aria-label="Eurogate home">
          <EurogateLogo height={30} />
        </Link>
        {backTo && (
          <Link
            to={backTo}
            className="inline-flex items-center gap-2 transition-colors"
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: "0.78rem",
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: "var(--ink-secondary)",
              textDecoration: "none",
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLAnchorElement).style.color =
                "var(--eg-navy)";
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLAnchorElement).style.color =
                "var(--ink-secondary)";
            }}
          >
            <ArrowLeft size={16} strokeWidth={2} />
            {backLabel}
          </Link>
        )}
      </div>

      {centerLabel && (
        <div
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: "0.82rem",
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            color: "var(--eg-navy)",
          }}
        >
          {centerLabel}
        </div>
      )}
    </header>
  );
}
