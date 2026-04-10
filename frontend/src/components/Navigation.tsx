import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { EurogateLogo } from "@/components/EurogateLogo";

interface NavigationProps {
  /** When true, renders on a dark background with white logo */
  variant?: "dark" | "light";
  /** Show a back link to the terminals overview */
  showBackLink?: boolean;
  /** Text to display in the center of the nav bar */
  centerLabel?: string;
}

/**
 * Shared navigation header for the Eurogate application.
 * On the locations page, uses the dark variant with a white logo.
 * On the dashboard page, uses the light variant with back link and center label.
 */
export function Navigation({ variant = "dark", showBackLink = false, centerLabel }: NavigationProps) {
  const isDark = variant === "dark";

  return (
    <header
      className="w-full px-5 md:px-8 py-4 md:py-5 flex items-center justify-between"
      style={{
        background: isDark ? "transparent" : "var(--bg-white)",
        borderBottom: isDark ? "none" : "1px solid var(--border-default)",
      }}
    >
      <EurogateLogo
        height={30}
        className={isDark ? "brightness-0 invert" : ""}
      />

      {centerLabel && (
        <span
          className="hidden md:inline text-xs font-medium tracking-wide"
          style={{ color: "var(--text-muted)" }}
        >
          {centerLabel}
        </span>
      )}

      {showBackLink ? (
        <Link
          to="/"
          className="flex items-center gap-1.5 text-xs font-medium no-underline transition-colors duration-150"
          style={{ color: "var(--text-secondary)" }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLAnchorElement).style.color = "var(--eg-navy)"; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLAnchorElement).style.color = "var(--text-secondary)"; }}
        >
          <ArrowLeft size={14} strokeWidth={2} />
          <span className="hidden sm:inline">Terminals</span>
        </Link>
      ) : (
        /* Keep the layout balanced when there's no back link */
        <div />
      )}
    </header>
  );
}
