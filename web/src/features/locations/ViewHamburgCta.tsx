import { useNavigate } from "react-router-dom";
import { ArrowRight } from "lucide-react";

/**
 * Full-width CTA pinned to the bottom of the right panel. Navigates to
 * the Hamburg dashboard. Primary blue background, uppercase Outfit label,
 * lucide ArrowRight on the trailing edge.
 */
export function ViewHamburgCta() {
  const navigate = useNavigate();
  return (
    <button
      type="button"
      onClick={() => navigate("/dashboard/hamburg")}
      className="w-full inline-flex items-center justify-center gap-2 transition-colors"
      style={{
        padding: "1rem 1.25rem",
        background: "var(--accent-primary)",
        color: "#ffffff",
        border: 0,
        borderRadius: "var(--r-panel)",
        fontFamily: "var(--font-sans)",
        fontSize: "0.82rem",
        fontWeight: 600,
        textTransform: "uppercase",
        letterSpacing: "0.1em",
        cursor: "pointer",
        boxShadow: "var(--shadow-panel)",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLButtonElement).style.background =
          "var(--accent-primary-hover)";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLButtonElement).style.background =
          "var(--accent-primary)";
      }}
    >
      View Hamburg Dashboard
      <ArrowRight size={16} strokeWidth={2.2} />
    </button>
  );
}
