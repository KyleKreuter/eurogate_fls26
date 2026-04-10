/**
 * Landing page hero stats — "EUROGATE NETWORK / 9 TERMINALS / 5 COUNTRIES".
 * Outfit uppercase label + JetBrains Mono big numbers.
 */
export function HeroStats() {
  return (
    <div className="flex flex-col gap-1.5">
      <div
        style={{
          fontFamily: "var(--font-sans)",
          fontSize: "0.72rem",
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.12em",
          color: "var(--accent-primary)",
        }}
      >
        Eurogate Network
      </div>
      <div className="flex items-baseline gap-2">
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "2.2rem",
            fontWeight: 600,
            color: "var(--ink-primary)",
            lineHeight: 1,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          9
        </span>
        <span
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: "0.8rem",
            fontWeight: 500,
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            color: "var(--ink-secondary)",
          }}
        >
          Terminals
        </span>
      </div>
      <div className="flex items-baseline gap-2">
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "2.2rem",
            fontWeight: 600,
            color: "var(--ink-primary)",
            lineHeight: 1,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          5
        </span>
        <span
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: "0.8rem",
            fontWeight: 500,
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            color: "var(--ink-secondary)",
          }}
        >
          Countries
        </span>
      </div>
    </div>
  );
}
