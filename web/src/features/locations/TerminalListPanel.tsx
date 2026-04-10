import { Star } from "lucide-react";
import { TERMINALS } from "@/data/terminals";
import type { TerminalLocation } from "@/data/terminals";

interface TerminalListPanelProps {
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  onClick: (id: string) => void;
}

/** Country grouping order as shown in the Python dashboard. */
const COUNTRY_ORDER: Array<{ code: string; label: string }> = [
  { code: "DE", label: "Deutschland" },
  { code: "IT", label: "Italia" },
  { code: "MA", label: "Global Network" }, // Morocco/Cyprus/Egypt grouped
  { code: "CY", label: "Global Network" },
  { code: "EG", label: "Global Network" },
];

function groupTerminals(): Array<{ label: string; rows: TerminalLocation[] }> {
  const groups = new Map<string, TerminalLocation[]>();
  for (const c of COUNTRY_ORDER) {
    if (!groups.has(c.label)) groups.set(c.label, []);
  }
  for (const t of TERMINALS) {
    const entry = COUNTRY_ORDER.find((c) => c.code === t.countryCode);
    const label = entry?.label ?? "Other";
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label)!.push(t);
  }
  return Array.from(groups.entries()).map(([label, rows]) => ({ label, rows }));
}

/**
 * Scrollable list of the 9 Eurogate terminals, grouped by country block.
 * Hamburg (active) is highlighted with a 3px left border + soft blue bg
 * + lucide Star icon.
 */
export function TerminalListPanel({ hoveredId, onHover, onClick }: TerminalListPanelProps) {
  const groups = groupTerminals();

  return (
    <div
      className="flex-1 overflow-y-auto"
      style={{
        borderTop: "1px solid var(--rule-soft)",
        borderBottom: "1px solid var(--rule-soft)",
      }}
    >
      {groups.map((group) => (
        <div key={group.label} className="py-4">
          <div
            className="px-6 mb-2"
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: "0.68rem",
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.12em",
              color: "var(--ink-muted)",
            }}
          >
            {group.label}
          </div>
          <ul className="flex flex-col">
            {group.rows.map((t) => {
              const isHovered = hoveredId === t.id;
              const isActive = t.active;
              return (
                <li key={t.id}>
                  <button
                    type="button"
                    onMouseEnter={() => onHover(t.id)}
                    onMouseLeave={() => onHover(null)}
                    onClick={() => onClick(t.id)}
                    className="w-full text-left transition-colors"
                    style={{
                      padding: "0.65rem 1.5rem",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      background: isActive
                        ? "var(--accent-primary-soft)"
                        : isHovered
                          ? "var(--bg-soft)"
                          : "transparent",
                      borderLeft: `3px solid ${isActive ? "var(--accent-primary)" : "transparent"}`,
                      cursor: isActive ? "pointer" : "default",
                      fontFamily: "var(--font-sans)",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "0.88rem",
                        fontWeight: isActive ? 600 : 500,
                        color: isActive
                          ? "var(--eg-navy)"
                          : "var(--ink-primary)",
                      }}
                    >
                      {t.city}
                    </span>
                    {isActive && (
                      <Star
                        size={13}
                        fill="var(--warning)"
                        strokeWidth={0}
                        aria-label="Active terminal"
                      />
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </div>
  );
}
