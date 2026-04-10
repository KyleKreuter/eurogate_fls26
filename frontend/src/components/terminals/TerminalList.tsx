import { ChevronRight, ArrowRight } from "lucide-react";
import type { TerminalLocation } from "@/data/terminals";

const LIST_GROUPS = [
  { id: "DE", label: "Deutschland", filter: (t: TerminalLocation) => t.countryCode === "DE" },
  { id: "IT", label: "Italia", filter: (t: TerminalLocation) => t.countryCode === "IT" },
  { id: "GLOBAL", label: "Global Network", filter: (t: TerminalLocation) => !["DE", "IT"].includes(t.countryCode) },
];

interface TerminalListProps {
  terminals: TerminalLocation[];
  hoveredTerminal: string | null;
  onHoverTerminal: (id: string | null) => void;
  onClickTerminal: (id: string) => void;
}

export function TerminalList({
  terminals,
  hoveredTerminal,
  onHoverTerminal,
  onClickTerminal,
}: TerminalListProps) {
  return (
    <div className="flex flex-col gap-8">
      {LIST_GROUPS.map((group) => {
        const groupTerminals = terminals.filter(group.filter);
        if (groupTerminals.length === 0) return null;

        return (
          <div
            key={group.id}
            className="p-6"
            style={{
              background: "var(--td-surface-container)",
              border: "1px solid rgba(255,255,255,0.04)",
            }}
          >
            {/* Group label */}
            <h3
              className="text-[0.6875rem] font-black tracking-[0.2em] uppercase mb-6 opacity-60"
              style={{
                fontFamily: "var(--font-display)",
                color: "var(--td-on-surface-variant)",
              }}
            >
              {group.id === "GLOBAL" ? group.label : `Region: ${group.label}`}
            </h3>

            {/* Terminal rows */}
            <div className="space-y-4">
              {groupTerminals.map((terminal) => {
                const isActive = terminal.active;
                const isHovered = hoveredTerminal === terminal.id;
                const isHighlighted = isActive || isHovered;

                return (
                  <button
                    key={terminal.id}
                    className="w-full flex items-center justify-between group cursor-pointer p-2 -mx-2 transition-all text-left"
                    style={{
                      background: isActive
                        ? "rgba(255,180,168,0.1)"
                        : isHovered
                          ? "rgba(255,255,255,0.03)"
                          : "transparent",
                      borderLeft: isActive
                        ? "4px solid var(--td-secondary)"
                        : "4px solid transparent",
                    }}
                    onMouseEnter={() => onHoverTerminal(terminal.id)}
                    onMouseLeave={() => onHoverTerminal(null)}
                    onClick={() => onClickTerminal(terminal.id)}
                    aria-label={
                      isActive
                        ? `Go to ${terminal.city} dashboard`
                        : `${terminal.city} — coming soon`
                    }
                  >
                    <div className="flex items-center gap-4">
                      <span
                        className="flex-shrink-0"
                        style={{
                          width: isActive ? 8 : 6,
                          height: isActive ? 8 : 6,
                          background: isHighlighted
                            ? "var(--td-secondary)"
                            : "rgba(255,255,255,0.4)",
                        }}
                      />
                      <span
                        className="text-lg font-bold tracking-tight"
                        style={{
                          fontFamily: "var(--font-display)",
                          color: isHighlighted
                            ? "var(--td-secondary)"
                            : "#fff",
                        }}
                      >
                        {group.id === "GLOBAL"
                          ? `${terminal.city} (${terminal.country})`
                          : terminal.city}
                      </span>
                    </div>

                    {isActive ? (
                      <ArrowRight
                        size={16}
                        style={{ color: "var(--td-secondary)" }}
                      />
                    ) : (
                      <ChevronRight
                        size={14}
                        className="opacity-0 group-hover:opacity-100 transition-opacity"
                        style={{ color: "var(--td-on-surface-variant)" }}
                      />
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
