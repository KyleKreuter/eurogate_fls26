import { Globe, Anchor, Sun, Waves } from "lucide-react";
import { REGIONS } from "@/data/terminals";
import type { Region } from "@/data/terminals";

const REGION_ICONS: Record<string, React.ReactNode> = {
  globe: <Globe size={18} />,
  anchor: <Anchor size={18} />,
  sun: <Sun size={18} />,
  waves: <Waves size={18} />,
};

interface TerminalSidebarProps {
  selectedRegion: Region;
  onSelectRegion: (region: Region) => void;
}

export function TerminalSidebar({
  selectedRegion,
  onSelectRegion,
}: TerminalSidebarProps) {
  return (
    <aside
      className="fixed left-0 top-20 flex-col h-[calc(100vh-5rem)] w-64 z-40 hidden xl:flex"
      style={{
        background: "var(--td-surface)",
        borderRight: "1px solid rgba(255,255,255,0.06)",
      }}
    >
      {/* Header */}
      <div
        className="p-8"
        style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}
      >
        <p
          className="uppercase tracking-[0.05em] text-[0.6875rem] font-bold"
          style={{
            fontFamily: "var(--font-display)",
            color: "var(--td-secondary)",
          }}
        >
          Terminal Selector
        </p>
        <p
          className="uppercase tracking-[0.05em] text-[0.6rem] mt-1"
          style={{
            fontFamily: "var(--font-display)",
            color: "var(--td-on-surface-variant)",
          }}
        >
          Operational Depth: 18M
        </p>
      </div>

      {/* Region list */}
      <nav className="flex-1 py-6">
        {REGIONS.map((region) => {
          const isActive = selectedRegion === region.id;

          return (
            <button
              key={region.id}
              className="w-full flex items-center px-8 py-4 gap-4 uppercase tracking-[0.05em] text-xs font-bold transition-all duration-300 cursor-pointer"
              style={{
                fontFamily: "var(--font-display)",
                color: isActive
                  ? "var(--td-secondary)"
                  : "var(--td-on-surface-variant)",
                background: isActive
                  ? "rgba(255,255,255,0.03)"
                  : "transparent",
                borderLeft: isActive
                  ? "4px solid var(--td-secondary)"
                  : "4px solid transparent",
              }}
              onClick={() => onSelectRegion(region.id)}
              onMouseEnter={(e) => {
                if (!isActive) {
                  (e.currentTarget as HTMLButtonElement).style.background =
                    "rgba(255,255,255,0.05)";
                }
              }}
              onMouseLeave={(e) => {
                if (!isActive) {
                  (e.currentTarget as HTMLButtonElement).style.background =
                    "transparent";
                }
              }}
            >
              {REGION_ICONS[region.iconName]}
              <span>{region.label}</span>
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
