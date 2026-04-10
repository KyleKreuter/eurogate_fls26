import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { EuropeMap } from "@/components/EuropeMap";
import { HeroStats } from "./HeroStats";
import { TerminalListPanel } from "./TerminalListPanel";
import { ViewHamburgCta } from "./ViewHamburgCta";

/**
 * Landing page 65/35 split-screen: Mapbox light-v11 on the left,
 * terminal list + Hamburg CTA on the right. On < 1280px the panel
 * stacks underneath the map.
 */
export function LandingSplitScreen() {
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleTerminalClick = (id: string) => {
    if (id === "hamburg") {
      navigate("/dashboard/hamburg");
    }
  };

  return (
    <div
      className="flex-1 flex flex-col xl:flex-row"
      style={{
        minHeight: "calc(100vh - 80px)",
        background: "var(--bg-page)",
      }}
    >
      {/* Map — 65% on desktop */}
      <div
        className="relative xl:flex-[65] bg-white"
        style={{
          minHeight: "50vh",
          borderRight: "1px solid var(--rule-soft)",
        }}
      >
        <EuropeMap
          hoveredTerminal={hoveredId}
          onHoverTerminal={setHoveredId}
          onClickTerminal={handleTerminalClick}
        />
      </div>

      {/* Right panel — 35% */}
      <aside
        className="xl:flex-[35] flex flex-col bg-white"
        style={{
          minWidth: 0,
          maxWidth: 520,
          width: "100%",
          borderLeft: "1px solid var(--rule-soft)",
        }}
      >
        {/* Hero stats */}
        <div className="px-6 py-6">
          <HeroStats />
        </div>

        {/* Terminal list */}
        <TerminalListPanel
          hoveredId={hoveredId}
          onHover={setHoveredId}
          onClick={handleTerminalClick}
        />

        {/* CTA */}
        <div className="px-6 py-5">
          <ViewHamburgCta />
        </div>
      </aside>
    </div>
  );
}
