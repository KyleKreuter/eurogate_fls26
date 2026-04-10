import { useState, useCallback, useMemo, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { TERMINALS } from "@/data/terminals";
import type { Region } from "@/data/terminals";
import { EuropeMap } from "@/components/EuropeMap";
import { TerminalNavBar } from "@/components/terminals/TerminalNavBar";
import { TerminalSidebar } from "@/components/terminals/TerminalSidebar";
import { TerminalList } from "@/components/terminals/TerminalList";
import { StatsBentoGrid } from "@/components/terminals/StatsBentoGrid";
import { TerminalFooter } from "@/components/terminals/TerminalFooter";

export function LocationsOverview() {
  const navigate = useNavigate();
  const [hoveredTerminal, setHoveredTerminal] = useState<string | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<Region>("all");

  // Match html background to dark theme for overscroll consistency
  useEffect(() => {
    const html = document.documentElement;
    const prev = html.style.background;
    html.style.background = "#131318";
    return () => { html.style.background = prev; };
  }, []);

  const filteredTerminals = useMemo(() => {
    if (selectedRegion === "all") return TERMINALS;
    return TERMINALS.filter((t) => t.region === selectedRegion);
  }, [selectedRegion]);

  const handleClickTerminal = useCallback(
    (id: string) => {
      if (id === "hamburg") {
        navigate("/dashboard/hamburg");
      }
    },
    [navigate],
  );

  return (
    <div
      className="terminals-dark min-h-screen flex flex-col page-enter"
      style={{ background: "var(--td-surface)" }}
    >
      <TerminalNavBar />
      <TerminalSidebar
        selectedRegion={selectedRegion}
        onSelectRegion={setSelectedRegion}
      />

      {/* Main Content Canvas */}
      <main
        className="pt-28 pb-20 xl:pl-64 flex-1"
        style={{ background: "var(--td-surface)" }}
      >
        <div className="max-w-[1440px] mx-auto px-8 md:px-12">
          {/* Hero Title */}
          <header
            className="mb-16 pl-8"
            style={{ borderLeft: "4px solid var(--td-secondary)" }}
          >
            <h1
              className="text-6xl md:text-7xl font-black tracking-tighter text-white uppercase mb-2"
              style={{ fontFamily: "var(--font-display)" }}
            >
              Unsere Terminals
            </h1>
            <p
              className="text-xl uppercase tracking-widest"
              style={{
                fontFamily: "var(--font-display)",
                color: "var(--td-on-surface-variant)",
              }}
            >
              9 Standorte &middot; 5 L&auml;nder
            </p>
          </header>

          {/* Map + Location List */}
          <div className="flex flex-col lg:flex-row gap-12 items-start">
            {/* Map Container */}
            <section
              className="w-full lg:w-[60%] h-[600px] relative overflow-hidden industrial-grid"
              style={{
                background: "var(--td-surface-container-low)",
                border: "1px solid rgba(255,255,255,0.06)",
              }}
            >
              {/* Red status ribbon */}
              <div
                className="absolute top-0 left-0 w-full h-1 z-10"
                style={{ background: "var(--td-secondary)" }}
              />
              <EuropeMap
                hoveredTerminal={hoveredTerminal}
                onHoverTerminal={setHoveredTerminal}
                onClickTerminal={handleClickTerminal}
              />
            </section>

            {/* Location List */}
            <section className="w-full lg:w-[40%]">
              <TerminalList
                terminals={filteredTerminals}
                hoveredTerminal={hoveredTerminal}
                onHoverTerminal={setHoveredTerminal}
                onClickTerminal={handleClickTerminal}
              />
            </section>
          </div>

          {/* Stats */}
          <StatsBentoGrid />
        </div>
      </main>

      <TerminalFooter />
    </div>
  );
}
