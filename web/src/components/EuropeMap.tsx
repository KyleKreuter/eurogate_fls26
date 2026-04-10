import { useRef, useCallback } from "react";
import Map, { Marker } from "react-map-gl/mapbox";
import "mapbox-gl/dist/mapbox-gl.css";
import { TERMINALS } from "@/data/terminals";
import type { TerminalLocation } from "@/data/terminals";

// Re-export for backward compatibility
export { TERMINALS } from "@/data/terminals";
export type { TerminalLocation } from "@/data/terminals";

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;

interface EuropeMapProps {
  hoveredTerminal: string | null;
  selectedTerminal?: string | null;
  onHoverTerminal: (id: string | null) => void;
  onClickTerminal: (id: string) => void;
}

const COLOR_INACTIVE = "#2980B9"; // var(--accent-primary)
const COLOR_HOVER = "#0b0222"; // var(--eg-navy)
const COLOR_ACTIVE = "#E67E22"; // var(--warning) — Hamburg beacon

/**
 * Eurogate 9-terminal map on Mapbox light-v11. Orange Hamburg beacon,
 * blue inactive markers, navy on hover. Used by the landing split-screen.
 */
export function EuropeMap({
  hoveredTerminal,
  selectedTerminal,
  onHoverTerminal,
  onClickTerminal,
}: EuropeMapProps) {
  const mapRef = useRef<mapboxgl.Map | null>(null);

  const handleMarkerClick = useCallback(
    (terminal: TerminalLocation) => {
      const map = mapRef.current;
      if (map) {
        map.flyTo({
          center: terminal.coordinates,
          zoom: 5.2,
          duration: 1400,
          essential: true,
        });
      }
      onClickTerminal(terminal.id);
    },
    [onClickTerminal],
  );

  return (
    <Map
      ref={(ref) => {
        mapRef.current = ref?.getMap() ?? null;
      }}
      mapboxAccessToken={MAPBOX_TOKEN}
      initialViewState={{
        longitude: 15,
        latitude: 44,
        zoom: 3.4,
      }}
      style={{ width: "100%", height: "100%" }}
      mapStyle="mapbox://styles/mapbox/light-v11"
      interactive={true}
      scrollZoom={false}
      dragPan={true}
      doubleClickZoom={false}
      attributionControl={false}
      logoPosition="bottom-left"
    >
      {TERMINALS.map((terminal) => {
        const isHovered = hoveredTerminal === terminal.id;
        const isSelected = selectedTerminal === terminal.id;
        const isActive = terminal.active;
        const showBeacon = isActive;
        const color = isHovered
          ? COLOR_HOVER
          : isActive
            ? COLOR_ACTIVE
            : isSelected
              ? COLOR_HOVER
              : COLOR_INACTIVE;

        return (
          <Marker
            key={terminal.id}
            longitude={terminal.coordinates[0]}
            latitude={terminal.coordinates[1]}
            anchor="center"
          >
            <div
              className="relative flex items-center justify-center"
              style={{ cursor: terminal.active ? "pointer" : "default" }}
              onMouseEnter={() => onHoverTerminal(terminal.id)}
              onMouseLeave={() => onHoverTerminal(null)}
              onClick={() => handleMarkerClick(terminal)}
            >
              {/* Pulse ring only on the Hamburg beacon */}
              {showBeacon && (
                <span
                  className="absolute rounded-full marker-pulse"
                  style={{
                    width: 26,
                    height: 26,
                    background: COLOR_ACTIVE,
                    opacity: 0.45,
                  }}
                />
              )}

              {/* Marker dot */}
              <span
                className="relative rounded-full transition-all duration-150"
                style={{
                  width: isHovered || isActive ? 12 : 8,
                  height: isHovered || isActive ? 12 : 8,
                  background: color,
                  border: "2px solid #ffffff",
                  boxShadow:
                    isHovered || isActive
                      ? `0 0 0 2px ${color}33, 0 2px 4px rgba(11,2,34,0.15)`
                      : "0 1px 3px rgba(11,2,34,0.15)",
                }}
              />

              {/* City label on hover */}
              {isHovered && (
                <div
                  className="absolute bottom-full mb-2 whitespace-nowrap"
                  style={{
                    fontFamily: "var(--font-sans)",
                    fontSize: "0.68rem",
                    fontWeight: 600,
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                    background: "var(--bg-card)",
                    color: "var(--eg-navy)",
                    border: "1px solid var(--rule-soft)",
                    borderRadius: "var(--r-btn)",
                    padding: "0.35rem 0.6rem",
                    boxShadow: "var(--shadow-panel)",
                  }}
                >
                  {terminal.city.toUpperCase()}
                </div>
              )}

              {/* Persistent HAMBURG label for the active beacon */}
              {isActive && !isHovered && (
                <div
                  className="absolute bottom-full mb-2 whitespace-nowrap"
                  style={{
                    fontFamily: "var(--font-sans)",
                    fontSize: "0.68rem",
                    fontWeight: 600,
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                    background: "var(--bg-card)",
                    color: "var(--eg-navy)",
                    border: "1px solid var(--rule-soft)",
                    borderRadius: "var(--r-btn)",
                    padding: "0.35rem 0.6rem",
                    boxShadow: "var(--shadow-panel)",
                  }}
                >
                  <span
                    style={{
                      display: "inline-block",
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                      background: COLOR_ACTIVE,
                      marginRight: "0.45rem",
                      verticalAlign: "middle",
                    }}
                  />
                  HAMBURG
                </div>
              )}
            </div>
          </Marker>
        );
      })}
    </Map>
  );
}
