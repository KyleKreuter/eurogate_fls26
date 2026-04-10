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
  onHoverTerminal: (id: string | null) => void;
  onClickTerminal: (id: string) => void;
}

export function EuropeMap({
  hoveredTerminal,
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
          zoom: 6,
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
        latitude: 42,
        zoom: 3.3,
      }}
      style={{ width: "100%", height: "100%" }}
      mapStyle="mapbox://styles/mapbox/dark-v11"
      interactive={true}
      scrollZoom={false}
      dragPan={true}
      doubleClickZoom={false}
      attributionControl={false}
      logoPosition="bottom-left"
    >
      {TERMINALS.map((terminal) => {
        const isHovered = hoveredTerminal === terminal.id;
        const isActive = terminal.active;
        const isHighlighted = isHovered || isActive;

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
              {/* Pulse ring for active/hovered */}
              {isHighlighted && (
                <span
                  className="absolute rounded-full marker-pulse"
                  style={{
                    width: 24,
                    height: 24,
                    background: "var(--td-secondary)",
                  }}
                />
              )}

              {/* Outer glow ring */}
              {isHighlighted && (
                <span
                  className="absolute rounded-full"
                  style={{
                    width: 20,
                    height: 20,
                    border: "1px solid var(--td-secondary)",
                    opacity: 0.4,
                  }}
                />
              )}

              {/* Marker dot */}
              <span
                className="relative rounded-full transition-all duration-150"
                style={{
                  width: isHovered ? 12 : isActive ? 10 : 7,
                  height: isHovered ? 12 : isActive ? 10 : 7,
                  background: isHighlighted
                    ? "var(--td-secondary)"
                    : "rgba(255,255,255,0.8)",
                  boxShadow: isHighlighted
                    ? "0 0 12px rgba(255,180,168,0.5)"
                    : "0 0 6px rgba(255,255,255,0.3)",
                }}
              />

              {/* City label on hover */}
              {isHovered && (
                <div
                  className="absolute bottom-full mb-2 whitespace-nowrap px-2.5 py-1 text-[10px] font-bold tracking-tight"
                  style={{
                    fontFamily: "var(--font-display)",
                    background: "var(--td-surface-container-highest)",
                    color: "#fff",
                    border: "1px solid rgba(255,255,255,0.1)",
                  }}
                >
                  <span
                    className="inline-block w-1.5 h-1.5 mr-1.5"
                    style={{ background: "var(--td-secondary)" }}
                  />
                  {terminal.city.toUpperCase()}
                </div>
              )}

              {/* Persistent Hamburg HUB label */}
              {isActive && !isHovered && (
                <div
                  className="absolute bottom-full mb-2 whitespace-nowrap px-2.5 py-1 text-[10px] font-bold tracking-tight"
                  style={{
                    fontFamily: "var(--font-display)",
                    background: "var(--td-surface-container-highest)",
                    color: "#fff",
                    border: "1px solid rgba(255,255,255,0.1)",
                  }}
                >
                  <span
                    className="inline-block w-1.5 h-1.5 mr-1.5"
                    style={{ background: "var(--td-secondary)" }}
                  />
                  HAMBURG HUB
                </div>
              )}
            </div>
          </Marker>
        );
      })}
    </Map>
  );
}
