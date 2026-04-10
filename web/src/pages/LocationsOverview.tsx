import { Navigation } from "@/components/Navigation";
import { LandingSplitScreen } from "@/features/locations/LandingSplitScreen";

/**
 * The landing page: sticky white navbar + 65/35 split-screen with the
 * Mapbox light map and the terminal list + Hamburg CTA panel.
 */
export function LocationsOverview() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navigation centerLabel="International Port Forecasting Network" />
      <LandingSplitScreen />
    </div>
  );
}
