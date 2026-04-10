import { Suspense, lazy } from "react";
import { useSearchParams } from "react-router-dom";
import { Navigation } from "@/components/Navigation";
import { Tabs } from "@/components/Tabs";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { LoadPredictionTab } from "@/features/load-prediction/LoadPredictionTab";

const HistoricalAnalyticsTab = lazy(() =>
  import("@/features/historical-analytics/HistoricalAnalyticsTab").then((m) => ({
    default: m.HistoricalAnalyticsTab,
  })),
);

const ContainerInspectorTab = lazy(() =>
  import("@/features/container-inspector/ContainerInspectorTab").then((m) => ({
    default: m.ContainerInspectorTab,
  })),
);

type TabId = "prediction" | "historical" | "inspector";

const TAB_ITEMS: Array<{ value: TabId; label: string }> = [
  { value: "prediction", label: "Load Prediction" },
  { value: "historical", label: "Historical Analytics" },
  { value: "inspector", label: "Container Inspector" },
];

function isTabId(value: string | null): value is TabId {
  return value === "prediction" || value === "historical" || value === "inspector";
}

/**
 * Hamburg dashboard shell. Owns the tab state via `?tab=` search param
 * and lazy-loads the Historical + Container Inspector tabs.
 */
export function HamburgDashboard() {
  const [searchParams, setSearchParams] = useSearchParams();
  const raw = searchParams.get("tab");
  const tab: TabId = isTabId(raw) ? raw : "prediction";

  const handleTabChange = (next: TabId) => {
    setSearchParams(
      (prev) => {
        const p = new URLSearchParams(prev);
        p.set("tab", next);
        return p;
      },
      { replace: true },
    );
  };

  return (
    <div
      className="min-h-screen flex flex-col"
      style={{ background: "var(--bg-page)" }}
    >
      <Navigation backTo="/" centerLabel="Hamburg Dashboard" />
      <main
        className="flex-1 w-full mx-auto px-6 md:px-10 py-10"
        style={{ maxWidth: 1400 }}
      >
        <Tabs<TabId>
          aria-label="Dashboard sections"
          items={TAB_ITEMS}
          value={tab}
          onChange={handleTabChange}
        />

        {tab === "prediction" && <LoadPredictionTab />}
        {tab === "historical" && (
          <Suspense fallback={<LoadingSpinner label="Loading analytics" />}>
            <HistoricalAnalyticsTab />
          </Suspense>
        )}
        {tab === "inspector" && (
          <Suspense fallback={<LoadingSpinner label="Loading inspector" />}>
            <ContainerInspectorTab />
          </Suspense>
        )}
      </main>
    </div>
  );
}
