import { useState } from "react";
import { GlassPanel } from "@/components/GlassPanel";
import { PageHeader } from "@/components/PageHeader";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ErrorState } from "@/components/ErrorState";
import type { Horizon } from "@/types/api";
import { useForecast } from "./useForecast";
import { HorizonToggle } from "./HorizonToggle";
import { PredictionChart } from "./PredictionChart";
import { KpiRow } from "./KpiRow";

/**
 * First tab of the Hamburg dashboard: KPIs + 24h/14d prediction chart.
 */
export function LoadPredictionTab() {
  const [horizon, setHorizon] = useState<Horizon>(336);
  const { data, isLoading, isError, error, refetch } = useForecast(horizon);

  return (
    <div className="flex flex-col gap-6 page-enter">
      <PageHeader
        title="Reefer Peak Load Forecast"
        subtitle="Intelligent Horizon Planning"
        actions={<HorizonToggle value={horizon} onChange={setHorizon} />}
      />

      {isLoading ? (
        <GlassPanel>
          <LoadingSpinner label="Loading forecast" />
        </GlassPanel>
      ) : isError ? (
        <GlassPanel>
          <ErrorState
            title="Failed to load forecast"
            description={(error as Error)?.message}
            onRetry={() => refetch()}
          />
        </GlassPanel>
      ) : data ? (
        <>
          <KpiRow points={data.points} />
          <GlassPanel
            title="Power Demand"
            subtitle={horizon === 24 ? "Last 24 Hours" : "Next 14 Days"}
          >
            <PredictionChart
              points={data.points}
              compactLabels={horizon === 24}
            />
          </GlassPanel>
        </>
      ) : null}
    </div>
  );
}
