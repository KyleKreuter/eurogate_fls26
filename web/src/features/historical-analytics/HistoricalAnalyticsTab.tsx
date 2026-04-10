import { GlassPanel } from "@/components/GlassPanel";
import { PageHeader } from "@/components/PageHeader";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ErrorState } from "@/components/ErrorState";
import { useOverviewAnalytics } from "./useOverviewAnalytics";
import { ActivePerDayChart } from "./ActivePerDayChart";
import { HardwareDoughnut } from "./HardwareDoughnut";
import { DurationHistogramChart } from "./DurationHistogramChart";
import { ContainerSizeChart } from "./ContainerSizeChart";
import { MonthlyEnergyChart } from "./MonthlyEnergyChart";
import { SetpointDistChart } from "./SetpointDistChart";
import { HourDowHeatmap } from "./HourDowHeatmap";
import { CalendarHeatmap } from "./CalendarHeatmap";

/**
 * Second tab of the Hamburg dashboard — fleet-wide analytics.
 * 6 Recharts visualisations + 2 custom SVG heatmaps, all fed from
 * the single /api/overview-analytics query (cached for session).
 */
export function HistoricalAnalyticsTab() {
  const { data, isLoading, isError, error, refetch } = useOverviewAnalytics();

  if (isLoading) {
    return (
      <GlassPanel>
        <LoadingSpinner label="Crunching fleet analytics" />
      </GlassPanel>
    );
  }

  if (isError || !data) {
    return (
      <GlassPanel>
        <ErrorState
          title="Failed to load analytics"
          description={(error as Error)?.message}
          onRetry={() => refetch()}
        />
      </GlassPanel>
    );
  }

  return (
    <div className="flex flex-col gap-6 page-enter">
      <PageHeader
        title="Historical Analytics"
        subtitle="Fleet-wide Insights"
      />

      <GlassPanel title="Active Containers per Day" subtitle="Distinct visit count">
        <ActivePerDayChart data={data.active_per_day} />
      </GlassPanel>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <GlassPanel title="Hardware Distribution" subtitle="Top 7 + Other">
          <HardwareDoughnut data={data.hardware_types} />
        </GlassPanel>
        <GlassPanel title="Visit Duration" subtitle="Spread across bins">
          <DurationHistogramChart data={data.duration_hist} />
        </GlassPanel>
        <GlassPanel title="Container Size" subtitle="vs. avg power draw">
          <ContainerSizeChart data={data.container_sizes} />
        </GlassPanel>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <GlassPanel title="Monthly Energy" subtitle="Consumption (MWh)">
          <MonthlyEnergyChart data={data.monthly_energy} />
        </GlassPanel>
        <GlassPanel title="Setpoint Distribution" subtitle="5° temperature bins">
          <SetpointDistChart data={data.setpoint_dist} />
        </GlassPanel>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <GlassPanel title="Hour × Day-of-Week" subtitle="Activity pattern">
          <HourDowHeatmap data={data.hourly_heatmap} />
        </GlassPanel>
        <GlassPanel title="Annual Calendar" subtitle="Daily activity (synthesized)">
          <CalendarHeatmap activePerDay={data.active_per_day} />
        </GlassPanel>
      </div>
    </div>
  );
}
