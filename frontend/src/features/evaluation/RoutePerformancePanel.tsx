import { useCallback, useEffect, useState } from 'react';
import {
  CloudRain,
  Gauge,
  Info,
  Route as RouteIcon,
  Ruler,
  Timer,
  TriangleAlert,
} from 'lucide-react';

import { getComparisonMetrics, type ComparisonMetrics } from '@/api/routeService';
import { DURATION_NOTE } from '@/api/routeService';
import { EmptyState, Spinner } from '@/components/ui';
import { useTripPlanner } from '@/features/trip/TripPlannerContext';
import { weatherCell, weatherLine } from '@/features/trip/formatInputs';

import MetricCard, { type Verdict } from './MetricCard';
import RoutePerformanceChart from './RoutePerformanceChart';
import RouteSummaryTable from './RouteSummaryTable';

/** Better/worse from the server's Δ%, which is already signed so positive = ANTRoute better. */
function verdictOf(improvementPct: number | null | undefined): Verdict {
  if (improvementPct == null) return null;
  if (improvementPct > 0.05) return 'better';
  if (improvementPct < -0.05) return 'worse';
  return 'even';
}

export default function RoutePerformancePanel() {
  const { selectedRoute, tripEvaluation, tripEvaluationError, hasPlannedTrip } = useTripPlanner();

  const [testSet, setTestSet] = useState<ComparisonMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setTestSet(await getComparisonMetrics());
    } catch (e: any) {
      setError(e?.message || 'Could not load the scenario results.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (!hasPlannedTrip) {
    return (
      <EmptyState
        title="No trip to evaluate yet"
        description="Plan a trip on Home. Its route performance against the Improved ACO baseline appears here."
        icon={<RouteIcon size={28} />}
      />
    );
  }

  const metricRow = (name: string) => tripEvaluation?.metrics.find((m) => m.metric === name);
  const optimality = metricRow('Route Optimality');
  const congestion = metricRow('Congestion exposure');
  const event = metricRow('Event exposure');
  const evaluated = tripEvaluation?.status === 'evaluated';
  const weather = selectedRoute?.inputs?.weather;

  return (
    <>
      <h2 className="mb-3 text-sm font-semibold text-slate-700">Route Performance Metrics Grid</h2>

      {/* Scored metrics only exist for the evaluated test trips; the server says why not. */}
      {!evaluated && (
        <p className="mb-4 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <Info size={16} className="mt-0.5 shrink-0" aria-hidden />
          <span>
            {tripEvaluationError ||
              tripEvaluation?.message ||
              'This trip has no evaluation of its own.'}
          </span>
        </p>
      )}

      <div className="mb-6 grid grid-cols-3 gap-4">
        <MetricCard
          icon={<Gauge size={14} />}
          label="Route Optimality"
          value={optimality?.antroute ?? '–'}
          note={
            optimality
              ? `vs ${tripEvaluation?.baselineName ?? 'baseline'}: ${optimality.baseline}`
              : 'Only the evaluated test trips have this.'
          }
          verdict={verdictOf(optimality?.improvementPct)}
          verdictLabel={
            optimality?.improvementPct != null
              ? `${optimality.improvementPct > 0 ? '+' : ''}${optimality.improvementPct}%`
              : undefined
          }
          muted={!optimality}
        />

        <MetricCard
          icon={<Gauge size={14} />}
          label="Congestion Exposure"
          value={congestion?.antroute ?? '–'}
          note={
            congestion
              ? `lower is better · vs baseline: ${congestion.baseline}`
              : 'Only the evaluated test trips have this.'
          }
          verdict={verdictOf(congestion?.improvementPct)}
          verdictLabel={
            congestion?.improvementPct != null
              ? `${congestion.improvementPct > 0 ? '+' : ''}${congestion.improvementPct}%`
              : undefined
          }
          arrow={
            congestion?.improvementPct == null ? 'none' : congestion.improvementPct > 0 ? 'down' : 'up'
          }
          muted={!congestion}
        />

        {/* No badge: ETA is not a thesis metric and was never scored against a reference. */}
        <MetricCard
          icon={<Timer size={14} />}
          label="Estimated Travel Time"
          value={selectedRoute ? `${selectedRoute.duration_min} min` : '–'}
          note={selectedRoute?.duration_note ?? DURATION_NOTE}
        />

        {/* No badge: route length is descriptive in the evaluation, never tested. */}
        <MetricCard
          icon={<Ruler size={14} />}
          label="Route Distance"
          value={selectedRoute ? `${selectedRoute.distance_km} km` : '–'}
          note="descriptive, not tested"
        />

        <MetricCard
          icon={<TriangleAlert size={14} />}
          label="Event Exposure"
          value={event?.antroute ?? '–'}
          note={
            event
              ? `lower is better · vs baseline: ${event.baseline}`
              : 'Only the evaluated test trips have this.'
          }
          verdict={verdictOf(event?.improvementPct)}
          verdictLabel={
            event?.improvementPct != null
              ? `${event.improvementPct > 0 ? '+' : ''}${event.improvementPct}%`
              : undefined
          }
          arrow={
            event?.improvementPct == null ? 'none' : event.improvementPct > 0 ? 'down' : 'up'
          }
          muted={!event}
        />

        {/* An input to the model, not a result -- so no better/worse badge. */}
        <MetricCard
          icon={<CloudRain size={14} />}
          label="Weather"
          value={weatherCell(weather)}
          note={weather ? `recorded · ${weatherLine(weather)}` : 'No weather recorded.'}
        />
      </div>

      <div className="grid grid-cols-[1.3fr_1fr] gap-5">
        <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-card">
          <h2 className="mb-1 text-sm font-semibold text-slate-700">
            Route Performance Comparison
          </h2>
          <p className="mb-3 text-[11px] text-slate-400">
            Across every evaluated test trip, by scenario &mdash; not this trip alone.
          </p>

          {loading && !testSet ? (
            <div className="flex justify-center py-12">
              <Spinner className="h-7 w-7 text-brand" />
            </div>
          ) : error ? (
            <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
              {error}
            </p>
          ) : testSet ? (
            <RoutePerformanceChart
              scenarios={testSet.scenarios}
              baselineName={testSet.baselineName}
            />
          ) : null}
        </section>

        <RouteSummaryTable />
      </div>
    </>
  );
}
