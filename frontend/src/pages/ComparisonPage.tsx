import { Link } from 'react-router-dom';
import { Route as RouteIcon } from 'lucide-react';

import type { RouteOption } from '@/api/routeService';
import PageHeader from '@/app/PageHeader';
import { Button, Card, EmptyState, Spinner } from '@/components/ui';
import MetricsTable from '@/features/metrics/MetricsTable';
import { useTripPlanner } from '@/features/trip/TripPlannerContext';
import { baselineMethodLabel } from '@/lib/routeLabels';
import { ANT_COLOR, BASELINE_COLOR } from '@/lib/theme';

function ModelCard({
  name,
  method,
  accent,
  route,
  notice,
  optimality,
}: {
  name: string;
  method: string;
  accent: string;
  route: RouteOption | undefined;
  notice: string | null;
  optimality: number | null;
}) {
  return (
    <Card className="flex-1 p-5">
      <div className="mb-4 flex items-baseline justify-between gap-3">
        <span className="flex items-center gap-2">
          <span
            className="h-2.5 w-2.5 rounded-full"
            style={{ backgroundColor: accent }}
            aria-hidden
          />
          <span className="font-bold text-slate-900">{name}</span>
        </span>
        {route && <span className="text-sm text-slate-500">{route.distance_km} km</span>}
      </div>

      <p className="mb-4 text-xs text-slate-400">{method}</p>

      {route ? (
        <dl className="grid grid-cols-2 gap-4">
          <div>
            <dt className="text-xs font-medium text-slate-500">ETA</dt>
            <dd className="mt-0.5 text-2xl font-bold text-slate-900">
              {route.duration_min} <span className="text-base font-medium">min</span>
            </dd>
          </div>
          <div>
            <dt className="text-xs font-medium text-slate-500">Route Optimality</dt>
            <dd className="mt-0.5 text-2xl font-bold text-slate-900">
              {optimality != null ? `${Math.round(optimality)}%` : '–'}
            </dd>
          </div>
        </dl>
      ) : (
        <p className="rounded-lg bg-slate-50 px-3 py-4 text-sm text-slate-500">
          {notice ?? 'No route for this trip.'}
        </p>
      )}
    </Card>
  );
}

/**
 * ANTRoute against the baseline for the trip planned on the Dashboard. The planner state
 * lives above the router, so switching to this page does not re-plan anything.
 *
 * The metrics here are this trip's own evaluation against Apple Maps: only trips that were
 * timed in Apple Maps have one, and any other trip shows dashes rather than borrowing
 * another trip's numbers. The whole-test-set evaluation is on the Evaluation page.
 */
export default function ComparisonPage() {
  const {
    hasPlannedTrip,
    antTop,
    baselineTop,
    baselineNotice,
    tripEvaluation,
    tripEvaluationError,
    tripMetricRows,
  } = useTripPlanner();

  const optimality = tripEvaluation?.status === 'evaluated' ? tripEvaluation.routeOptimality : null;

  return (
    <>
      <PageHeader
        title="Comparison"
        subtitle="ANTRoute against the baseline for the trip planned on the Dashboard."
      />

      <div className="flex-1 overflow-y-auto p-6">
        {!hasPlannedTrip ? (
          <div className="flex flex-col items-center gap-4 py-16">
            <EmptyState
              title="No trip planned yet"
              description="Plan a trip on the Dashboard and its comparison appears here."
              icon={<RouteIcon size={28} />}
            />
            <Link to="/">
              <Button>Go to Dashboard</Button>
            </Link>
          </div>
        ) : (
          <div className="max-w-4xl">
            <h2 className="mb-3 text-sm font-semibold text-slate-700">Comparison Result</h2>
            <div className="flex gap-4">
              <ModelCard
                name="ANTRoute"
                method="Ant colony optimization on predicted congestion risk"
                accent={ANT_COLOR}
                route={antTop}
                notice={null}
                optimality={optimality?.antroute ?? null}
              />
              <ModelCard
                name="Baseline"
                method={baselineMethodLabel(baselineTop)}
                accent={BASELINE_COLOR}
                route={baselineTop}
                notice={baselineNotice}
                optimality={optimality?.baseline ?? null}
              />
            </div>

            <h2 className="mb-3 mt-8 text-sm font-semibold text-slate-700">Evaluation Metrics</h2>
            {tripEvaluationError ? (
              <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
                {tripEvaluationError}
              </p>
            ) : !tripEvaluation ? (
              <div className="flex justify-center py-10">
                <Spinner className="h-7 w-7 text-brand" />
              </div>
            ) : (
              <Card className="p-5">
                <MetricsTable
                  rows={tripMetricRows}
                  baselineName={tripEvaluation.baselineName}
                />
                <p className="mt-3 text-xs leading-relaxed text-slate-500">
                  {tripEvaluation.status === 'evaluated'
                    ? 'Lower is better for MAE, RMSE, MSE, MAPE. Higher is better for Route Optimality and R².'
                    : tripEvaluation.message}
                </p>
              </Card>
            )}
          </div>
        )}
      </div>
    </>
  );
}
