import { useCallback, useEffect, useMemo, useState } from 'react';
import { CarFront, FlaskConical, RotateCcw, ScrollText, Table2 } from 'lucide-react';

import {
  getComparisonMetrics,
  getComparisonTripDetail,
  getComparisonTrips,
  type ComparisonMetrics,
  type ComparisonTripDetail,
  type ComparisonTripsResponse,
  type ConditionUpdateKind,
} from '@/api/routeService';
import { Button, EmptyState, Spinner } from '@/components/ui';
import { cn } from '@/lib/cn';
import DifferenceChart from '@/features/comparison/DifferenceChart';
import MetricComparisonTable, {
  responseTimeRow,
  toRows,
  type Row,
} from '@/features/comparison/MetricComparisonTable';
import SystemRouteMaps from '@/features/comparison/SystemRouteMaps';

type Tab = 'metrics' | 'statistical' | 'summary';

const SCENARIOS = [
  { key: 'overall', label: 'All scenarios' },
  { key: 'single_destination', label: 'Single destination' },
  { key: 'multi_destination', label: 'Multi-destination' },
  { key: 'recommended', label: 'Recommended' },
  { key: 'incident_exposed', label: 'Incident-exposed' },
];

function dateOnly(iso: string | null | undefined): string {
  if (!iso) return '–';
  return iso.slice(0, 10);
}

/**
 * ANTRoute against the Improved ACO baseline over the same test trips (research question 3).
 *
 * Every number is read from the evaluation outputs: the paired Wilcoxon tests with Holm
 * correction, the routes each system actually planned, and the measured recalculation
 * response time. Nothing is routed or re-tested in the browser.
 */
export default function ComparisonPage() {
  const [tab, setTab] = useState<Tab>('metrics');
  const [scenario, setScenario] = useState('overall');
  const [update, setUpdate] = useState<ConditionUpdateKind>('event');

  const [metrics, setMetrics] = useState<ComparisonMetrics | null>(null);
  const [trips, setTrips] = useState<ComparisonTripsResponse | null>(null);
  const [tripId, setTripId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ComparisonTripDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [m, t] = await Promise.all([
        getComparisonMetrics(),
        getComparisonTrips(scenario, update),
      ]);
      setMetrics(m);
      setTrips(t);
      setTripId((current) =>
        current && t.trips.some((x) => x.trip_id === current) ? current : (t.trips[0]?.trip_id ?? null)
      );
    } catch (e: any) {
      setError(e?.message || 'Could not load the comparison.');
      setMetrics(null);
      setTrips(null);
    } finally {
      setLoading(false);
    }
  }, [scenario, update]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!tripId) return;
    let cancelled = false;
    setDetailLoading(true);
    getComparisonTripDetail(tripId)
      .then((d) => !cancelled && setDetail(d))
      .catch(() => !cancelled && setDetail(null))
      .finally(() => !cancelled && setDetailLoading(false));
    return () => {
      cancelled = true;
    };
  }, [tripId]);

  const shown = useMemo(
    () =>
      scenario === 'overall'
        ? metrics?.metrics
        : metrics?.scenarios.find((s) => s.scenario === scenario)?.metrics,
    [metrics, scenario]
  );
  const nTrials = useMemo(
    () =>
      scenario === 'overall'
        ? (metrics?.scenarios.find((s) => s.scenario === 'overall')?.nTrials ??
          trips?.trips.length ??
          0)
        : (metrics?.scenarios.find((s) => s.scenario === scenario)?.nTrials ?? 0),
    [metrics, scenario, trips]
  );

  const rows: Row[] = useMemo(() => {
    const base = shown ? toRows(shown) : [];
    const rt = responseTimeRow(trips?.response_time ?? null);
    return rt ? [...base, rt] : base;
  }, [shown, trips]);

  const baselineName = metrics?.baselineName ?? 'Improved ACO (Cheng 2023)';
  const provenance = trips?.provenance;

  return (
    <div className="flex flex-1 bg-slate-50">
      <aside className="flex w-[220px] shrink-0 flex-col border-r border-slate-200 bg-white px-6 py-6">
        <p className="mt-auto flex items-start gap-1.5 text-[11px] text-slate-400">
          <CarFront size={13} className="mt-0.5 shrink-0" aria-hidden />
          Routes optimized for private 4-wheel vehicles.
        </p>
      </aside>

      <div className="min-w-0 flex-1 overflow-y-auto px-6 py-6">
        <div className="mb-1 flex items-baseline gap-3">
          <h1 className="text-xl font-bold text-slate-900">Comparison</h1>
          <p className="text-sm text-slate-500">
            Against {baselineName} on the same test trips.
          </p>
        </div>

        {/* Sub-tabs */}
        <div className="mb-4 mt-3 flex gap-1 border-b border-slate-200">
          {(
            [
              { key: 'metrics', label: 'Performance Metrics', icon: Table2 },
              { key: 'statistical', label: 'Statistical Test', icon: FlaskConical },
              { key: 'summary', label: 'Summary', icon: ScrollText },
            ] as const
          ).map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              type="button"
              onClick={() => setTab(key)}
              className={cn(
                '-mb-px flex items-center gap-1.5 border-b-2 px-4 py-2 text-sm font-medium transition-colors',
                tab === key
                  ? 'border-brand text-brand'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              )}
            >
              <Icon size={14} aria-hidden />
              {label}
            </button>
          ))}
        </div>

        {/* Controls */}
        <div className="mb-4 flex flex-wrap items-end gap-4 rounded-xl border border-slate-200 bg-white p-4 shadow-card">
          <label className="text-xs text-slate-500">
            <span className="mb-1 block font-medium text-slate-600">Test scenario</span>
            <select
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs text-slate-700 focus:border-brand focus:outline-none"
            >
              {SCENARIOS.map((s) => (
                <option key={s.key} value={s.key}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>

          <label className="text-xs text-slate-500">
            <span className="mb-1 block font-medium text-slate-600">Response time from</span>
            <select
              value={update}
              onChange={(e) => setUpdate(e.target.value as ConditionUpdateKind)}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs text-slate-700 focus:border-brand focus:outline-none"
            >
              {(trips?.available_updates ?? ['event']).map((u) => (
                <option key={u} value={u}>
                  {u === 'event' ? 'Road event' : u === 'traffic' ? 'Traffic update' : 'Session change'}
                </option>
              ))}
            </select>
          </label>

          {/* Provenance, not a filter: every test trip is on the same recorded day. */}
          {provenance && (
            <div className="min-w-0 flex-1 text-[11px] leading-relaxed text-slate-500">
              <span className="font-medium text-slate-600">Evaluation run:</span>{' '}
              {provenance.n_trips} trips on recorded traffic from{' '}
              {dateOnly(provenance.recorded_window_first)}, timed in {provenance.reference}{' '}
              for a {dateOnly(provenance.reference_departure_first)} departure, queried{' '}
              {dateOnly(provenance.reference_queried_first)}. {provenance.scope}.
            </div>
          )}

          <Button variant="secondary" onClick={load} disabled={loading}>
            {loading ? <Spinner /> : <RotateCcw size={15} aria-hidden />}
            Run comparison
          </Button>
        </div>

        {loading && !metrics ? (
          <div className="flex justify-center py-20">
            <Spinner className="h-8 w-8 text-brand" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center gap-4 py-16">
            <EmptyState title="Could not load the comparison" description={error} />
            <Button onClick={load}>Retry</Button>
          </div>
        ) : (
          <>
            {tab === 'metrics' && (
              <>
                <section className="mb-5 rounded-xl border border-slate-200 bg-white p-5 shadow-card">
                  <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                    <h2 className="text-sm font-semibold text-slate-700">
                      The route each system planned
                    </h2>
                    <label className="flex items-center gap-2 text-xs text-slate-500">
                      Trip
                      <select
                        value={tripId ?? ''}
                        onChange={(e) => setTripId(e.target.value)}
                        className="rounded-lg border border-slate-300 bg-white px-2.5 py-1 text-xs text-slate-700 focus:border-brand focus:outline-none"
                      >
                        {(trips?.trips ?? []).map((t) => (
                          <option key={t.trip_id} value={t.trip_id}>
                            {t.trip_id} &mdash; {t.stops}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>

                  {detailLoading && !detail ? (
                    <div className="flex justify-center py-16">
                      <Spinner className="h-7 w-7 text-brand" />
                    </div>
                  ) : detail ? (
                    <>
                      <SystemRouteMaps detail={detail} baselineName={baselineName} />
                      <p className="mt-2 text-[11px] text-slate-400">
                        Trip {detail.trip.trip_id} &middot; {detail.trip.stops} &middot;
                        departing {detail.trip.window.slice(11, 16)} on{' '}
                        {dateOnly(detail.trip.window)}
                      </p>
                    </>
                  ) : (
                    <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-6 text-center text-sm text-slate-500">
                      Could not draw this trip.
                    </p>
                  )}
                </section>

                <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-card">
                  <h2 className="mb-1 text-sm font-semibold text-slate-700">Metric comparison</h2>
                  <p className="mb-3 text-[11px] text-slate-400">
                    Means over {nTrials} test trips in this scenario.
                  </p>
                  <MetricComparisonTable rows={rows} baselineName={baselineName} />
                </section>
              </>
            )}

            {tab === 'statistical' && (
              <>
                <section className="mb-5 rounded-xl border border-slate-200 bg-white p-5 shadow-card">
                  <h2 className="mb-1 text-sm font-semibold text-slate-700">Paired results</h2>
                  <p className="mb-3 text-[11px] text-slate-400">
                    Every trip routed by both systems on the same stops and departure, then
                    compared pairwise over {nTrials} trips.
                  </p>
                  <MetricComparisonTable rows={rows} baselineName={baselineName} showPValue />
                </section>

                <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-card">
                  <h2 className="mb-1 text-sm font-semibold text-slate-700">
                    Performance difference (ANTRoute vs {baselineName})
                  </h2>
                  <p className="mb-3 text-[11px] text-slate-400">
                    Relative difference per tested metric; * marks a significant result.
                  </p>
                  <DifferenceChart rows={rows} />
                  <p className="mt-3 text-[11px] text-slate-500">
                    Wilcoxon signed-rank test with Holm correction (&alpha; = 0.05).
                  </p>
                </section>
              </>
            )}

            {tab === 'summary' && (
              <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-card">
                <h2 className="mb-3 text-sm font-semibold text-slate-700">
                  What the comparison shows
                </h2>
                <ul className="flex flex-col gap-3">
                  {rows.map((row) => {
                    const tested = row.improvementPct != null;
                    const even = Math.abs(row.improvementPct ?? 0) < 0.05;
                    const better = (row.improvementPct ?? 0) > 0;
                    return (
                      <li key={row.metric} className="flex gap-3 text-sm">
                        <span
                          className={cn(
                            'mt-0.5 h-2 w-2 shrink-0 rounded-full',
                            !tested || even
                              ? 'bg-slate-300'
                              : row.significant
                                ? better
                                  ? 'bg-emerald-500'
                                  : 'bg-red-500'
                                : 'bg-amber-400'
                          )}
                          aria-hidden
                        />
                        <span className="text-slate-700">
                          <span className="font-semibold">{row.metric}:</span>{' '}
                          {!tested ? (
                            <>
                              {row.antroute} against {row.baseline}. Descriptive &mdash; not
                              tested.
                            </>
                          ) : even ? (
                            <>
                              ANTRoute {row.antroute}, {baselineName} {row.baseline} &mdash; no
                              measurable difference,{' '}
                              {row.significant ? 'significant' : 'not significant'}
                              {row.pValue != null
                                ? ` (p = ${row.pValue < 0.001 ? '<0.001' : row.pValue.toFixed(3)})`
                                : ''}
                              .
                            </>
                          ) : (
                            <>
                              ANTRoute {row.antroute}, {baselineName} {row.baseline} &mdash;{' '}
                              {better ? 'ANTRoute better' : 'the baseline better'} by{' '}
                              {Math.abs(row.improvementPct as number)}%,{' '}
                              {row.significant ? 'significant' : 'not significant'}
                              {row.pValue != null
                                ? ` (p = ${row.pValue < 0.001 ? '<0.001' : row.pValue.toFixed(3)})`
                                : ''}
                              .
                            </>
                          )}
                        </span>
                      </li>
                    );
                  })}
                </ul>
                <p className="mt-4 text-[11px] leading-relaxed text-slate-500">
                  Over {nTrials} test trips, Wilcoxon signed-rank with Holm correction at
                  &alpha; = 0.05. Travel time is absent throughout: it is not a thesis metric
                  and was never scored against a reference.
                </p>
              </section>
            )}
          </>
        )}
      </div>
    </div>
  );
}
