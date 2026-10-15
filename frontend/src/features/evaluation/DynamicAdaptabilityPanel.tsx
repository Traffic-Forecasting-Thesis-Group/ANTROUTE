import { useCallback, useEffect, useState } from 'react';
import { Gauge, Repeat, Timer, TriangleAlert } from 'lucide-react';

import {
  getConditionUpdateDetail,
  getConditionUpdates,
  type ConditionUpdateDetail,
  type ConditionUpdateKind,
  type ConditionUpdatesResponse,
} from '@/api/routeService';
import { DURATION_NOTE } from '@/api/routeService';
import { EmptyState, Spinner } from '@/components/ui';
import { cn } from '@/lib/cn';

import BeforeAfterMaps from './BeforeAfterMaps';
import MetricCard, { type Verdict } from './MetricCard';

const KINDS: { key: ConditionUpdateKind; label: string }[] = [
  { key: 'traffic', label: 'Traffic update' },
  { key: 'event', label: 'Road event' },
  { key: 'session', label: 'Session change' },
];

function pct(value: number | null | undefined): string {
  return value == null ? '–' : `${Math.round(value * 100)}%`;
}

/** Exposure improvement as a share of what the old route was exposed to. */
function improvementPct(before: number | null, improvement: number | null): number | null {
  if (before == null || improvement == null || before === 0) return null;
  return (improvement / before) * 100;
}

function clockOf(iso: string | null | undefined): string {
  if (!iso) return '–';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const h = d.getHours();
  return `${h % 12 || 12}:${String(d.getMinutes()).padStart(2, '0')} ${h < 12 ? 'AM' : 'PM'}`;
}

/**
 * Dynamic adaptability -- research question 2.
 *
 * Everything here comes from the offline condition-update harness: a test trip planned under
 * the conditions of one moment, then replanned under the conditions after an update, with both
 * routes judged by the conditions the driver actually meets. Response times were measured
 * there, on one machine in one run; replanning inside a web request would time the server's
 * load instead of the method.
 */
export default function DynamicAdaptabilityPanel() {
  const [kind, setKind] = useState<ConditionUpdateKind>('event');
  const [data, setData] = useState<ConditionUpdatesResponse | null>(null);
  const [tripId, setTripId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ConditionUpdateDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState('');

  const load = useCallback(async (next: ConditionUpdateKind) => {
    setLoading(true);
    setError('');
    setDetail(null);
    try {
      const response = await getConditionUpdates(next);
      setData(response);
      // Open on a case where the route actually changed -- that is what the panel is about.
      const first = response.cases.find((c) => c.route_changed) ?? response.cases[0];
      setTripId(first?.trip_id ?? null);
    } catch (e: any) {
      setData(null);
      setTripId(null);
      setError(e?.message || 'Could not load the condition-update results.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(kind);
  }, [kind, load]);

  useEffect(() => {
    if (!tripId) return;
    let cancelled = false;
    setDetailLoading(true);
    getConditionUpdateDetail(tripId, kind)
      .then((d) => !cancelled && setDetail(d))
      .catch(() => !cancelled && setDetail(null))
      .finally(() => !cancelled && setDetailLoading(false));
    return () => {
      cancelled = true;
    };
  }, [tripId, kind]);

  if (loading && !data) {
    return (
      <div className="flex justify-center py-20">
        <Spinner className="h-8 w-8 text-brand" />
      </div>
    );
  }

  if (error) {
    return <EmptyState title="No adaptability results" description={error} />;
  }
  if (!data) return null;

  const summary = data.summary;
  const current = data.cases.find((c) => c.trip_id === tripId) ?? null;
  const congestionDelta = current
    ? improvementPct(current.old_congestion_exposure, current.congestion_exposure_improvement)
    : null;
  const etaDelta =
    detail?.old_duration_min != null && detail?.recalculated_duration_min != null
      ? detail.recalculated_duration_min - detail.old_duration_min
      : null;
  // The exposures are scored; lower is better, and improvement is already old - recalculated.
  const congestionVerdict: Verdict =
    congestionDelta == null ? null : congestionDelta > 0.5 ? 'better' : congestionDelta < -0.5 ? 'worse' : 'even';

  return (
    <>
      {/* Which condition changed. */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {KINDS.filter((k) => data.available_updates.includes(k.key)).map((k) => (
          <button
            key={k.key}
            type="button"
            onClick={() => setKind(k.key)}
            className={cn(
              'rounded-full border px-3.5 py-1.5 text-xs font-medium transition-colors',
              kind === k.key
                ? 'border-brand bg-brand text-white'
                : 'border-slate-300 bg-white text-slate-600 hover:bg-slate-50'
            )}
          >
            {k.label}
          </button>
        ))}
        <span className="ml-auto text-[11px] text-slate-400">
          {summary.n_trips} trips &middot; route changed in {summary.n_route_changed} (
          {pct(summary.share_route_changed)})
        </span>
      </div>

      {/* Before / after. */}
      <section className="mb-5 rounded-xl border border-slate-200 bg-white p-5 shadow-card">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-sm font-semibold text-slate-700">Before and After Comparison</h2>
          <label className="flex items-center gap-2 text-xs text-slate-500">
            Case
            <select
              value={tripId ?? ''}
              onChange={(e) => setTripId(e.target.value)}
              className="rounded-lg border border-slate-300 bg-white px-2.5 py-1 text-xs text-slate-700 focus:border-brand focus:outline-none"
            >
              {data.cases.map((c) => (
                <option key={c.trip_id} value={c.trip_id}>
                  {c.trip_id}
                  {c.route_changed ? ' — route changed' : ' — unchanged'}
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
          <BeforeAfterMaps detail={detail} />
        ) : (
          <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-6 text-center text-sm text-slate-500">
            Could not draw this case.
          </p>
        )}
      </section>

      {/* The four RQ2 measures for the selected case. */}
      <h2 className="mb-3 text-sm font-semibold text-slate-700">Adaptability Metrics</h2>
      <div className="mb-5 grid grid-cols-4 gap-4">
        <MetricCard
          icon={<Timer size={14} />}
          label="Recalculation Response Time"
          value={
            current?.recalculation_seconds != null
              ? `${current.recalculation_seconds.toFixed(2)} sec`
              : '–'
          }
          note={
            summary.response_time_s?.median != null
              ? `measured · median ${summary.response_time_s.median.toFixed(2)}s, p95 ${summary.response_time_s.p95?.toFixed(2)}s`
              : 'measured in the harness'
          }
        />

        <MetricCard
          icon={<Repeat size={14} />}
          label="Route Changed"
          value={current ? (current.route_changed ? 'Yes' : 'No') : '–'}
          note={
            current?.shared_length_fraction != null
              ? `route overlap: ${pct(current.shared_length_fraction)}`
              : null
          }
        />

        <MetricCard
          icon={<Gauge size={14} />}
          label="Congestion Exposure"
          value={congestionDelta == null ? '–' : `${congestionDelta > 0 ? '-' : '+'}${Math.abs(congestionDelta).toFixed(1)}%`}
          note={
            current?.old_congestion_exposure != null
              ? `lower is better · ${current.old_congestion_exposure.toFixed(1)} → ${current.recalculated_congestion_exposure?.toFixed(1)}`
              : 'lower is better'
          }
          verdict={congestionVerdict}
          arrow={congestionDelta == null ? 'none' : congestionDelta > 0 ? 'down' : 'up'}
        />

        {/* No verdict badge: travel time is not a scored metric anywhere in this app. */}
        <MetricCard
          icon={<Timer size={14} />}
          label="Estimated Travel Time"
          value={
            etaDelta == null
              ? '–'
              : `${etaDelta > 0 ? '+' : ''}${etaDelta} min`
          }
          note={
            detail?.old_duration_min != null
              ? `${detail.old_duration_min} → ${detail.recalculated_duration_min} min · ${DURATION_NOTE}`
              : DURATION_NOTE
          }
        />
      </div>

      {/* What changed. */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-card">
        <h2 className="mb-3 text-sm font-semibold text-slate-700">Event Details</h2>
        {detail && detail.incidents.length > 0 ? (
          <div className="overflow-hidden rounded-lg border border-slate-200">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="bg-slate-50 text-left text-slate-600">
                  <th className="px-3 py-2 font-semibold">Type</th>
                  <th className="px-3 py-2 font-semibold">Location</th>
                  <th className="px-3 py-2 font-semibold">In force after</th>
                  <th className="px-3 py-2 font-semibold">Time</th>
                </tr>
              </thead>
              <tbody>
                {detail.incidents.map((i) => (
                  <tr key={`${i.location}-${i.reported_at}`} className="border-t border-slate-200">
                    <td className="px-3 py-2">
                      <span className="inline-flex items-center gap-1.5 text-slate-700">
                        <TriangleAlert size={13} className="text-red-600" aria-hidden />
                        {i.kind}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-slate-700">{i.location}</td>
                    <td className="px-3 py-2 text-slate-600">{i.in_force_after}</td>
                    <td className="px-3 py-2 text-slate-600">{clockOf(i.reported_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-600">
            {kind === 'event'
              ? 'No incident recorded for this case.'
              : 'This update changed the recorded traffic conditions, not an incident — the routes were replanned on a different recorded window.'}
          </p>
        )}

        <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 text-[11px] leading-relaxed text-slate-500">
          <div>
            <dt className="font-semibold text-slate-600">Update</dt>
            <dd>{summary.update_label}</dd>
          </div>
          <div>
            <dt className="font-semibold text-slate-600">Response time</dt>
            <dd>{summary.response_time_definition}</dd>
          </div>
          <div>
            <dt className="font-semibold text-slate-600">Route overlap</dt>
            <dd>{summary.shared_length_definition}</dd>
          </div>
          <div>
            <dt className="font-semibold text-slate-600">Quality</dt>
            <dd>{summary.quality_definition}</dd>
          </div>
        </dl>
      </section>
    </>
  );
}
