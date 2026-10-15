import { useEffect, useRef } from 'react';
import { X } from 'lucide-react';

import {
  DURATION_NOTE,
  RISK_FACTOR_KEYS,
  type RiskFactorKey,
  type RouteOption,
} from '@/api/routeService';
import { cn } from '@/lib/cn';

import { eventCell, isRainy, weatherCell } from './formatInputs';

function Stat({
  value,
  label,
  tone = 'default',
  note,
}: {
  value: string;
  label: string;
  tone?: 'default' | 'brand' | 'good' | 'warn' | 'bad' | 'muted';
  note?: string;
}) {
  return (
    <div className="flex-1 px-4 text-center">
      <p
        className={cn(
          'text-2xl font-bold',
          tone === 'brand' && 'text-brand',
          tone === 'good' && 'text-emerald-600',
          tone === 'warn' && 'text-amber-600',
          tone === 'bad' && 'text-red-600',
          tone === 'muted' && 'text-slate-400',
          tone === 'default' && 'text-slate-900'
        )}
      >
        {value}
      </p>
      <p className="mt-0.5 text-sm text-slate-500">{label}</p>
      {note && <p className="mt-0.5 text-[11px] italic text-slate-400">{note}</p>}
    </div>
  );
}

const CONGESTION_TONE = {
  clear: 'good',
  moderate: 'warn',
  heavy: 'bad',
  unknown: 'muted',
} as const;

const CONGESTION_DOT = {
  clear: 'bg-emerald-500',
  moderate: 'bg-amber-500',
  heavy: 'bg-red-500',
  unknown: 'bg-slate-300',
} as const;

const FACTOR_HEADS: Record<RiskFactorKey, string> = {
  cctv: 'CCTV',
  events: 'Events',
  weather: 'Weather',
  flood: 'Flood',
  others: 'Others',
};

/** A share as a plain percentage. Nothing measured means 0, never a blank. */
function Share({ value }: { value: number | undefined }) {
  const pct = (value ?? 0) * 100;
  const zero = pct < 0.05;
  return (
    <span
      className={cn('tabular-nums', zero ? 'text-slate-300' : 'font-medium text-slate-800')}
      title={zero ? 'Nothing from this input on this stretch' : undefined}
    >
      {zero ? '0%' : `${pct.toFixed(1)}%`}
    </span>
  );
}

/**
 * Route Information: the route's headline numbers, then its stretches one row each.
 *
 * The input columns are shares of that stretch's congestion risk, so each row adds to 100%.
 * "CCTV" is the part the camera-fed decoder produced -- a neural prediction, reported whole
 * because it cannot be split into inputs. The rest is the assumed tier, whose model is a
 * ridge regression and so decomposes exactly into its input groups; "Others" collects clock,
 * road class and text reports. Everything comes from the server's `legs`, measured over each
 * leg's own nodes with the same thresholds and tables as the whole route.
 */
export default function RouteInformationModal({
  route,
  onClose,
}: {
  route: RouteOption | null;
  onClose: () => void;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!route) return;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = previous;
    };
  }, [route, onClose]);

  if (!route) return null;

  const legs = route.legs ?? [];
  const routeInputs = route.inputs;
  const events = eventCell(routeInputs);
  const weather = weatherCell(routeInputs?.weather);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-6"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Route information"
        onClick={(e) => e.stopPropagation()}
        className="flex max-h-[85vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl bg-white shadow-2xl"
      >
        <header className="flex shrink-0 items-center justify-between border-b border-slate-200 px-7 py-5">
          <h2 className="text-xl font-bold text-slate-900">Route Information</h2>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-lg p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-700"
          >
            <X size={22} />
          </button>
        </header>

        <div className="overflow-y-auto px-7 py-5">
          <h3 className="text-lg font-semibold text-slate-900">Via {route.via}</h3>

          <div className="my-5 flex divide-x divide-slate-200">
            <Stat
              value={`${route.duration_min} min`}
              label="Travel Time"
              tone="brand"
              note={route.duration_note ?? DURATION_NOTE}
            />
            <Stat value={`${route.distance_km} km`} label="Distance" tone="brand" />
            <Stat
              value={route.congestion_level}
              label="Congestion"
              tone={CONGESTION_TONE[route.congestion_level]}
            />
            <Stat value={events ?? '–'} label="Event" tone={events ? 'bad' : 'muted'} />
            <Stat
              value={weather}
              label="Weather"
              tone={isRainy(routeInputs?.weather) ? 'brand' : 'muted'}
            />
          </div>

          {route.congestion_note && (
            <p className="mb-4 text-xs italic text-slate-400">{route.congestion_note}</p>
          )}

          <h3 className="mb-2 text-lg font-semibold text-slate-900">Route Details</h3>

          {legs.length > 0 ? (
            <>
              <div className="overflow-x-auto rounded-lg border border-slate-200">
                <table className="w-full border-collapse text-sm">
                  <thead>
                    <tr className="bg-brand-light text-left text-slate-700">
                      <th className="px-3 py-2.5 font-semibold">#</th>
                      <th className="px-3 py-2.5 font-semibold">From &rarr; To</th>
                      <th className="px-3 py-2.5 text-right font-semibold">Length (km)</th>
                      <th className="px-3 py-2.5 text-right font-semibold">Est. Time</th>
                      {RISK_FACTOR_KEYS.map((key) => (
                        <th key={key} className="px-3 py-2.5 text-right font-semibold">
                          {FACTOR_HEADS[key]}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {legs.map((leg) => (
                      <tr key={leg.index} className="border-t border-slate-200 align-top">
                        <td className="px-3 py-2.5 text-slate-500">{leg.index}</td>
                        <td className="px-3 py-2.5">
                          <span className="block text-slate-800">
                            {leg.from_name} &rarr; {leg.to_name}
                          </span>
                          <span className="mt-0.5 flex items-center gap-1.5 text-xs text-slate-400">
                            <span
                              className={cn(
                                'inline-block h-1.5 w-1.5 shrink-0 rounded-full',
                                CONGESTION_DOT[leg.congestion_level]
                              )}
                              aria-hidden
                            />
                            {leg.congestion_level} &middot; via {leg.road}
                          </span>
                        </td>
                        <td className="px-3 py-2.5 text-right tabular-nums text-slate-700">
                          {leg.distance_km}
                        </td>
                        <td className="px-3 py-2.5 text-right tabular-nums text-slate-700">
                          {leg.duration_min} min
                        </td>
                        {RISK_FACTOR_KEYS.map((key) => (
                          <td key={key} className="px-3 py-2.5 text-right">
                            <Share value={leg.risk_factors?.[key]} />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <p className="mt-3 text-xs leading-relaxed text-slate-500">
                A leg is one stretch on a named road; very short stretches are merged into the
                one before them. The last five columns are that stretch&rsquo;s congestion risk
                by where it came from, so each row adds to 100%. <strong>CCTV</strong> is the
                share the camera-fed model predicted &mdash; a neural prediction, so it is not
                broken down further. The remainder is estimated from the inputs every road has,
                with <strong>Others</strong> covering time of day, road class and traffic
                reports. Est. time comes from the same estimator as the route total &mdash;
                estimated, not evaluated. All of it is recorded data for the planned window,
                not a live feed.
              </p>
            </>
          ) : (
            <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
              No leg breakdown for this route. The server sends one once road names are loaded
              (<code>edge_names.csv</code>).
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
