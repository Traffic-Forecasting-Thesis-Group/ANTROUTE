import { useEffect, useRef } from 'react';
import { X } from 'lucide-react';

import { DURATION_NOTE, type RouteOption } from '@/api/routeService';
import { cn } from '@/lib/cn';

import {
  cameraLabel,
  eventCell,
  floodCell,
  isRainy,
  levelChipClass,
  weatherCell,
} from './formatInputs';

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

function Chip({ text, className }: { text: string; className: string }) {
  return (
    <span className={cn('inline-block rounded-full px-2 py-0.5 text-xs font-medium', className)}>
      {text}
    </span>
  );
}

const CONGESTION_TONE = {
  clear: 'good',
  moderate: 'warn',
  heavy: 'bad',
  unknown: 'muted',
} as const;

/**
 * Route Information: the route's headline numbers and then its stretches, one row each.
 *
 * Every per-leg figure comes from the server's `legs`, measured over that leg's own nodes
 * with the same thresholds, ETA engine and recorded tables as the whole route -- so the rows
 * add up to the totals above them. Nothing here is derived in the browser, and nothing is a
 * live feed: the cameras, incidents, weather and flood are all recorded.
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
    // The page behind must not scroll while the dialog is open.
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
        className="flex max-h-[85vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl bg-white shadow-2xl"
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
                      <th className="px-3 py-2.5 font-semibold">CCTV Congestion</th>
                      <th className="px-3 py-2.5 font-semibold">Events</th>
                      <th className="px-3 py-2.5 font-semibold">Weather</th>
                      <th className="px-3 py-2.5 font-semibold">Flood</th>
                    </tr>
                  </thead>
                  <tbody>
                    {legs.map((leg) => {
                      const camera = cameraLabel(leg.inputs);
                      const legEvents = eventCell(leg.inputs);
                      return (
                        <tr key={leg.index} className="border-t border-slate-200 align-top">
                          <td className="px-3 py-2.5 text-slate-500">{leg.index}</td>
                          <td className="px-3 py-2.5">
                            <span className="block text-slate-800">
                              {leg.from_name} &rarr; {leg.to_name}
                            </span>
                            <span className="block text-xs text-slate-400">via {leg.road}</span>
                          </td>
                          <td className="px-3 py-2.5 text-right tabular-nums text-slate-700">
                            {leg.distance_km}
                          </td>
                          <td className="px-3 py-2.5 text-right tabular-nums text-slate-700">
                            {leg.duration_min} min
                          </td>
                          <td className="px-3 py-2.5">
                            {camera ? (
                              <Chip text={camera} className={levelChipClass(camera)} />
                            ) : (
                              <Chip
                                text={leg.congestion_level}
                                className={levelChipClass(leg.congestion_level)}
                              />
                            )}
                          </td>
                          <td className="px-3 py-2.5">
                            {legEvents ? (
                              <span className="font-medium text-red-600">{legEvents}</span>
                            ) : (
                              <span className="text-slate-400">&ndash;</span>
                            )}
                          </td>
                          <td className="px-3 py-2.5">
                            <Chip
                              text={weatherCell(leg.inputs?.weather)}
                              className={
                                isRainy(leg.inputs?.weather)
                                  ? 'bg-sky-100 text-sky-700'
                                  : 'bg-slate-100 text-slate-500'
                              }
                            />
                          </td>
                          <td className="px-3 py-2.5 text-slate-600">{floodCell(leg.inputs)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <p className="mt-3 text-xs leading-relaxed text-slate-500">
                A leg is one stretch on a named road; very short stretches are merged into the
                one before them. CCTV congestion is the recorded human label at the cameras on
                that leg, falling back to the model&rsquo;s level where no camera covers it.
                Est. time comes from the same estimator as the route total &mdash; estimated,
                not evaluated. Cameras, incidents, weather and flood are recorded data for the
                planned window, not a live feed.
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
