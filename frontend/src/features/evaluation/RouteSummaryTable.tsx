import { useState } from 'react';
import { FileText } from 'lucide-react';

import type { RouteOption } from '@/api/routeService';
import { cn } from '@/lib/cn';
import RouteInformationModal from '@/features/trip/RouteInformationModal';
import { useTripPlanner } from '@/features/trip/TripPlannerContext';

const CONGESTION_CHIP = {
  clear: 'bg-emerald-100 text-emerald-700',
  moderate: 'bg-amber-100 text-amber-700',
  heavy: 'bg-red-100 text-red-700',
  unknown: 'bg-slate-100 text-slate-500',
} as const;

/** The routes ANTRoute returned for this trip, recommended first. */
export default function RouteSummaryTable() {
  const { antRouteOptions, selectedIndex, setSelectedIndex } = useTripPlanner();
  const [detailsFor, setDetailsFor] = useState<RouteOption | null>(null);

  const selected = antRouteOptions[selectedIndex];

  return (
    <section className="flex flex-col rounded-xl border border-slate-200 bg-white p-5 shadow-card">
      <h2 className="mb-3 text-sm font-semibold text-slate-700">Route Summary</h2>

      {antRouteOptions.length > 0 ? (
        <>
          <div className="overflow-hidden rounded-lg border border-slate-200">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="bg-slate-50 text-left text-slate-600">
                  <th className="px-3 py-2 font-semibold">Route</th>
                  <th className="px-3 py-2 text-right font-semibold">Travel Time</th>
                  <th className="px-3 py-2 text-right font-semibold">Distance</th>
                  <th className="px-3 py-2 font-semibold">Congestion</th>
                </tr>
              </thead>
              <tbody>
                {antRouteOptions.map((route, i) => (
                  <tr
                    key={i}
                    onClick={() => setSelectedIndex(i)}
                    className={cn(
                      'cursor-pointer border-t border-slate-200',
                      i === selectedIndex ? 'bg-brand-light/50' : 'hover:bg-slate-50'
                    )}
                  >
                    <td className="px-3 py-2 text-slate-700">
                      {i === 0 ? 'Recommended' : `Alternative ${i}`}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-slate-800">
                      {route.duration_min} min
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-slate-800">
                      {route.distance_km} km
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={cn(
                          'inline-block rounded-full px-2 py-0.5 text-xs font-medium capitalize',
                          CONGESTION_CHIP[route.congestion_level]
                        )}
                      >
                        {route.congestion_level}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="mt-2 text-[11px] text-slate-400">
            Travel time is estimated, not evaluated.
          </p>

          <button
            type="button"
            disabled={!selected}
            onClick={() => selected && setDetailsFor(selected)}
            className="mt-auto flex w-full items-center justify-center gap-2 rounded-lg bg-brand px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-brand-darker disabled:opacity-60"
          >
            <FileText size={15} aria-hidden />
            View Route Details
          </button>
        </>
      ) : (
        <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-6 text-center text-sm text-slate-500">
          No routes yet.
        </p>
      )}

      <RouteInformationModal route={detailsFor} onClose={() => setDetailsFor(null)} />
    </section>
  );
}
