import { Navigation, X } from 'lucide-react';

import { useTripPlanner } from '@/features/trip/TripPlannerContext';
import type { TurnByTurnState } from '@/features/navigation/useTurnByTurn';
import { getArrivalTime } from '@/lib/time';

/**
 * The trip summary that floats over the bottom-left of the map: remaining time and
 * distance, and the Start / End control. Replaces the summary bar that sat in the phone
 * build's bottom sheet.
 */
export default function TripControls({ nav }: { nav: TurnByTurnState }) {
  const { selectedRoute, departAt } = useTripPlanner();

  if (!selectedRoute) return null;

  const { isNavigating, arrived, remainingMin, remainingKm, error } = nav;

  return (
    <div className="pointer-events-auto flex flex-col gap-2">
      {error && (
        <p className="max-w-xs rounded-lg bg-red-600 px-3 py-2 text-xs font-medium text-white shadow-panel">
          {error}
        </p>
      )}

      <div className="flex items-center gap-4 rounded-xl border border-slate-200 bg-white px-4 py-3 shadow-panel">
        <span>
          <span className="block text-lg font-bold leading-tight text-slate-900">
            {remainingMin} min
          </span>
          <span className="block text-xs text-slate-500">
            {remainingKm} km
            {!arrived && ` · arrive ${getArrivalTime(remainingMin, isNavigating ? null : departAt)}`}
          </span>
        </span>

        {isNavigating ? (
          <button
            type="button"
            onClick={nav.stop}
            className="flex items-center gap-1.5 rounded-lg bg-red-500 px-4 py-2 text-sm font-semibold text-white hover:bg-red-600"
          >
            <X size={15} aria-hidden />
            End
          </button>
        ) : (
          <button
            type="button"
            onClick={nav.start}
            className="flex items-center gap-1.5 rounded-lg bg-brand px-4 py-2 text-sm font-semibold text-white hover:bg-brand-darker"
          >
            <Navigation size={15} aria-hidden />
            Start
          </button>
        )}
      </div>
    </div>
  );
}
