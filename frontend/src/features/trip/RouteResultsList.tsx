import { useState } from 'react';
import { Clock, Info } from 'lucide-react';

import { DURATION_NOTE, type RouteOption } from '@/api/routeService';
import { cn } from '@/lib/cn';

import RouteInformationModal from './RouteInformationModal';
import { useTripPlanner } from './TripPlannerContext';

const CONGESTION_TEXT = {
  clear: 'text-emerald-600',
  moderate: 'text-amber-600',
  heavy: 'text-red-600',
  unknown: 'text-slate-400',
} as const;

function Figure({
  value,
  label,
  className,
}: {
  value: string;
  label: string;
  className?: string;
}) {
  return (
    <div className="flex-1 px-1 text-center">
      <p className={cn('text-sm font-bold', className ?? 'text-slate-900')}>{value}</p>
      <p className="mt-0.5 text-[11px] text-slate-500">{label}</p>
    </div>
  );
}

function RouteCard({
  route,
  index,
  isSelected,
  onSelect,
  onViewDetails,
}: {
  route: RouteOption;
  index: number;
  isSelected: boolean;
  onSelect: () => void;
  onViewDetails: () => void;
}) {
  return (
    <div
      onClick={onSelect}
      className={cn(
        'flex cursor-pointer flex-col rounded-xl border p-4 transition-colors',
        isSelected
          ? 'border-brand bg-brand-light/40 shadow-card'
          : 'border-slate-200 bg-white hover:border-brand/50'
      )}
    >
      {index === 0 ? (
        <span className="mb-2 self-start rounded-full bg-brand-light px-2.5 py-1 text-[11px] font-semibold text-brand">
          Recommended Route
        </span>
      ) : (
        <span className="mb-2 h-[26px]" aria-hidden />
      )}

      <h4 className="mb-3 truncate font-semibold text-slate-900" title={route.via}>
        Via {route.via}
      </h4>

      <div className="mb-1 flex divide-x divide-slate-200">
        <Figure value={`${route.duration_min} min`} label="Travel Time" />
        <Figure value={`${route.distance_km} km`} label="Distance" />
        <Figure
          value={route.congestion_level}
          label="Congestion"
          className={cn('capitalize', CONGESTION_TEXT[route.congestion_level])}
        />
      </div>

      {/* The SOP does not score travel time, so it never appears without this caption. */}
      <p className="mb-3 text-center text-[10px] italic text-slate-400">
        {route.duration_note ?? DURATION_NOTE}
      </p>

      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onViewDetails();
        }}
        className={cn(
          'mt-auto w-full rounded-lg border px-4 py-2 text-sm font-semibold transition-colors',
          isSelected
            ? 'border-brand bg-brand text-white hover:bg-brand-darker'
            : 'border-brand bg-white text-brand hover:bg-brand-light'
        )}
      >
        View Details
      </button>
    </div>
  );
}

/**
 * The route options, below the map: the recommended one first, then the alternatives. The
 * card carries only what fits at a glance; everything else -- the leg breakdown and the
 * recorded inputs along the route -- is in the Route Information modal.
 */
export default function RouteResultsList() {
  const {
    antRouteOptions,
    baselineModelOptions,
    baselineNotice,
    trafficNote,
    selectedModelTab,
    setSelectedModelTab,
    selectedIndex,
    setSelectedIndex,
    hasPlannedTrip,
  } = useTripPlanner();

  const [detailsFor, setDetailsFor] = useState<RouteOption | null>(null);

  if (!hasPlannedTrip) return null;

  const options = selectedModelTab === 'baseline' ? baselineModelOptions : antRouteOptions;

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-card">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="font-semibold text-slate-900">
            {selectedModelTab === 'baseline'
              ? 'Route by Baseline Model (Improved ACO)'
              : 'Alternative Route by ANTROUTE'}
          </h3>
          {trafficNote && (
            <p className="mt-1 flex items-center gap-1.5 text-xs text-slate-400">
              <Clock size={12} aria-hidden />
              {trafficNote}
            </p>
          )}
        </div>

        <div className="inline-flex rounded-lg border border-slate-300 bg-white p-0.5">
          <button
            type="button"
            onClick={() => setSelectedModelTab('antroute')}
            className={cn(
              'rounded-md px-4 py-1.5 text-xs font-semibold transition-colors',
              selectedModelTab === 'antroute'
                ? 'bg-brand text-white'
                : 'text-slate-600 hover:bg-slate-50'
            )}
          >
            ANTRoute
          </button>
          <button
            type="button"
            onClick={() => setSelectedModelTab('baseline')}
            className={cn(
              'rounded-md px-4 py-1.5 text-xs font-semibold transition-colors',
              selectedModelTab === 'baseline'
                ? 'bg-brand text-white'
                : 'text-slate-600 hover:bg-slate-50'
            )}
          >
            Baseline
          </button>
        </div>
      </div>

      {options.length > 0 ? (
        <div className="grid grid-cols-3 gap-4">
          {options.map((route, index) => (
            <RouteCard
              key={index}
              route={route}
              index={index}
              isSelected={index === selectedIndex}
              onSelect={() => setSelectedIndex(index)}
              onViewDetails={() => {
                setSelectedIndex(index);
                setDetailsFor(route);
              }}
            />
          ))}
        </div>
      ) : (
        <p className="flex items-start gap-2 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
          <Info size={16} className="mt-0.5 shrink-0 text-slate-400" aria-hidden />
          {baselineNotice ?? 'The baseline has no route for this trip.'}
        </p>
      )}

      <RouteInformationModal route={detailsFor} onClose={() => setDetailsFor(null)} />
    </section>
  );
}
