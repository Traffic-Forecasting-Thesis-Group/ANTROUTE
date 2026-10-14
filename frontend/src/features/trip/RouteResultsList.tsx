import { Camera, Clock, Info } from 'lucide-react';

import { DURATION_NOTE, type RouteOption } from '@/api/routeService';
import { cn } from '@/lib/cn';
import { CONGESTION_COLORS, CONGESTION_WIDTH } from '@/lib/theme';
import { getArrivalTime } from '@/lib/time';

import { weatherLine } from './RecordedInputsPanel';
import { useTripPlanner } from './TripPlannerContext';

function InfoNote({ children }: { children: React.ReactNode }) {
  return (
    <p className="mt-2 flex items-start gap-1.5 rounded-md bg-blue-50 px-2 py-1.5 text-xs text-slate-600">
      <Info size={13} className="mt-0.5 shrink-0 text-blue-500" aria-hidden />
      <span>{children}</span>
    </p>
  );
}

const TIER_LABEL = {
  observed: 'Observed',
  propagated: 'Propagated',
  assumed: 'Assumed',
} as const;
const TIER_COLOR = { observed: '#1d4ed8', propagated: '#60a5fa', assumed: '#cbd5e1' } as const;

/**
 * What this route is made of, so alternatives can be compared: where its risk comes from, the
 * cameras and incidents on it, the weather and flood along it. Recorded, not live.
 */
function RouteFacts({ route }: { route: RouteOption }) {
  const inputs = route.inputs;
  const sources = route.risk_sources;
  if (!inputs && !sources) return null;

  return (
    <div className="mt-3 border-t border-slate-200 pt-2.5 text-[11px] text-slate-600">
      {sources && (
        <div className="mb-2">
          <div className="flex h-1.5 overflow-hidden rounded-full bg-slate-200" aria-hidden>
            {(Object.keys(TIER_LABEL) as (keyof typeof TIER_LABEL)[]).map((tier) => (
              <span
                key={tier}
                style={{ width: `${Math.round((sources[tier] ?? 0) * 100)}%`, backgroundColor: TIER_COLOR[tier] }}
              />
            ))}
          </div>
          <p className="mt-1 text-slate-500">
            Risk from{' '}
            {(Object.keys(TIER_LABEL) as (keyof typeof TIER_LABEL)[])
              .map((tier) => `${TIER_LABEL[tier].toLowerCase()} ${Math.round((sources[tier] ?? 0) * 100)}%`)
              .join(' · ')}
          </p>
        </div>
      )}

      {inputs && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
          <dt className="flex items-center gap-1 text-slate-400">
            <Camera size={11} aria-hidden /> Cameras
          </dt>
          <dd>
            {inputs.cameras.length === 0
              ? 'none within 200 m'
              : inputs.cameras
                  .map(
                    (c) =>
                      `${c.camera}${c.human_label ? ` (${c.human_label.toLowerCase()}` : ' ('}${
                        c.vehicle_count != null ? `${c.human_label ? ', ' : ''}${c.vehicle_count} veh` : ''
                      })`
                  )
                  .join('; ')}
          </dd>

          <dt className="text-slate-400">Incidents</dt>
          <dd>
            {inputs.incidents.length === 0
              ? 'none on the route'
              : inputs.incidents
                  .map((i) => `${i.location} (${i.lanes} lane${i.lanes === 1 ? '' : 's'}, ${i.minutes_ago ?? '?'} min ago)`)
                  .join('; ')}
          </dd>

          <dt className="text-slate-400">Weather</dt>
          <dd>{weatherLine(inputs.weather)}</dd>

          <dt className="text-slate-400">Flood</dt>
          <dd>
            {inputs.flood.share_of_route_flood_prone
              ? `${Math.round(inputs.flood.share_of_route_flood_prone * 100)}% of the route flood-prone (up to level ${inputs.flood.max_level ?? 0})`
              : 'no flood-prone roads'}
          </dd>

          {inputs.vehicle_count_mean != null && (
            <>
              <dt className="text-slate-400">Vehicles</dt>
              <dd>{inputs.vehicle_count_mean} per frame at its cameras</dd>
            </>
          )}
        </dl>
      )}
      {inputs && <p className="mt-1.5 text-[10px] italic text-slate-400">Recorded data, not a live feed.</p>}
    </div>
  );
}

function RouteCard({
  route,
  index,
  isSelected,
  departAt,
  onSelect,
}: {
  route: RouteOption;
  index: number;
  isSelected: boolean;
  departAt: Date | null;
  onSelect: () => void;
}) {
  const level = route.congestion_level;

  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        'w-full rounded-xl border p-4 text-left transition-colors',
        isSelected
          ? 'border-brand bg-brand-light shadow-card'
          : 'border-slate-200 bg-white hover:border-brand/50'
      )}
    >
      <div className="flex items-baseline justify-between gap-3">
        <span className="flex items-baseline gap-2">
          <span className="text-xl font-bold text-slate-900">{route.duration_min} min</span>
          <span className="text-sm text-slate-500">{route.distance_km} km</span>
        </span>
        {index === 0 && (
          <span className="rounded-full bg-brand px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-white">
            Recommended
          </span>
        )}
      </div>

      <p className="mt-0.5 text-[11px] italic text-slate-400">
        {route.duration_note ?? DURATION_NOTE}
      </p>

      <p className="mt-1.5 truncate text-sm text-slate-600">Via {route.via}</p>
      <p className="mt-0.5 text-xs text-slate-400">
        Arrive about {getArrivalTime(route.duration_min, departAt)}
      </p>

      <div className="mt-3 flex items-center gap-2">
        <span className="text-xs text-slate-500">Congestion</span>
        <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-200">
          <span
            className="block h-full rounded-full"
            style={{
              width: CONGESTION_WIDTH[level],
              backgroundColor: CONGESTION_COLORS[level],
            }}
          />
        </span>
        <span className="text-xs font-medium capitalize text-slate-600">{level}</span>
      </div>

      {route.congestion_note && (
        <p className="mt-1 text-[11px] italic text-slate-400">{route.congestion_note}</p>
      )}

      {route.event_note && <InfoNote>{route.event_note}</InfoNote>}
      {route.fallback_reason && <InfoNote>{route.fallback_reason}</InfoNote>}
      <RouteFacts route={route} />
    </button>
  );
}

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
    departAt,
    simulationWindow,
    hasPlannedTrip,
  } = useTripPlanner();

  if (!hasPlannedTrip) return null;

  const options = selectedModelTab === 'baseline' ? baselineModelOptions : antRouteOptions;

  return (
    <section className="mt-6">
      <div className="mb-3 inline-flex rounded-lg border border-slate-300 bg-white p-0.5">
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

      <h3 className="mb-1 text-sm font-semibold text-brand">
        {selectedModelTab === 'baseline'
          ? 'Route by Baseline Model (Improved ACO)'
          : 'Alternative Routes by ANTRoute Model'}
      </h3>

      {trafficNote && (
        <p className="mb-3 flex items-center gap-1.5 text-xs text-slate-400">
          <Clock size={12} aria-hidden />
          {trafficNote}
        </p>
      )}

      {options.length > 0 ? (
        <div className="flex flex-col gap-2">
          {options.map((route, index) => (
            <RouteCard
              key={index}
              route={route}
              index={index}
              isSelected={index === selectedIndex}
              departAt={simulationWindow ? new Date(simulationWindow) : departAt}
              onSelect={() => setSelectedIndex(index)}
            />
          ))}
        </div>
      ) : (
        <p className="flex items-start gap-2 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
          <Info size={16} className="mt-0.5 shrink-0 text-slate-400" aria-hidden />
          {baselineNotice ?? 'The baseline has no route for this trip.'}
        </p>
      )}
    </section>
  );
}
