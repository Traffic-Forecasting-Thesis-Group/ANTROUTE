import { CarFront, LocateFixed, MapPin } from 'lucide-react';
import { Link } from 'react-router-dom';

import { useTripPlanner } from '@/features/trip/TripPlannerContext';
import { SIMULATION_DATE_LABEL } from '@/lib/constants';
import { formatClock } from '@/lib/time';

function Field({
  icon,
  value,
  placeholder,
}: {
  icon: React.ReactNode;
  value: string;
  placeholder: string;
}) {
  const empty = value.trim().length === 0;
  return (
    <div className="flex items-center gap-2 rounded-lg border border-slate-300 bg-slate-50 px-3 py-2">
      <span className="shrink-0 text-brand">{icon}</span>
      <span
        className={`min-w-0 flex-1 truncate text-xs ${empty ? 'text-slate-400' : 'text-slate-700'}`}
        title={empty ? undefined : value}
      >
        {empty ? placeholder : value}
      </span>
    </div>
  );
}

/**
 * The trip being evaluated, read-only. It is whatever was planned on Home -- the evaluation
 * describes that trip, so editing it here would let the two pages disagree.
 */
export default function TripSummaryRail() {
  const { origin, destinations, departAt, hasPlannedTrip } = useTripPlanner();

  return (
    <aside className="flex w-[260px] shrink-0 flex-col gap-5 border-r border-slate-200 bg-white px-6 py-6">
      <section>
        <h2 className="mb-2 text-sm font-semibold text-slate-700">Trip Details</h2>
        <div className="flex flex-col gap-2">
          <Field icon={<LocateFixed size={15} />} value={origin} placeholder="Start trip" />
          {destinations.map((d, i) => (
            <Field
              key={i}
              icon={<MapPin size={15} />}
              value={d.name}
              placeholder={destinations.length > 1 ? `Destination ${i + 1}` : 'Destination'}
            />
          ))}
        </div>

        <p className="mt-2 text-[11px] text-slate-400">
          {SIMULATION_DATE_LABEL}, {formatClock(departAt)}
        </p>

        {!hasPlannedTrip && (
          <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
            No trip planned yet.{' '}
            <Link to="/" className="font-medium text-brand hover:underline">
              Plan one on Home
            </Link>{' '}
            and its evaluation appears here.
          </p>
        )}
      </section>

      <p className="mt-auto flex items-start gap-1.5 text-[11px] text-slate-400">
        <CarFront size={13} className="mt-0.5 shrink-0" aria-hidden />
        Routes optimized for private 4-wheel vehicles.
      </p>
    </aside>
  );
}
