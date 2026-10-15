import { CalendarDays, Clock } from 'lucide-react';

import { SIMULATION_DATE_LABEL } from '@/lib/constants';
import { toTimeInputValue, withTimeInput } from '@/lib/time';

import { useTripPlanner } from './TripPlannerContext';

/**
 * Departure time for the simulation.
 *
 * There is no "leave now": every route is planned on recorded traffic, so the date is fixed
 * to the simulation day (the one day held out of the model's training and validation) and
 * only the time of day is chosen. Any time is allowed -- outside the recorded AM and PM
 * sessions the server still plans the route and says which window it fell back to, which
 * arrives as the traffic note under the results.
 */
export default function DeparturePicker() {
  const { departAt, changeDeparture } = useTripPlanner();

  return (
    <section>
      <h2 className="mb-2 text-sm font-semibold text-slate-700">Departure Time</h2>

      <div className="flex gap-2">
        {/* The simulation day is the only recorded day held out of training, so it is shown,
            not chosen. */}
        <div
          className="flex flex-1 items-center gap-2 rounded-lg border border-slate-300 bg-slate-50 px-3 py-2"
          title="The simulation day. Routes are planned on recorded traffic, so the date is fixed."
        >
          <CalendarDays size={15} className="shrink-0 text-brand" aria-hidden />
          <span className="min-w-0 flex-1 truncate text-xs text-slate-700">
            {SIMULATION_DATE_LABEL}
          </span>
        </div>

        <label className="flex flex-1 items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 focus-within:border-brand focus-within:ring-2 focus-within:ring-brand/20">
          <Clock size={15} className="shrink-0 text-brand" aria-hidden />
          <input
            type="time"
            value={toTimeInputValue(departAt)}
            onChange={(e) => changeDeparture(withTimeInput(departAt, e.target.value))}
            aria-label="Departure time"
            className="min-w-0 flex-1 bg-transparent text-xs text-slate-800 focus:outline-none"
          />
        </label>
      </div>
    </section>
  );
}
