import { CalendarDays, Clock } from 'lucide-react';

import { cn } from '@/lib/cn';
import {
  currentMinute,
  daysFromToday,
  toDateInputValue,
  toTimeInputValue,
  withDateInput,
  withTimeInput,
} from '@/lib/time';

import { useTripPlanner } from './TripPlannerContext';

/**
 * Departure time. The native build used @react-native-community/datetimepicker, which has
 * no web implementation; the browser's own date and time inputs replace it.
 *
 * A planned departure is clamped to between now and the last minute of tomorrow -- see
 * clampDeparture, which changeDeparture applies.
 */
export default function DeparturePicker() {
  const { departAt, changeDeparture, setDepartureDay } = useTripPlanner();

  const leaveNow = departAt === null;
  const value = departAt ?? currentMinute();
  const dayOffset = daysFromToday(value);

  return (
    <section>
      <h2 className="mb-2 text-sm font-semibold text-slate-700">Departure Time</h2>

      <div className="mb-2 flex gap-2">
        <button
          type="button"
          onClick={() => changeDeparture(null)}
          className={cn(
            'flex-1 rounded-lg border px-3 py-1.5 text-xs font-semibold transition-colors',
            leaveNow
              ? 'border-brand bg-brand text-white'
              : 'border-slate-300 bg-white text-slate-600 hover:bg-slate-50'
          )}
        >
          Leave now
        </button>
        <button
          type="button"
          onClick={() => changeDeparture(currentMinute())}
          className={cn(
            'flex-1 rounded-lg border px-3 py-1.5 text-xs font-semibold transition-colors',
            !leaveNow
              ? 'border-brand bg-brand text-white'
              : 'border-slate-300 bg-white text-slate-600 hover:bg-slate-50'
          )}
        >
          Depart at
        </button>
      </div>

      <div className="flex gap-2">
        <label
          className={cn(
            'flex flex-1 items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2',
            leaveNow && 'opacity-50'
          )}
        >
          <CalendarDays size={15} className="shrink-0 text-brand" aria-hidden />
          <input
            type="date"
            disabled={leaveNow}
            value={toDateInputValue(value)}
            min={toDateInputValue(new Date())}
            onChange={(e) => changeDeparture(withDateInput(value, e.target.value))}
            aria-label="Departure date"
            className="min-w-0 flex-1 bg-transparent text-xs text-slate-800 focus:outline-none disabled:cursor-not-allowed"
          />
        </label>

        <label
          className={cn(
            'flex flex-1 items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2',
            leaveNow && 'opacity-50'
          )}
        >
          <Clock size={15} className="shrink-0 text-brand" aria-hidden />
          <input
            type="time"
            disabled={leaveNow}
            value={toTimeInputValue(value)}
            onChange={(e) => changeDeparture(withTimeInput(value, e.target.value))}
            aria-label="Departure time"
            className="min-w-0 flex-1 bg-transparent text-xs text-slate-800 focus:outline-none disabled:cursor-not-allowed"
          />
        </label>
      </div>

      {!leaveNow && (
        <div className="mt-2 flex gap-2">
          <button
            type="button"
            onClick={() => setDepartureDay(0)}
            className={cn(
              'rounded-full border px-3 py-1 text-[11px] font-medium transition-colors',
              dayOffset === 0
                ? 'border-brand bg-brand-light text-brand'
                : 'border-slate-300 bg-white text-slate-600 hover:bg-slate-50'
            )}
          >
            Today
          </button>
          <button
            type="button"
            onClick={() => setDepartureDay(1)}
            className={cn(
              'rounded-full border px-3 py-1 text-[11px] font-medium transition-colors',
              dayOffset === 1
                ? 'border-brand bg-brand-light text-brand'
                : 'border-slate-300 bg-white text-slate-600 hover:bg-slate-50'
            )}
          >
            Tomorrow
          </button>
        </div>
      )}
    </section>
  );
}
