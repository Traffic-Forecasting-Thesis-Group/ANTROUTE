import { useEffect, useMemo, useState } from 'react';
import { FlaskConical } from 'lucide-react';

import { getRecordedSessions, type RecordedSession } from '@/api/routeService';
import { cn } from '@/lib/cn';

import { useTripPlanner } from './TripPlannerContext';

const SPLIT_LABEL: Record<string, string> = {
  train: 'training',
  val: 'validation',
  test: 'held out',
  infer: 'unlabelled',
};

/** "5:20 PM" from an ISO window start, read as the Manila wall-clock time it is written in. */
function clock(iso: string): string {
  const [, time] = iso.split('T');
  const [h, m] = time.split(':').map(Number);
  return `${h % 12 || 12}:${String(m).padStart(2, '0')} ${h < 12 ? 'AM' : 'PM'}`;
}

/**
 * Simulation mode: pick a recorded day and session and plan routes on what was recorded then --
 * that day's incidents, weather, flood and camera readings -- instead of the window matching
 * the departure time. There is no live feed; the held-out day (May 25, split "test") is the one
 * the model never saw in training or validation.
 */
export default function SimulationPicker() {
  const { simulationWindow, setSimulationWindow } = useTripPlanner();
  const [sessions, setSessions] = useState<RecordedSession[]>([]);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    getRecordedSessions()
      .then((s) => !cancelled && setSessions(s))
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, []);

  const on = simulationWindow !== null;
  const current = useMemo(
    () => sessions.find((s) => s.windows.some((w) => w.start === simulationWindow)),
    [sessions, simulationWindow]
  );

  const turnOn = () => {
    // Start on the held-out day: it is the one the thesis evaluates on.
    const first = sessions.find((s) => s.held_out) ?? sessions[sessions.length - 1];
    if (first) setSimulationWindow(first.windows[0].start);
  };

  if (error && !on) {
    return (
      <section>
        <h2 className="mb-1 text-sm font-semibold text-slate-700">Simulation</h2>
        <p className="text-xs text-slate-400">Recorded sessions are not available: {error}</p>
      </section>
    );
  }

  return (
    <section>
      <div className="mb-2 flex items-center justify-between">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold text-slate-700">
          <FlaskConical size={14} aria-hidden /> Simulation
        </h2>
        <button
          type="button"
          role="switch"
          aria-checked={on}
          aria-label="Simulation mode"
          disabled={sessions.length === 0}
          onClick={() => (on ? setSimulationWindow(null) : turnOn())}
          className={cn(
            'relative h-5 w-9 rounded-full transition-colors disabled:opacity-50',
            on ? 'bg-brand' : 'bg-slate-300'
          )}
        >
          <span
            className={cn(
              'absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all',
              on ? 'left-[18px]' : 'left-0.5'
            )}
          />
        </button>
      </div>

      {!on ? (
        <p className="text-xs text-slate-400">
          Off: routes use the recorded window that matches the departure time. Turn on to plan
          on a chosen recorded day and session.
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          <select
            aria-label="Recorded day and session"
            value={current ? `${current.day}|${current.session}` : ''}
            onChange={(e) => {
              const session = sessions.find((s) => `${s.day}|${s.session}` === e.target.value);
              if (session) setSimulationWindow(session.windows[0].start);
            }}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus:border-brand focus:outline-none"
          >
            {sessions.map((s) => (
              <option key={`${s.day}|${s.session}`} value={`${s.day}|${s.session}`}>
                {s.label} — {SPLIT_LABEL[s.split] ?? s.split}
              </option>
            ))}
          </select>

          {current && (
            <select
              aria-label="Recorded time window"
              value={simulationWindow ?? ''}
              onChange={(e) => setSimulationWindow(e.target.value)}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 focus:border-brand focus:outline-none"
            >
              {current.windows.map((w) => (
                <option key={w.start} value={w.start}>
                  {clock(w.start)}
                  {w.end ? ` – ${clock(w.end)}` : ''}
                </option>
              ))}
            </select>
          )}

          {current && (
            <p
              className={cn(
                'rounded-md px-2 py-1.5 text-[11px]',
                current.held_out ? 'bg-amber-50 text-amber-800' : 'bg-slate-50 text-slate-500'
              )}
            >
              {current.note.charAt(0).toUpperCase() + current.note.slice(1)}.
            </p>
          )}
        </div>
      )}
    </section>
  );
}
