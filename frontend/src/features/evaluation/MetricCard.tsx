import type { ReactNode } from 'react';
import { ArrowDown, ArrowUp, Minus } from 'lucide-react';

import { cn } from '@/lib/cn';

export type Verdict = 'better' | 'worse' | 'even' | null;

/**
 * One tile in the Route Performance grid.
 *
 * A tile only carries a better/worse badge when the thesis actually scores that metric.
 * Travel time and route length are shown without one and say so in their note: travel time
 * was never scored against a reference, and route length is descriptive. A badge there
 * would read as a validated result.
 */
export default function MetricCard({
  icon,
  label,
  value,
  note,
  verdict = null,
  verdictLabel,
  arrow,
  muted = false,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  note?: string | null;
  verdict?: Verdict;
  verdictLabel?: string;
  /**
   * Which way the number moved. Kept separate from the verdict because they disagree on a
   * lower-is-better metric: congestion exposure falling is an arrow down AND a green badge.
   * Defaults to following the verdict.
   */
  arrow?: 'up' | 'down' | 'none';
  muted?: boolean;
}) {
  const direction = arrow ?? (verdict === 'better' ? 'up' : verdict === 'worse' ? 'down' : 'none');
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-card">
      <p className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-slate-500">
        <span className="text-slate-400">{icon}</span>
        {label}
      </p>

      <p
        className={cn(
          'text-2xl font-bold leading-tight',
          muted ? 'text-slate-500' : 'text-slate-900'
        )}
      >
        {value}
      </p>

      {note && <p className="mt-0.5 text-[11px] leading-snug text-slate-400">{note}</p>}

      {verdict && (
        <span
          className={cn(
            'mt-2 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold',
            verdict === 'better' && 'bg-emerald-100 text-emerald-700',
            verdict === 'worse' && 'bg-red-100 text-red-700',
            verdict === 'even' && 'bg-slate-100 text-slate-600'
          )}
        >
          {direction === 'up' && <ArrowUp size={11} aria-hidden />}
          {direction === 'down' && <ArrowDown size={11} aria-hidden />}
          {direction === 'none' && <Minus size={11} aria-hidden />}
          {verdictLabel ?? (verdict === 'better' ? 'Better' : verdict === 'worse' ? 'Worse' : 'Even')}
        </span>
      )}
    </div>
  );
}
