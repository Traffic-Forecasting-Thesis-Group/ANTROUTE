import { ArrowDown, ArrowUp, Minus } from 'lucide-react';

import type { ComparisonMetricRow, ResponseTimeComparison } from '@/api/routeService';
import { cn } from '@/lib/cn';

/**
 * ANTRoute against the baseline, metric by metric.
 *
 * Route length is carried through as the server sends it -- descriptive, with no Δ and no
 * test -- and recalculation response time is appended from the condition-update harness,
 * which is the other half of research question 3.
 */

export interface Row {
  metric: string;
  antroute: string;
  baseline: string;
  improvementPct: number | null;
  higherIsBetter: boolean;
  pValue: number | null;
  significant: boolean | null;
  /** True for a metric the thesis reports but does not test. */
  descriptive?: boolean;
  note?: string | null;
}

/** The response-time result as a table row, reported in whichever direction it actually went. */
export function responseTimeRow(rt: ResponseTimeComparison | null): Row | null {
  if (!rt || rt.antroute_median_s == null || rt.baseline_median_s == null) return null;
  const antFaster = rt.faster === 'antroute';
  // The server computes this with the same relative_difference the thesis uses; recomputing
  // it here with a different denominator would disagree with the published tables.
  const delta =
    rt.relative_difference_pct == null ? null : Math.round(rt.relative_difference_pct * 10) / 10;
  return {
    metric: 'Recalculation response time',
    antroute: `${rt.antroute_median_s.toFixed(2)} s`,
    baseline: `${rt.baseline_median_s.toFixed(2)} s`,
    improvementPct: delta,
    higherIsBetter: false,
    pValue: rt.p_value,
    significant: rt.significant,
    note: antFaster
      ? null
      : 'The baseline is faster here: it searches on distance and flow alone, while ANTRoute also applies the risk tiers and the Event-Aware layer before searching.',
  };
}

export function toRows(metrics: ComparisonMetricRow[]): Row[] {
  return metrics.map((m) => ({
    metric: m.metric,
    antroute: m.antroute,
    baseline: m.baseline,
    improvementPct: m.improvementPct,
    higherIsBetter: m.higherIsBetter,
    pValue: m.pValue,
    significant: m.significant,
    descriptive: m.improvementPct == null && m.pValue == null,
  }));
}

function Delta({ row }: { row: Row }) {
  if (row.improvementPct == null) {
    return <span className="text-slate-400">&ndash;</span>;
  }
  const even = Math.abs(row.improvementPct) < 0.05;
  const better = row.improvementPct > 0;
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-semibold',
        even
          ? 'bg-slate-100 text-slate-600'
          : better
            ? 'bg-emerald-100 text-emerald-700'
            : 'bg-red-100 text-red-700'
      )}
    >
      {even ? (
        <Minus size={11} aria-hidden />
      ) : better ? (
        <ArrowUp size={11} aria-hidden />
      ) : (
        <ArrowDown size={11} aria-hidden />
      )}
      {!even && better ? '+' : ''}
      {row.improvementPct}%
    </span>
  );
}

export default function MetricComparisonTable({
  rows,
  baselineName,
  showPValue = false,
}: {
  rows: Row[];
  baselineName: string;
  showPValue?: boolean;
}) {
  const notes = rows.filter((r) => r.note);

  return (
    <>
      <div className="overflow-hidden rounded-lg border border-slate-200">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="bg-slate-50 text-left text-slate-600">
              <th className="px-4 py-2.5 font-semibold">Metric</th>
              <th className="px-4 py-2.5 text-right font-semibold">ANTRoute</th>
              <th className="px-4 py-2.5 text-right font-semibold">{baselineName}</th>
              <th className="px-4 py-2.5 text-right font-semibold">&Delta;</th>
              {showPValue && <th className="px-4 py-2.5 text-right font-semibold">p (Holm)</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.metric} className="border-t border-slate-200">
                <td className="px-4 py-2.5">
                  <span className="flex items-center gap-1.5 text-slate-700">
                    {row.descriptive ? (
                      <Minus size={12} className="text-slate-300" aria-hidden />
                    ) : row.higherIsBetter ? (
                      <ArrowUp size={12} className="text-slate-400" aria-hidden />
                    ) : (
                      <ArrowDown size={12} className="text-slate-400" aria-hidden />
                    )}
                    {row.metric}
                    {row.significant ? <span className="text-slate-400">*</span> : null}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-right tabular-nums text-slate-800">
                  {row.antroute}
                </td>
                <td className="px-4 py-2.5 text-right tabular-nums text-slate-800">
                  {row.baseline}
                </td>
                <td className="px-4 py-2.5 text-right">
                  <Delta row={row} />
                </td>
                {showPValue && (
                  <td className="px-4 py-2.5 text-right tabular-nums">
                    {row.pValue == null ? (
                      <span className="text-slate-400">&ndash;</span>
                    ) : (
                      <span
                        className={cn(
                          'rounded-full px-2 py-0.5 text-xs font-medium',
                          row.significant
                            ? 'bg-emerald-50 text-emerald-700'
                            : 'bg-slate-100 text-slate-500'
                        )}
                      >
                        {row.pValue < 0.001 ? '<0.001' : row.pValue.toFixed(3)}
                      </span>
                    )}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {notes.map((row) => (
        <p key={row.metric} className="mt-2 text-[11px] leading-relaxed text-slate-500">
          <span className="font-semibold">{row.metric}:</span> {row.note}
        </p>
      ))}

      <p className="mt-2 text-[11px] leading-relaxed text-slate-400">
        An arrow by a metric says which direction is better. &Delta; is the relative
        difference (Equations 7&ndash;8): positive means ANTRoute is better. A metric with no
        &Delta; is descriptive and was not tested. * significant at &alpha; = 0.05, Wilcoxon
        signed-rank with Holm correction. Travel time is not here: it is not a thesis metric
        and was never scored.
      </p>
    </>
  );
}
