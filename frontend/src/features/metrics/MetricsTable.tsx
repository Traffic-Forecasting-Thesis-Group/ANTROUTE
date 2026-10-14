import type { ComparisonMetricRow } from '@/api/routeService';
import { cn } from '@/lib/cn';

/**
 * The ANTRoute-vs-baseline metrics table. In the React Native app this was written twice --
 * once inside HomeScreen and again, verbatim, inside EvaluationScreen. It is one component
 * now, used by the Dashboard's trip panel, the Evaluation page and the Comparison page.
 */
export default function MetricsTable({
  rows,
  baselineName = 'Baseline',
}: {
  rows: ComparisonMetricRow[];
  baselineName?: string;
}) {
  return (
    <div className="overflow-hidden rounded-lg border border-slate-200">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="bg-slate-50 text-left">
            <th className="px-4 py-2.5 font-semibold text-slate-600">Metric</th>
            <th className="px-4 py-2.5 text-right font-semibold text-slate-600">ANTRoute</th>
            <th className="px-4 py-2.5 text-right font-semibold text-slate-600">{baselineName}</th>
            <th className="px-4 py-2.5 text-right font-semibold text-slate-600">Improvement</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const isGood = row.improvementPct != null && row.improvementPct > 0;
            return (
              <tr key={row.metric} className="border-t border-slate-200">
                <td className="px-4 py-2.5 text-slate-700">
                  {row.metric}
                  {row.significant ? ' *' : ''}
                </td>
                <td className="px-4 py-2.5 text-right tabular-nums text-slate-800">
                  {row.antroute}
                </td>
                <td className="px-4 py-2.5 text-right tabular-nums text-slate-800">
                  {row.baseline}
                </td>
                <td
                  className={cn(
                    'px-4 py-2.5 text-right font-semibold tabular-nums',
                    isGood ? 'text-emerald-600' : 'text-slate-500'
                  )}
                >
                  {row.improvementPct == null
                    ? '–'
                    : `${row.improvementPct > 0 ? '+' : ''}${row.improvementPct}%`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** The reading guide that sits under the table wherever it appears. */
export function MetricsFootnote() {
  return (
    <p className="mt-3 text-xs leading-relaxed text-slate-500">
      Higher is better for Route Optimality; lower is better for the exposures. Route
      Optimality times both routes in Google Maps. Route length is descriptive only and is
      not tested. Improvement is the relative difference &Delta;% (Equations 7&ndash;8):
      positive means ANTRoute is better. * significant at &alpha; = 0.05, Wilcoxon
      signed-rank with Holm correction within each test family. Travel time is not a thesis
      metric and is never scored here.
    </p>
  );
}
