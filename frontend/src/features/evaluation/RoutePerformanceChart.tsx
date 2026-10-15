import { useMemo, useState } from 'react';

import type { ScenarioEvaluation } from '@/api/routeService';
import { cn } from '@/lib/cn';
import { CHART_ANT, CHART_BASELINE, CHART_GRID } from '@/lib/theme';

/**
 * ANTRoute against the Improved ACO baseline, by scenario, for one tested metric.
 *
 * Only metrics the thesis actually scores are offered: Route Optimality and the exposures.
 * Travel time is deliberately absent -- it was never scored against a reference, so there is
 * no ANTRoute-vs-baseline result to plot.
 */

/** The tested metrics, by the label the server sends. */
const CHARTABLE = [
  { metric: 'Route Optimality', higherIsBetter: true, unit: '%' },
  { metric: 'Congestion exposure', higherIsBetter: false, unit: '' },
  { metric: 'Event exposure', higherIsBetter: false, unit: '' },
];

/** The server formats values for display ("77.6%", "13.7"); the bars need the number back. */
function toNumber(formatted: string): number | null {
  const n = Number.parseFloat(formatted.replace(/,/g, ''));
  return Number.isFinite(n) ? n : null;
}

interface Bar {
  group: string;
  nTrials: number;
  antroute: number;
  baseline: number;
  improvementPct: number | null;
  significant: boolean | null;
}

/** Round an axis maximum up to a clean number so the ticks read 0 / 25 / 50 / 75 / 100. */
function niceMax(value: number): number {
  if (!(value > 0)) return 1;
  const exponent = Math.floor(Math.log10(value));
  const magnitude = Math.pow(10, exponent);
  const steps = [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10];
  for (const step of steps) {
    if (value <= step * magnitude) return step * magnitude;
  }
  return 10 * magnitude;
}

const CHART_H = 200;
const BAR_W = 24; // mark spec: bars never fill the slot
const BAR_GAP = 2; // the surface gap that separates touching bars
const GROUP_GAP = 56;
const PAD_L = 44;
const PAD_T = 28;
const PAD_B = 34;

export default function RoutePerformanceChart({
  scenarios,
  baselineName,
}: {
  scenarios: ScenarioEvaluation[];
  baselineName: string;
}) {
  const [metric, setMetric] = useState(CHARTABLE[0].metric);
  const [hovered, setHovered] = useState<{ bar: Bar; series: 'ant' | 'base' } | null>(null);

  const spec = CHARTABLE.find((c) => c.metric === metric) ?? CHARTABLE[0];

  // RQ1 asks about single- and multi-destination trips, so those lead; "overall" is the pool.
  const ORDER = ['single_destination', 'multi_destination', 'incident_exposed'];
  const bars = useMemo<Bar[]>(() => {
    return ORDER.map((key) => {
      const sc = scenarios.find((s) => s.scenario === key);
      const row = sc?.metrics.find((m) => m.metric === metric);
      if (!sc || !row) return null;
      const antroute = toNumber(row.antroute);
      const baseline = toNumber(row.baseline);
      if (antroute === null || baseline === null) return null;
      return {
        group: sc.label,
        nTrials: sc.nTrials,
        antroute,
        baseline,
        improvementPct: row.improvementPct,
        significant: row.significant,
      };
    }).filter((b): b is Bar => b !== null);
  }, [scenarios, metric]);

  if (bars.length === 0) {
    return (
      <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-6 text-center text-sm text-slate-500">
        No scenario results for this metric yet.
      </p>
    );
  }

  const max = Math.max(...bars.flatMap((b) => [b.antroute, b.baseline]));
  const scaleMax = niceMax(max * 1.1);
  const groupW = BAR_W * 2 + BAR_GAP;
  // The group caption is centred under the group and is wider than it, so the last one needs
  // room past the final bar or it is clipped at the edge.
  const PAD_R = 56;
  const width = PAD_L + 12 + bars.length * groupW + (bars.length - 1) * GROUP_GAP + PAD_R;
  const height = CHART_H + PAD_T + PAD_B;
  const y = (v: number) => PAD_T + CHART_H - (v / scaleMax) * CHART_H;

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * scaleMax);
  const fmt = (v: number) =>
    `${Number.isInteger(v) ? v : v.toFixed(v < 10 ? 1 : 0)}${spec.unit}`;

  return (
    <div>
      {/* Metric picker, one row above the chart. */}
      <div className="mb-4 flex gap-1 border-b border-slate-200">
        {CHARTABLE.map((c) => (
          <button
            key={c.metric}
            type="button"
            onClick={() => setMetric(c.metric)}
            className={cn(
              '-mb-px border-b-2 px-3 py-2 text-xs font-medium transition-colors',
              metric === c.metric
                ? 'border-brand text-brand'
                : 'border-transparent text-slate-500 hover:text-slate-700'
            )}
          >
            {c.metric}
          </button>
        ))}
      </div>

      {/* Legend: identity never rests on colour alone. */}
      <div className="mb-1 flex items-center gap-4 text-xs text-slate-600">
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{ backgroundColor: CHART_ANT }}
            aria-hidden
          />
          ANTRoute
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{ backgroundColor: CHART_BASELINE }}
            aria-hidden
          />
          {baselineName}
        </span>
        <span className="ml-auto text-slate-400">
          {spec.higherIsBetter ? 'higher is better' : 'lower is better'}
        </span>
      </div>

      <div className="relative overflow-x-auto">
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`${metric}: ANTRoute against ${baselineName} by scenario`}
        >
          {/* Recessive hairline grid. */}
          {ticks.map((t) => (
            <g key={t}>
              <line
                x1={PAD_L}
                x2={width - PAD_R}
                y1={y(t)}
                y2={y(t)}
                stroke={CHART_GRID}
                strokeWidth={1}
              />
              <text
                x={PAD_L - 8}
                y={y(t) + 4}
                textAnchor="end"
                className="fill-slate-400 text-[10px]"
              >
                {fmt(t)}
              </text>
            </g>
          ))}

          {bars.map((bar, i) => {
            const x0 = PAD_L + 12 + i * (groupW + GROUP_GAP);
            const series: Array<{ key: 'ant' | 'base'; v: number; color: string; x: number }> = [
              { key: 'ant', v: bar.antroute, color: CHART_ANT, x: x0 },
              { key: 'base', v: bar.baseline, color: CHART_BASELINE, x: x0 + BAR_W + BAR_GAP },
            ];
            return (
              <g key={bar.group}>
                {series.map((s) => {
                  const top = y(s.v);
                  return (
                    <g
                      key={s.key}
                      onMouseEnter={() => setHovered({ bar, series: s.key })}
                      onMouseLeave={() => setHovered(null)}
                    >
                      {/* Hit target larger than the mark. */}
                      <rect
                        x={s.x - 3}
                        y={PAD_T}
                        width={BAR_W + 6}
                        height={CHART_H}
                        fill="transparent"
                      />
                      {/* 4px rounded data-end, square at the baseline. */}
                      <path
                        d={`M${s.x},${PAD_T + CHART_H} L${s.x},${top + 4} Q${s.x},${top} ${s.x + 4},${top} L${s.x + BAR_W - 4},${top} Q${s.x + BAR_W},${top} ${s.x + BAR_W},${top + 4} L${s.x + BAR_W},${PAD_T + CHART_H} Z`}
                        fill={s.color}
                        opacity={hovered && hovered.bar.group !== bar.group ? 0.45 : 1}
                      />
                      <text
                        x={s.x + BAR_W / 2}
                        y={top - 6}
                        textAnchor="middle"
                        className="fill-slate-600 text-[10px] font-semibold"
                      >
                        {fmt(s.v)}
                      </text>
                      <title>{`${bar.group} · ${s.key === 'ant' ? 'ANTRoute' : baselineName}: ${fmt(s.v)} (n=${bar.nTrials})`}</title>
                    </g>
                  );
                })}
                <text
                  x={x0 + groupW / 2}
                  y={height - 14}
                  textAnchor="middle"
                  className="fill-slate-600 text-[11px]"
                >
                  {bar.group}
                </text>
                <text
                  x={x0 + groupW / 2}
                  y={height - 2}
                  textAnchor="middle"
                  className="fill-slate-400 text-[10px]"
                >
                  n={bar.nTrials}
                  {bar.improvementPct != null
                    ? ` · ${bar.improvementPct > 0 ? '+' : ''}${bar.improvementPct}%${bar.significant ? ' *' : ''}`
                    : ''}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* The table view: every plotted value, never gated behind a hover. */}
      <details className="mt-3">
        <summary className="cursor-pointer text-xs text-slate-500 hover:text-slate-700">
          View as table
        </summary>
        <table className="mt-2 w-full border-collapse text-xs">
          <thead>
            <tr className="bg-slate-50 text-left text-slate-600">
              <th className="px-3 py-2 font-semibold">Scenario</th>
              <th className="px-3 py-2 text-right font-semibold">Trips</th>
              <th className="px-3 py-2 text-right font-semibold">ANTRoute</th>
              <th className="px-3 py-2 text-right font-semibold">{baselineName}</th>
              <th className="px-3 py-2 text-right font-semibold">&Delta;%</th>
            </tr>
          </thead>
          <tbody>
            {bars.map((b) => (
              <tr key={b.group} className="border-t border-slate-200">
                <td className="px-3 py-2 text-slate-700">{b.group}</td>
                <td className="px-3 py-2 text-right tabular-nums text-slate-600">{b.nTrials}</td>
                <td className="px-3 py-2 text-right tabular-nums text-slate-800">
                  {fmt(b.antroute)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-slate-800">
                  {fmt(b.baseline)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-slate-600">
                  {b.improvementPct == null
                    ? '–'
                    : `${b.improvementPct > 0 ? '+' : ''}${b.improvementPct}%${b.significant ? ' *' : ''}`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>

      <p className="mt-2 text-[11px] leading-relaxed text-slate-400">
        Means over the test trips in each scenario. &Delta;% is the relative difference
        (positive = ANTRoute better); * significant at &alpha; = 0.05, Wilcoxon signed-rank
        with Holm correction.
      </p>
    </div>
  );
}
