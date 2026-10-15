import { CHART_GRID } from '@/lib/theme';

export interface DiffRow {
  metric: string;
  antroute: string;
  baseline: string;
  improvementPct: number | null;
  significant: boolean | null;
}

/**
 * The relative difference per metric, ANTRoute against the baseline.
 *
 * One series, so no legend box -- the title names what is plotted. A diverging encoding
 * would be wrong here: this is one measure with a meaningful zero, so bars grow from a zero
 * line and the status colour says which side of it they land on.
 */

const GOOD = '#047857';
const BAD = '#B91C1C';

const CHART_H = 170;
const BAR_W = 34;
const GAP = 44;
const PAD_L = 46;
const PAD_T = 24;
const PAD_B = 46;
const PAD_R = 64;

function niceMax(value: number): number {
  if (!(value > 0)) return 10;
  const exponent = Math.floor(Math.log10(value));
  const magnitude = Math.pow(10, exponent);
  for (const step of [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10]) {
    if (value <= step * magnitude) return step * magnitude;
  }
  return 10 * magnitude;
}

/** Shorter captions than the server's metric names, which overflow a 34px column. */
const SHORT: Record<string, string> = {
  'Route Optimality': 'Optimality',
  'Congestion exposure': 'Congestion',
  'Event exposure': 'Event',
  'Avoidable event exposure': 'Avoidable',
  'Recalculation response time': 'Response time',
};

export default function DifferenceChart({ rows }: { rows: DiffRow[] }) {
  const plotted = rows.filter((r) => r.improvementPct != null);
  if (plotted.length === 0) {
    return (
      <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-6 text-center text-sm text-slate-500">
        No tested differences to plot.
      </p>
    );
  }

  const extent = niceMax(Math.max(...plotted.map((r) => Math.abs(r.improvementPct as number))) * 1.15);
  const width = PAD_L + plotted.length * BAR_W + (plotted.length - 1) * GAP + PAD_R;
  const height = CHART_H + PAD_T + PAD_B;
  const zero = PAD_T + CHART_H / 2;
  const y = (v: number) => zero - (v / extent) * (CHART_H / 2);

  const ticks = [extent, extent / 2, 0, -extent / 2, -extent];

  return (
    <div>
      <div className="overflow-x-auto">
        <svg
          width={width}
          height={height}
          role="img"
          aria-label="Relative difference per metric, ANTRoute against the baseline"
        >
          {ticks.map((t) => (
            <g key={t}>
              <line
                x1={PAD_L}
                x2={width - PAD_R}
                y1={y(t)}
                y2={y(t)}
                stroke={t === 0 ? '#cbd5e1' : CHART_GRID}
                strokeWidth={1}
              />
              <text x={PAD_L - 8} y={y(t) + 4} textAnchor="end" className="fill-slate-400 text-[10px]">
                {t > 0 ? '+' : ''}
                {Math.round(t)}%
              </text>
            </g>
          ))}

          {plotted.map((row, i) => {
            const value = row.improvementPct as number;
            const x = PAD_L + 10 + i * (BAR_W + GAP);
            const top = value >= 0 ? y(value) : zero;
            const h = Math.max(Math.abs(y(value) - zero), 1);
            const color = value >= 0 ? GOOD : BAD;
            // 4px rounded data-end, square where it meets the zero line.
            const d =
              value >= 0
                ? `M${x},${zero} L${x},${top + 4} Q${x},${top} ${x + 4},${top} L${x + BAR_W - 4},${top} Q${x + BAR_W},${top} ${x + BAR_W},${top + 4} L${x + BAR_W},${zero} Z`
                : `M${x},${zero} L${x},${zero + h - 4} Q${x},${zero + h} ${x + 4},${zero + h} L${x + BAR_W - 4},${zero + h} Q${x + BAR_W},${zero + h} ${x + BAR_W},${zero + h - 4} L${x + BAR_W},${zero} Z`;
            return (
              <g key={row.metric}>
                <path d={d} fill={color} />
                <text
                  x={x + BAR_W / 2}
                  y={value >= 0 ? top - 6 : zero + h + 14}
                  textAnchor="middle"
                  className="fill-slate-700 text-[11px] font-semibold"
                >
                  {value > 0 ? '+' : ''}
                  {value}%{row.significant ? ' *' : ''}
                </text>
                <text
                  x={x + BAR_W / 2}
                  y={height - 20}
                  textAnchor="middle"
                  className="fill-slate-600 text-[11px]"
                >
                  {SHORT[row.metric] ?? row.metric}
                </text>
                <title>{`${row.metric}: ${value > 0 ? '+' : ''}${value}% (ANTRoute ${row.antroute} vs baseline ${row.baseline})`}</title>
              </g>
            );
          })}

          <text x={PAD_L} y={height - 4} className="fill-slate-400 text-[10px]">
            above zero = ANTRoute better
          </text>
        </svg>
      </div>
    </div>
  );
}
