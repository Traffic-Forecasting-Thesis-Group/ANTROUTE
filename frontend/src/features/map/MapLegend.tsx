import { INCIDENT_LEGEND, ROUTE_LEGEND, TRAFFIC_LEGEND } from './legend';

function RouteSwatch({ color, dashed }: { color: string; dashed: boolean }) {
  return (
    <span
      aria-hidden
      className="h-0.5 w-6 shrink-0 rounded-full"
      style={
        dashed
          ? { backgroundImage: `repeating-linear-gradient(to right, ${color} 0 4px, transparent 4px 7px)` }
          : { backgroundColor: color }
      }
    />
  );
}

/**
 * The legend panel from the mockup. Every row comes from legend.ts, which the marker layer
 * reads too -- the incident rows are listed even while the data source is empty, so the
 * demonstration shows what the map can display once a feed is wired in.
 */
export default function MapLegend() {
  return (
    <div className="pointer-events-auto max-h-[calc(100%-4rem)] overflow-y-auto rounded-lg border border-slate-200 bg-white/95 px-3 py-2.5 shadow-panel backdrop-blur">
      <ul className="flex flex-col gap-1.5 text-[11px] text-slate-700">
        {ROUTE_LEGEND.map((entry) => (
          <li key={entry.label} className="flex items-center gap-2">
            <RouteSwatch color={entry.color} dashed={entry.dashed} />
            {entry.label}
          </li>
        ))}

        <li className="my-0.5 border-t border-slate-200" aria-hidden />

        {Object.entries(TRAFFIC_LEGEND).map(([key, entry]) => (
          <li key={key} className="flex items-center gap-2">
            <span
              aria-hidden
              className="h-0.5 w-6 shrink-0 rounded-full"
              style={{ backgroundColor: entry.color }}
            />
            {entry.label}
          </li>
        ))}

        <li className="my-0.5 border-t border-slate-200" aria-hidden />

        {Object.entries(INCIDENT_LEGEND).map(([key, entry]) => {
          const Icon = entry.icon;
          return (
            <li key={key} className="flex items-center gap-2">
              <Icon size={13} className="shrink-0" style={{ color: entry.color }} aria-hidden />
              {entry.label}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
