import { Camera, CloudRain, Waves, AlertTriangle } from 'lucide-react';

import type { CameraReading, RecordedInputs } from '@/api/routeService';
import { cn } from '@/lib/cn';

import { useTripPlanner } from './TripPlannerContext';

const LABEL_STYLE: Record<string, string> = {
  Light: 'bg-emerald-100 text-emerald-700',
  Medium: 'bg-amber-100 text-amber-700',
  Heavy: 'bg-red-100 text-red-700',
};

function Row({ icon: Icon, title, children }: { icon: typeof Camera; title: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-2">
      <Icon size={14} className="mt-0.5 shrink-0 text-slate-400" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">{title}</p>
        <div className="text-xs text-slate-700">{children}</div>
      </div>
    </div>
  );
}

function CameraLine({ camera }: { camera: CameraReading }) {
  return (
    <li className="flex items-center justify-between gap-2 py-0.5">
      <span className="truncate">{camera.camera}</span>
      <span className="flex shrink-0 items-center gap-1.5 tabular-nums text-slate-500">
        {camera.vehicle_count != null && <span>{camera.vehicle_count} veh</span>}
        {camera.model_risk != null && <span>risk {camera.model_risk.toFixed(2)}</span>}
        {camera.human_label && (
          <span className={cn('rounded-full px-1.5 py-0.5 text-[10px] font-semibold', LABEL_STYLE[camera.human_label])}>
            {camera.human_label}
          </span>
        )}
      </span>
    </li>
  );
}

/** Weather as one line: "32.7 °C, 0.2 mm rain, 56% humidity". */
export function weatherLine(w: RecordedInputs['weather']): string {
  if (!w.available) return w.note ?? 'No weather recorded.';
  const parts = [
    w.temp_c != null ? `${w.temp_c} °C` : null,
    w.precip_mm != null ? `${w.precip_mm} mm rain` : null,
    w.humidity_pct != null ? `${w.humidity_pct}% humidity` : null,
  ].filter(Boolean);
  return parts.join(', ');
}

/**
 * "Inputs used": every recorded input of the window the routes were planned on -- the cameras'
 * labels and vehicle counts, the incidents in force, that day's weather, the flood hazard. All
 * of it is labelled "recorded": there is no live feed.
 */
export default function RecordedInputsPanel() {
  const { recordedInputs, simulationWindow, selectedRoute } = useTripPlanner();
  if (!recordedInputs) return null;
  const alongRoute = selectedRoute?.inputs?.weather;
  const { matched_window: w, cameras, incidents, weather, flood } = recordedInputs;

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-card">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-slate-800">Inputs used</h3>
        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-slate-500">
          {recordedInputs.source === 'uploaded' ? 'Uploaded' : 'Recorded'}
        </span>
      </div>

      <p className="mb-3 text-xs text-slate-600">
        {recordedInputs.source === 'uploaded'
          ? `From ${recordedInputs.source_name}: `
          : simulationWindow
            ? 'Simulated window: '
            : 'Matched window: '}
        <span className="font-semibold">{w.label}</span>
        {w.held_out && (
          <span className="ml-1.5 rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold text-amber-800">
            held out
          </span>
        )}
      </p>

      <div className="flex flex-col gap-3">
        <Row icon={AlertTriangle} title={`Active incidents (${incidents.length})`}>
          {incidents.length === 0 ? (
            <span className="text-slate-400">None reported in force.</span>
          ) : (
            <ul>
              {incidents.map((i, n) => (
                <li key={n} className="py-0.5">
                  {i.location} — {i.kind}, {i.lanes} lane{i.lanes === 1 ? '' : 's'}
                  {i.minutes_ago != null && `, ${i.minutes_ago} min ago`}
                  <span className="text-slate-400"> (strength {i.strength})</span>
                </li>
              ))}
            </ul>
          )}
        </Row>

        <Row icon={CloudRain} title="Weather that day">
          {alongRoute?.available ? (
            <>
              Along the selected route: {weatherLine(alongRoute)}
              <span className="text-slate-400"> · {alongRoute.where}</span>
              <br />
            </>
          ) : null}
          {alongRoute?.available ? 'City-wide: ' : ''}
          {weatherLine(weather)}
          {weather.available && weather.hour && <span className="text-slate-400"> · {weather.hour}</span>}
        </Row>

        <Row icon={Waves} title="Flood hazard">
          {flood.share_of_roads_flood_prone != null
            ? `${Math.round(flood.share_of_roads_flood_prone * 100)}% of roads are flood-prone (NOAH, mean level ${flood.mean_level ?? '–'} of 3)`
            : 'Not available.'}
        </Row>

        <Row icon={Camera} title={`Cameras (${cameras.length})`}>
          <ul className="divide-y divide-slate-100">
            {cameras.map((c) => (
              <CameraLine key={c.camera} camera={c} />
            ))}
          </ul>
          <p className="mt-1 text-[10px] text-slate-400">
            Human label, YOLO vehicle count and the model&rsquo;s risk at each camera in this window.
            {!recordedInputs.vehicle_counts_available && ' Vehicle counts are not loaded on the server.'}
          </p>
        </Row>
      </div>
    </section>
  );
}
