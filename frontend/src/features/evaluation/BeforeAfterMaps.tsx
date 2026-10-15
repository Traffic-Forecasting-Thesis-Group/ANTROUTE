import { useEffect, useMemo } from 'react';
import { AdvancedMarker, AdvancedMarkerAnchorPoint, Map, useMap, useMapsLibrary } from '@vis.gl/react-google-maps';
import { TriangleAlert } from 'lucide-react';

import type { ConditionUpdateDetail, Coordinates } from '@/api/routeService';
import { MANILA_CENTER, MANILA_DEFAULT_ZOOM } from '@/lib/constants';
import { GOOGLE_MAPS_MAP_ID, hasMapsKey } from '@/lib/maps-env';
import { ANT_COLOR } from '@/lib/theme';
import { usePolyline } from '@/features/map/usePolyline';

const AFTER_COLOR = '#B91C1C'; // the recalculated route, so before/after never read as the same line
const OLD_GHOST = '#94A3B8';

function Line(props: Parameters<typeof usePolyline>[0]) {
  usePolyline(props);
  return null;
}

/** Fits a small map to one route and keeps it there. */
function FitTo({ path }: { path: Coordinates[] }) {
  const map = useMap();
  const core = useMapsLibrary('core');
  useEffect(() => {
    if (!map || !core || path.length < 2) return;
    const bounds = new core.LatLngBounds();
    for (const p of path) bounds.extend({ lat: p.latitude, lng: p.longitude });
    map.fitBounds(bounds, 28);
  }, [map, core, path]);
  return null;
}

function Endpoints({ path }: { path: Coordinates[] }) {
  if (path.length < 2) return null;
  const start = path[0];
  const end = path[path.length - 1];
  return (
    <>
      <AdvancedMarker
        position={{ lat: start.latitude, lng: start.longitude }}
        anchorPoint={AdvancedMarkerAnchorPoint.CENTER}
        title="Start"
      >
        <span className="block h-3.5 w-3.5 rounded-full border-2 border-white bg-emerald-500 shadow" />
      </AdvancedMarker>
      <AdvancedMarker
        position={{ lat: end.latitude, lng: end.longitude }}
        anchorPoint={AdvancedMarkerAnchorPoint.CENTER}
        title="Destination"
      >
        <span className="block h-3.5 w-3.5 rounded-full border-2 border-white bg-red-500 shadow" />
      </AdvancedMarker>
    </>
  );
}

function IncidentMarkers({ detail }: { detail: ConditionUpdateDetail }) {
  return (
    <>
      {detail.incidents
        .filter((i) => i.lat != null && i.lon != null)
        .map((i) => (
          <AdvancedMarker
            key={`${i.location}-${i.reported_at}`}
            position={{ lat: i.lat as number, lng: i.lon as number }}
            anchorPoint={AdvancedMarkerAnchorPoint.CENTER}
            title={`${i.kind} at ${i.location}`}
          >
            <span className="flex h-6 w-6 items-center justify-center rounded-full border-2 border-white bg-red-600 shadow">
              <TriangleAlert size={13} color="#fff" aria-hidden />
            </span>
          </AdvancedMarker>
        ))}
    </>
  );
}

function MiniMap({
  path,
  color,
  ghost,
  detail,
  showIncidents,
}: {
  path: Coordinates[];
  color: string;
  ghost?: Coordinates[];
  detail: ConditionUpdateDetail;
  showIncidents: boolean;
}) {
  return (
    <div className="h-[230px] overflow-hidden rounded-lg border border-slate-200">
      <Map
        mapId={GOOGLE_MAPS_MAP_ID || undefined}
        defaultCenter={MANILA_CENTER}
        defaultZoom={MANILA_DEFAULT_ZOOM}
        disableDefaultUI
        clickableIcons={false}
        gestureHandling="cooperative"
        className="h-full w-full"
      >
        {/* The route being replaced, under the new one, so the change is visible at a glance. */}
        {ghost && ghost.length > 1 && (
          <Line path={ghost} color={OLD_GHOST} weight={5} opacity={0.65} zIndex={1} />
        )}
        <Line path={path} color={color} weight={5} opacity={1} zIndex={2} />
        <Endpoints path={path} />
        {showIncidents && <IncidentMarkers detail={detail} />}
        <FitTo path={path.length > 1 ? path : (ghost ?? [])} />
      </Map>
    </div>
  );
}

function Swatch({ color, label, dashed = false }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="flex items-center gap-1.5 text-[11px] text-slate-600">
      <span
        aria-hidden
        className="h-0.5 w-5 shrink-0 rounded-full"
        style={
          dashed
            ? { backgroundImage: `repeating-linear-gradient(to right, ${color} 0 4px, transparent 4px 7px)` }
            : { backgroundColor: color }
        }
      />
      {label}
    </span>
  );
}

/**
 * The route before the conditions changed, and after recalculating under them. Both are the
 * routes the harness actually planned -- their node paths are stored with the results, so
 * these maps draw the evaluated routes, not a re-run.
 */
export default function BeforeAfterMaps({ detail }: { detail: ConditionUpdateDetail }) {
  const before = detail.old_path;
  const after = detail.recalculated_path;
  const changed = detail.case.route_changed;

  const hasGeometry = useMemo(() => before.length > 1 || after.length > 1, [before, after]);

  if (!hasMapsKey) {
    return (
      <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-6 text-center text-sm text-slate-500">
        Set VITE_GOOGLE_MAPS_API_KEY to see the before and after routes on a map. The metrics
        below do not need it.
      </p>
    );
  }
  if (!hasGeometry) {
    return (
      <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-6 text-center text-sm text-slate-500">
        The routing graph is not loaded on the server, so these routes cannot be drawn.
      </p>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-5">
      <div>
        <p className="mb-1.5 text-xs font-semibold text-slate-700">
          Before <span className="font-normal text-slate-400">(original route)</span>
        </p>
        <MiniMap path={before} color={ANT_COLOR} detail={detail} showIncidents={false} />
        <div className="mt-1.5 flex flex-wrap gap-3">
          <Swatch color={ANT_COLOR} label="Original route" />
        </div>
      </div>

      <div>
        <p className="mb-1.5 text-xs font-semibold text-slate-700">
          After{' '}
          <span className="font-normal text-slate-400">
            ({changed ? 'recalculated route' : 'unchanged after the update'})
          </span>
        </p>
        <MiniMap
          path={after}
          color={changed ? AFTER_COLOR : ANT_COLOR}
          ghost={changed ? before : undefined}
          detail={detail}
          showIncidents
        />
        <div className="mt-1.5 flex flex-wrap gap-3">
          <Swatch color={changed ? AFTER_COLOR : ANT_COLOR} label="Recalculated route" />
          {changed && <Swatch color={OLD_GHOST} label="Original route" />}
          {detail.incidents.some((i) => i.lat != null) && (
            <span className="flex items-center gap-1.5 text-[11px] text-slate-600">
              <TriangleAlert size={11} className="text-red-600" aria-hidden />
              Event
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
