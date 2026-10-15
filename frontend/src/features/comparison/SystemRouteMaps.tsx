import { useEffect } from 'react';
import {
  AdvancedMarker,
  AdvancedMarkerAnchorPoint,
  Map,
  useMap,
  useMapsLibrary,
} from '@vis.gl/react-google-maps';

import type { ComparisonTripDetail, Coordinates } from '@/api/routeService';
import { MANILA_CENTER, MANILA_DEFAULT_ZOOM } from '@/lib/constants';
import { GOOGLE_MAPS_MAP_ID, hasMapsKey } from '@/lib/maps-env';
import { CHART_ANT, CHART_BASELINE } from '@/lib/theme';
import { usePolyline } from '@/features/map/usePolyline';

function Line(props: Parameters<typeof usePolyline>[0]) {
  usePolyline(props);
  return null;
}

/** Both routes share the frame, so the two maps are directly comparable. */
function FitTo({ paths }: { paths: Coordinates[][] }) {
  const map = useMap();
  const core = useMapsLibrary('core');
  const flat = paths.flat();
  useEffect(() => {
    if (!map || !core || flat.length < 2) return;
    const bounds = new core.LatLngBounds();
    for (const p of flat) bounds.extend({ lat: p.latitude, lng: p.longitude });
    map.fitBounds(bounds, 30);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, core, flat.length]);
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

function SystemMap({
  title,
  color,
  path,
  bothPaths,
}: {
  title: string;
  color: string;
  path: Coordinates[];
  bothPaths: Coordinates[][];
}) {
  return (
    <div>
      <p className="mb-1.5">
        <span
          className="inline-block rounded-md px-2.5 py-1 text-xs font-semibold text-white"
          style={{ backgroundColor: color }}
        >
          {title}
        </span>
      </p>
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
          <Line path={path} color={color} weight={5} opacity={1} zIndex={2} />
          <Endpoints path={path} />
          {/* Both maps frame the same area so the routes can be compared by eye. */}
          <FitTo paths={bothPaths} />
        </Map>
      </div>
    </div>
  );
}

/**
 * The route each system planned for the same trip, side by side. These are the evaluated
 * routes themselves -- trials.csv stores each one's node path, so nothing is re-routed here.
 */
export default function SystemRouteMaps({
  detail,
  baselineName,
}: {
  detail: ComparisonTripDetail;
  baselineName: string;
}) {
  const both = [detail.antroute_path, detail.baseline_path];

  if (!hasMapsKey) {
    return (
      <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-6 text-center text-sm text-slate-500">
        Set VITE_GOOGLE_MAPS_API_KEY to see both routes on a map. The metrics below do not
        need it.
      </p>
    );
  }
  if (detail.antroute_path.length < 2 && detail.baseline_path.length < 2) {
    return (
      <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-6 text-center text-sm text-slate-500">
        The routing graph is not loaded on the server, so these routes cannot be drawn.
      </p>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-5">
      <SystemMap title="ANTRoute" color={CHART_ANT} path={detail.antroute_path} bothPaths={both} />
      <SystemMap
        title={baselineName}
        color={CHART_BASELINE}
        path={detail.baseline_path}
        bothPaths={both}
      />
    </div>
  );
}
