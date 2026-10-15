import { useMemo } from 'react';
import { AdvancedMarker, AdvancedMarkerAnchorPoint } from '@vis.gl/react-google-maps';
import { MapPin } from 'lucide-react';

import type { Coordinates } from '@/api/routeService';
import { ANT_COLOR, BASELINE_COLOR, ROUTE_OPACITY, TRAVELLED_OPACITY } from '@/lib/theme';
import { remainingPath, type RouteProgress } from '@/lib/guidance';
import { useTripPlanner } from '@/features/trip/TripPlannerContext';
import type { MapIncident } from '@/types/markers';

import { isInForce } from '@/features/upload/parseInputFeaturesCsv';

import { alternativeColor } from './legend';
import { INCIDENT_LEGEND } from './legend';
import { useFitBounds, usePanTo } from './useFitBounds';
import { usePolyline } from './usePolyline';

/** One polyline. A component only so each one gets its own hook instance. */
function Polyline(props: Parameters<typeof usePolyline>[0]) {
  usePolyline(props);
  return null;
}

/**
 * The route overlays. The selected route draws on top; the other ANTRoute options draw
 * underneath as alternatives, and the baseline draws alongside when it has a route.
 */
export function RoutePolylines({ navProgress = null }: { navProgress?: RouteProgress | null }) {
  const { antRouteOptions, baselineModelOptions, selectedModelTab, selectedIndex, selectedRoute } =
    useTripPlanner();

  const alternatives = selectedModelTab === 'antroute' ? antRouteOptions : baselineModelOptions;
  const baselineTop = baselineModelOptions[0];
  const selectedColor = selectedModelTab === 'baseline' ? BASELINE_COLOR : ANT_COLOR;

  return (
    <>
      {/* Baseline, under everything, whenever it differs from what is selected. */}
      {selectedModelTab === 'antroute' && baselineTop?.path?.length ? (
        <Polyline
          path={baselineTop.path}
          color={BASELINE_COLOR}
          weight={5}
          opacity={ROUTE_OPACITY}
          zIndex={1}
        />
      ) : null}

      {/* The options that are not selected. */}
      {alternatives.map((route, index) =>
        index === selectedIndex ? null : (
          <Polyline
            key={`alt-${index}`}
            path={route.path}
            color={alternativeColor(index === 0 ? 1 : index)}
            weight={4}
            opacity={0.75}
            zIndex={2}
          />
        )
      )}

      {/* The selected route, on top. While navigating it fades to show the part already
          driven, and the part still ahead is drawn over it in full colour. */}
      {selectedRoute?.path?.length ? (
        <Polyline
          path={selectedRoute.path}
          color={selectedColor}
          weight={navProgress ? 8 : 6}
          opacity={navProgress ? TRAVELLED_OPACITY : ROUTE_OPACITY}
          zIndex={3}
        />
      ) : null}

      {navProgress && selectedRoute?.path?.length ? (
        <Polyline
          path={remainingPath(selectedRoute.path, navProgress)}
          color={selectedColor}
          weight={8}
          opacity={ROUTE_OPACITY}
          zIndex={4}
        />
      ) : null}
    </>
  );
}

/** Origin dot and numbered stop pins, replacing the custom react-native-maps markers. */
export function StopMarkers() {
  const { originCoords, destinations } = useTripPlanner();
  const lastPlacedIndex = destinations.reduce((last, d, i) => (d.coords ? i : last), -1);
  const showNumbers = destinations.length > 1;

  return (
    <>
      {originCoords && (
        <AdvancedMarker
          position={{ lat: originCoords.latitude, lng: originCoords.longitude }}
          anchorPoint={AdvancedMarkerAnchorPoint.CENTER}
          title="Start"
        >
          <span className="block h-4 w-4 rounded-full border-2 border-white bg-brand shadow-md" />
        </AdvancedMarker>
      )}

      {destinations.map((dest, index) => {
        if (!dest.coords) return null;
        const isLast = index === lastPlacedIndex;
        return (
          <AdvancedMarker
            key={`stop-${index}`}
            position={{ lat: dest.coords.latitude, lng: dest.coords.longitude }}
            anchorPoint={AdvancedMarkerAnchorPoint.CENTER}
            title={showNumbers ? `Stop ${index + 1}: ${dest.name}` : dest.name}
          >
            <span
              className={[
                'flex h-6 w-6 items-center justify-center rounded-full border-2 border-white text-[11px] font-bold text-white shadow-md',
                isLast ? 'bg-red-500' : 'bg-brand',
              ].join(' ')}
            >
              {showNumbers ? index + 1 : <MapPin size={12} />}
            </span>
          </AdvancedMarker>
        );
      })}
    </>
  );
}

/**
 * Incident markers, driven entirely by data. The source is empty today (see
 * src/data/incidentSource.ts) so nothing renders; the glyphs come from the same legend
 * map the legend panel uses, so wiring a real feed needs no change here.
 */
export function IncidentMarkers({
  incidents,
  at,
}: {
  incidents: MapIncident[];
  /** When the trip departs; an event outside its own window is dimmed, not hidden. */
  at?: Date | null;
}) {
  return (
    <>
      {incidents.map((incident) => {
        const entry = INCIDENT_LEGEND[incident.kind];
        const Icon = entry.icon;
        const live = at ? isInForce(incident, at) : true;
        return (
          <AdvancedMarker
            key={incident.id}
            position={{ lat: incident.position.latitude, lng: incident.position.longitude }}
            anchorPoint={AdvancedMarkerAnchorPoint.CENTER}
            title={
              (incident.description ? `${incident.title} — ${incident.description}` : incident.title) +
              (live ? '' : ' (not in force at the planned departure)')
            }
          >
            <span
              className="flex h-6 w-6 items-center justify-center rounded-full border-2 border-white shadow-md"
              style={{ backgroundColor: entry.color, opacity: live ? 1 : 0.35 }}
            >
              <Icon size={13} color="#fff" aria-hidden />
            </span>
          </AdvancedMarker>
        );
      })}
    </>
  );
}

const LABEL_COLOR = { Light: '#10b981', Medium: '#f59e0b', Heavy: '#ef4444' } as const;

/**
 * The cameras and incidents of the recorded window the routes were planned on: each camera is
 * coloured by its human congestion label there, each incident by the incident glyph. Recorded
 * data, so nothing appears until a trip has been planned.
 */
export function RecordedMarkers() {
  const { recordedInputs } = useTripPlanner();
  if (!recordedInputs) return null;
  const camera = INCIDENT_LEGEND.camera;
  const incident = INCIDENT_LEGEND.accident;
  const CameraIcon = camera.icon;
  const IncidentIcon = incident.icon;

  return (
    <>
      {recordedInputs.cameras.map((c) => (
        <AdvancedMarker
          key={`cam-${c.camera}`}
          position={{ lat: c.lat, lng: c.lon }}
          anchorPoint={AdvancedMarkerAnchorPoint.CENTER}
          title={`${c.camera}: ${c.human_label ?? 'no label'}${c.vehicle_count != null ? `, ${c.vehicle_count} vehicles` : ''} (recorded)`}
        >
          <span
            className="flex h-6 w-6 items-center justify-center rounded-full border-2 border-white shadow-md"
            style={{ backgroundColor: c.human_label ? LABEL_COLOR[c.human_label] : camera.color }}
          >
            <CameraIcon size={12} color="#fff" aria-hidden />
          </span>
        </AdvancedMarker>
      ))}
      {recordedInputs.incidents.map((i, n) =>
        i.lat != null && i.lon != null ? (
          <AdvancedMarker
            key={`inc-${n}`}
            position={{ lat: i.lat, lng: i.lon }}
            anchorPoint={AdvancedMarkerAnchorPoint.CENTER}
            title={`${i.location}: ${i.kind}, ${i.lanes} lane(s) (recorded)`}
          >
            <span
              className="flex h-6 w-6 items-center justify-center rounded-full border-2 border-white shadow-md"
              style={{ backgroundColor: incident.color }}
            >
              <IncidentIcon size={13} color="#fff" aria-hidden />
            </span>
          </AdvancedMarker>
        ) : null
      )}
    </>
  );
}

/** Keeps the camera on whatever the planner last asked for. */
export function CameraController({ followingDriver = false }: { followingDriver?: boolean }) {
  const { cameraTarget, selectedRoute, antTop, baselineTop, hasPlannedTrip } = useTripPlanner();

  // Fit to the planned route; falls back to both models' paths so a comparison fits too.
  const fitPoints = useMemo<Coordinates[]>(() => {
    if (!hasPlannedTrip) return [];
    if (selectedRoute?.path?.length) return selectedRoute.path;
    return [...(antTop?.path ?? []), ...(baselineTop?.path ?? [])];
  }, [hasPlannedTrip, selectedRoute, antTop, baselineTop]);

  // While navigating the camera follows the driver instead (see useTurnByTurn).
  useFitBounds(fitPoints, !followingDriver && fitPoints.length > 1);
  usePanTo(followingDriver ? null : cameraTarget);

  return null;
}
