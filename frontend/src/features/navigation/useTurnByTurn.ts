import { useCallback, useEffect, useRef, useState } from 'react';
import { useMap } from '@vis.gl/react-google-maps';

import type { Coordinates } from '@/api/routeService';
import {
  bearingDeg,
  cumulativeM,
  distanceM,
  nextManeuver,
  progressOnRoute,
  type Maneuver,
  type RouteProgress,
} from '@/lib/guidance';
import {
  ARRIVAL_RADIUS_M,
  NAV_FOLLOW_ZOOM,
  OFF_ROUTE_FIXES,
  OFF_ROUTE_M,
} from '@/lib/constants';
import { useTripPlanner } from '@/features/trip/TripPlannerContext';

import { useGeolocation } from './useGeolocation';

export interface TurnByTurnState {
  isNavigating: boolean;
  progress: RouteProgress | null;
  maneuver: Maneuver | null;
  arrived: boolean;
  isRerouting: boolean;
  targetName: string;
  error: string;
  remainingMin: number;
  remainingKm: number;
  start: () => void;
  stop: () => void;
}

/**
 * Turn-by-turn guidance along the selected route.
 *
 * On a desktop two things behave differently from the phone build, by nature rather than
 * by bug:
 *   - coords.speed and coords.heading are null in virtually every desktop browser, so the
 *     camera heading always comes from the route's own bearing.
 *   - coords.accuracy routinely exceeds OFF_ROUTE_M, and an inaccurate fix does not count
 *     toward the off-route tally, so auto-reroute rarely triggers on a laptop. That is the
 *     correct trade -- a 1 km fix must not be read as a wrong turn.
 */
export function useTurnByTurn(): TurnByTurnState {
  const map = useMap();
  const { startWatching, stopWatching } = useGeolocation();
  const {
    selectedRoute,
    destinations,
    setGpsCoords,
    closeSuggestions,
    replanFrom,
  } = useTripPlanner();

  const [isNavigating, setIsNavigating] = useState(false);
  const [progress, setProgress] = useState<RouteProgress | null>(null);
  const [maneuver, setManeuver] = useState<Maneuver | null>(null);
  const [arrived, setArrived] = useState(false);
  const [isRerouting, setIsRerouting] = useState(false);
  const [stopIndex, setStopIndex] = useState(0);
  const [error, setError] = useState('');

  // The fix handler outlives the render that created it, so it reads the live route and
  // stops through refs rather than through a closed-over render's values.
  const routeRef = useRef(selectedRoute);
  routeRef.current = selectedRoute;
  const placedStopsRef = useRef(destinations.filter((d) => d.coords));
  placedStopsRef.current = destinations.filter((d) => d.coords);
  const cumRef = useRef<{ path: Coordinates[] | null; cum: number[] }>({ path: null, cum: [] });
  const stopIndexRef = useRef(0);
  const offRouteCountRef = useRef(0);
  const reroutingRef = useRef(false);

  const onFix = useCallback(
    (pos: GeolocationPosition) => {
      const here: Coordinates = {
        latitude: pos.coords.latitude,
        longitude: pos.coords.longitude,
      };
      setGpsCoords(here);

      const path = routeRef.current?.path ?? [];
      if (path.length < 2) return;
      if (cumRef.current.path !== path) cumRef.current = { path, cum: cumulativeM(path) };
      const cum = cumRef.current.cum;

      const next = progressOnRoute(path, cum, here);
      setProgress(next);
      setManeuver(nextManeuver(path, cum, next));

      // Follow the driver, pointing where they are heading: the GPS heading when moving,
      // else the direction of the road they are on (the usual case on a desktop).
      const moving = (pos.coords.speed ?? 0) > 1 && (pos.coords.heading ?? -1) >= 0;
      const heading = moving
        ? (pos.coords.heading as number)
        : bearingDeg(path[next.segment], path[Math.min(next.segment + 1, path.length - 1)]);
      map?.moveCamera({
        center: { lat: here.latitude, lng: here.longitude },
        heading,
        tilt: 45,
        zoom: NAV_FOLLOW_ZOOM,
      });

      // Reaching a stop: the next becomes the target, and the last one ends the trip.
      const stops = placedStopsRef.current;
      const target = stops[stopIndexRef.current];
      if (target?.coords && distanceM(here, target.coords) <= ARRIVAL_RADIUS_M) {
        if (stopIndexRef.current >= stops.length - 1) {
          setArrived(true);
          stopWatching();
          return;
        }
        stopIndexRef.current += 1;
        setStopIndex(stopIndexRef.current);
      }

      // Off route for several fixes in a row -- one bad fix is GPS noise, not a wrong turn.
      // An inaccurate fix never counts, which on a desktop means this rarely fires.
      const accurate = (pos.coords.accuracy ?? 0) <= OFF_ROUTE_M;
      offRouteCountRef.current =
        accurate && next.offRouteM > OFF_ROUTE_M ? offRouteCountRef.current + 1 : 0;
      if (offRouteCountRef.current >= OFF_ROUTE_FIXES && !reroutingRef.current) {
        reroutingRef.current = true;
        offRouteCountRef.current = 0;
        setIsRerouting(true);
        const ahead = stops.slice(stopIndexRef.current);
        stopIndexRef.current = 0;
        setStopIndex(0);
        void replanFrom(here, ahead).finally(() => {
          reroutingRef.current = false;
          setIsRerouting(false);
        });
      }
    },
    [map, setGpsCoords, stopWatching, replanFrom]
  );

  const start = useCallback(() => {
    if (!selectedRoute?.path?.length) {
      setError('Plan a route first.');
      return;
    }
    setError('');
    closeSuggestions();
    setIsNavigating(true);
    setArrived(false);
    setProgress(null);
    setManeuver(null);
    stopIndexRef.current = 0;
    setStopIndex(0);
    offRouteCountRef.current = 0;

    startWatching(onFix, () => {
      setError('Could not get your location. Check that location access is allowed.');
    });
  }, [selectedRoute, closeSuggestions, startWatching, onFix]);

  const stop = useCallback(() => {
    stopWatching();
    setIsNavigating(false);
    setArrived(false);
    setProgress(null);
    setManeuver(null);
    setError('');
  }, [stopWatching]);

  // Never leave a GPS watch running behind the page.
  useEffect(() => stopWatching, [stopWatching]);

  // What is left of the trip: the route's own time scaled by the share of its distance
  // still ahead.
  const remainingMin =
    isNavigating && progress && selectedRoute && progress.totalM > 0
      ? Math.max(
          arrived ? 0 : 1,
          Math.round((selectedRoute.duration_min * progress.remainingM) / progress.totalM)
        )
      : (selectedRoute?.duration_min ?? 0);
  const remainingKm =
    isNavigating && progress
      ? Math.round(progress.remainingM / 100) / 10
      : (selectedRoute?.distance_km ?? 0);

  const targetName =
    placedStopsRef.current[stopIndex]?.name ?? destinations[destinations.length - 1]?.name ?? '';

  return {
    isNavigating,
    progress,
    maneuver,
    arrived,
    isRerouting,
    targetName,
    error,
    remainingMin,
    remainingKm,
    start,
    stop,
  };
}
