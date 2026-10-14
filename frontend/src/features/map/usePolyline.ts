import { useEffect, useRef } from 'react';
import { useMap, useMapsLibrary } from '@vis.gl/react-google-maps';

import type { Coordinates } from '@/api/routeService';

export interface PolylineOptions {
  path: Coordinates[] | undefined;
  color: string;
  weight: number;
  opacity?: number;
  zIndex?: number;
  visible?: boolean;
}

/**
 * One google.maps.Polyline, kept in sync with its props.
 *
 * @vis.gl/react-google-maps ships no overlay primitives -- no <Polyline> component -- so
 * the overlay is managed imperatively. The construction and the update are deliberately
 * two separate effects: building the Polyline in the same effect that sets its path would
 * tear down and recreate the overlay on every GPS fix, which flickers and leaks.
 */
export function usePolyline({
  path,
  color,
  weight,
  opacity = 1,
  zIndex = 0,
  visible = true,
}: PolylineOptions) {
  const map = useMap();
  const mapsLib = useMapsLibrary('maps');
  const polylineRef = useRef<google.maps.Polyline | null>(null);

  // Create once, when both the map and the maps library are ready.
  useEffect(() => {
    if (!map || !mapsLib) return;
    const polyline = new mapsLib.Polyline({ map, geodesic: false });
    polylineRef.current = polyline;
    return () => {
      polyline.setMap(null);
      polylineRef.current = null;
    };
  }, [map, mapsLib]);

  // Update in place.
  useEffect(() => {
    const polyline = polylineRef.current;
    if (!polyline) return;
    polyline.setOptions({
      path: (path ?? []).map((p) => ({ lat: p.latitude, lng: p.longitude })),
      strokeColor: color,
      strokeWeight: weight,
      // The Maps JS API ignores an alpha channel inside strokeColor, so the half-faded
      // "already driven" styling goes through strokeOpacity instead of an 8-digit hex.
      strokeOpacity: opacity,
      zIndex,
      visible: visible && (path?.length ?? 0) > 1,
    });
  }, [path, color, weight, opacity, zIndex, visible]);
}
