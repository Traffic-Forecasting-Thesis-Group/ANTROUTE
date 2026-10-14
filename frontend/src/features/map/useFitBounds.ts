import { useEffect } from 'react';
import { useMap, useMapsLibrary } from '@vis.gl/react-google-maps';

import type { Coordinates } from '@/api/routeService';
import { FIT_PADDING, PLACE_ZOOM } from '@/lib/constants';
import type { CameraTarget } from '@/features/trip/TripPlannerContext';

/**
 * Fits the camera to a set of points. Replaces react-native-maps' fitToCoordinates.
 *
 * The padding is now relative to the map card rather than the screen -- the native code
 * reserved 55% of the screen height for the bottom sheet, which has no equivalent here.
 * The extra top/right keeps the route clear of the floating legend panel.
 *
 * fitBounds is a silent no-op on a zero-size map, which is easy to hit on first paint
 * inside a flex card, so this also depends on `map` becoming available.
 */
export function useFitBounds(points: Coordinates[], enabled = true) {
  const map = useMap();
  const coreLib = useMapsLibrary('core');

  useEffect(() => {
    if (!map || !coreLib || !enabled) return;
    if (points.length < 2) return;

    const bounds = new coreLib.LatLngBounds();
    for (const p of points) bounds.extend({ lat: p.latitude, lng: p.longitude });
    map.fitBounds(bounds, FIT_PADDING);
    // points is rebuilt per render; the caller memoises what it passes in.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, coreLib, enabled, points]);
}

/** Pans to one place, replacing animateToRegion. */
export function usePanTo(target: CameraTarget | null) {
  const map = useMap();

  useEffect(() => {
    if (!map || !target) return;
    map.panTo({ lat: target.coords.latitude, lng: target.coords.longitude });
    map.setZoom(PLACE_ZOOM);
    // nonce makes repeating the same coordinates still move the camera.
  }, [map, target?.nonce]); // eslint-disable-line react-hooks/exhaustive-deps
}
