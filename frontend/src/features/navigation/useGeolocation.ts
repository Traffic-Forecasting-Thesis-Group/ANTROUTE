import { useCallback, useEffect, useRef, useState } from 'react';

import type { Coordinates } from '@/api/routeService';

/**
 * Browser geolocation, replacing expo-location.
 *
 * GeolocationPosition.coords carries the same field names as expo's LocationObject.coords
 * (latitude, longitude, accuracy, speed, heading), so the navigation code that consumes a
 * fix did not need to change -- only the subscribe/unsubscribe plumbing did.
 *
 * Two things differ on a desktop and are expected rather than bugs:
 *   - speed and heading are null in virtually every desktop browser (they are GPS-derived),
 *     so the navigation camera always falls back to the route's own bearing.
 *   - accuracy is routinely 1-3 km on a first Wi-Fi fix, which is why getAccuratePosition
 *     below keeps polling for a better one.
 *
 * Geolocation also needs a secure context. http://localhost counts, so `npm run dev` is
 * fine, but opening the dev server over a LAN IP silently never calls back.
 */

const TARGET_ACCURACY_M = 30;
const ACCURACY_WAIT_MS = 8000;

export type PermissionState = 'unknown' | 'granted' | 'denied' | 'prompt';

function accuracyOf(p: GeolocationPosition): number {
  return p.coords.accuracy ?? Infinity;
}

export function toCoordinates(p: GeolocationPosition): Coordinates {
  return { latitude: p.coords.latitude, longitude: p.coords.longitude };
}

function getCurrentPosition(options: PositionOptions): Promise<GeolocationPosition> {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('This browser has no geolocation support.'));
      return;
    }
    navigator.geolocation.getCurrentPosition(resolve, reject, options);
  });
}

/**
 * A first fix, then a short watch for a better one. The first fix on a desktop is often
 * far too coarse to start a trip from, so settle for it only once it is good enough or the
 * wait is up.
 */
export async function getAccuratePosition(): Promise<GeolocationPosition> {
  let best = await getCurrentPosition({
    enableHighAccuracy: true,
    timeout: 10000,
    maximumAge: 0,
  });
  if (accuracyOf(best) <= TARGET_ACCURACY_M) return best;

  return new Promise((resolve) => {
    let watchId: number | null = null;
    let done = false;

    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      if (watchId !== null) navigator.geolocation.clearWatch(watchId);
      resolve(best);
    };

    const timer = setTimeout(finish, ACCURACY_WAIT_MS);

    try {
      watchId = navigator.geolocation.watchPosition(
        (pos) => {
          if (accuracyOf(pos) < accuracyOf(best)) best = pos;
          if (accuracyOf(best) <= TARGET_ACCURACY_M) finish();
        },
        finish,
        { enableHighAccuracy: true, maximumAge: 0 }
      );
    } catch {
      finish();
    }
  });
}

export function useGeolocation() {
  const [permission, setPermission] = useState<PermissionState>('unknown');
  const [isLocating, setIsLocating] = useState(false);
  const watchRef = useRef<number | null>(null);

  // There is no explicit permission request in the browser -- the prompt fires on the
  // first getCurrentPosition. This only reads the current state so the UI can explain
  // itself when access was already refused.
  useEffect(() => {
    if (!navigator.permissions?.query) return;
    let cancelled = false;
    navigator.permissions
      .query({ name: 'geolocation' as PermissionName })
      .then((status) => {
        if (cancelled) return;
        setPermission(status.state as PermissionState);
        status.onchange = () => setPermission(status.state as PermissionState);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  /** A single accurate fix. Throws with a message fit to show the user. */
  const locate = useCallback(async (): Promise<Coordinates> => {
    setIsLocating(true);
    try {
      const position = await getAccuratePosition();
      setPermission('granted');
      return toCoordinates(position);
    } catch (error: any) {
      if (error?.code === 1) {
        setPermission('denied');
        throw new Error('Location permission denied.');
      }
      throw new Error('Could not get your location.');
    } finally {
      setIsLocating(false);
    }
  }, []);

  const stopWatching = useCallback(() => {
    if (watchRef.current !== null) {
      navigator.geolocation.clearWatch(watchRef.current);
      watchRef.current = null;
    }
  }, []);

  const startWatching = useCallback(
    (onFix: (position: GeolocationPosition) => void, onError?: (error: unknown) => void) => {
      if (!navigator.geolocation) return;
      stopWatching();
      watchRef.current = navigator.geolocation.watchPosition(
        onFix,
        (err) => onError?.(err),
        { enableHighAccuracy: true, maximumAge: 1000 }
      );
    },
    [stopWatching]
  );

  // Never leave a watch running after the component goes away.
  useEffect(() => stopWatching, [stopWatching]);

  return { permission, isLocating, locate, startWatching, stopWatching };
}
