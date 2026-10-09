import { Coordinates } from '../api/routeService';

/**
 * Route-following geometry for turn-by-turn navigation on a route the server planned.
 *
 * The server returns a route as a list of road-network points, not as instructions, so
 * everything here is derived from that line: where the driver is along it, how far they
 * have strayed from it, and where the road next bends enough to call it a turn.
 */

const EARTH_RADIUS_M = 6_371_000;
const toRad = (deg: number) => (deg * Math.PI) / 180;
const toDeg = (rad: number) => (rad * 180) / Math.PI;

export function distanceM(a: Coordinates, b: Coordinates): number {
  const dLat = toRad(b.latitude - a.latitude);
  const dLon = toRad(b.longitude - a.longitude);
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(a.latitude)) * Math.cos(toRad(b.latitude)) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_RADIUS_M * Math.asin(Math.sqrt(h));
}

/** Compass bearing from a to b, degrees clockwise from north in [0, 360). */
export function bearingDeg(a: Coordinates, b: Coordinates): number {
  const p1 = toRad(a.latitude);
  const p2 = toRad(b.latitude);
  const dLon = toRad(b.longitude - a.longitude);
  const y = Math.sin(dLon) * Math.cos(p2);
  const x = Math.cos(p1) * Math.sin(p2) - Math.sin(p1) * Math.cos(p2) * Math.cos(dLon);
  return (toDeg(Math.atan2(y, x)) + 360) % 360;
}

/** Distance from the route's start to each of its points, in metres. */
export function cumulativeM(path: Coordinates[]): number[] {
  const out = [0];
  for (let i = 1; i < path.length; i++) out.push(out[i - 1] + distanceM(path[i - 1], path[i]));
  return out;
}

export interface RouteProgress {
  segment: number; // index of the route segment (path[segment] -> path[segment + 1]) the driver is on
  point: Coordinates; // the driver's position snapped onto the route
  offRouteM: number; // how far the driver is from the route
  travelledM: number;
  remainingM: number;
  totalM: number;
}

/**
 * Where `p` is along the route: the nearest point on the segments from `fromSegment` (where
 * the driver last was) up to `aheadM` metres further on.
 *
 * Searching only forward and only so far keeps progress from jumping: a route that loops
 * back near itself, or a driver who is not on the route at all, would otherwise snap to
 * whichever part of the route happens to be closest -- often near its end, which reads as
 * the trip being almost over.
 */
export function progressOnRoute(
  path: Coordinates[],
  cum: number[],
  p: Coordinates,
  fromSegment = 0,
  aheadM = Infinity
): RouteProgress {
  const totalM = cum[cum.length - 1] ?? 0;
  const first = Math.max(0, Math.min(fromSegment, path.length - 2));
  let best: RouteProgress = {
    segment: first,
    point: path[first],
    offRouteM: distanceM(p, path[first]),
    travelledM: cum[first],
    remainingM: Math.max(0, totalM - cum[first]),
    totalM,
  };
  // Metres per degree around the driver; flat-earth is exact enough over one road segment.
  const mLat = 110_540;
  const mLon = 111_320 * Math.cos(toRad(p.latitude));
  for (let i = first; i < path.length - 1 && cum[i] - cum[first] <= aheadM; i++) {
    const a = path[i];
    const b = path[i + 1];
    const bx = (b.longitude - a.longitude) * mLon;
    const by = (b.latitude - a.latitude) * mLat;
    const px = (p.longitude - a.longitude) * mLon;
    const py = (p.latitude - a.latitude) * mLat;
    const len2 = bx * bx + by * by;
    const t = len2 > 0 ? Math.max(0, Math.min(1, (px * bx + py * by) / len2)) : 0;
    const off = Math.hypot(px - t * bx, py - t * by);
    if (off < best.offRouteM) {
      const travelledM = cum[i] + t * (cum[i + 1] - cum[i]);
      best = {
        segment: i,
        point: { latitude: a.latitude + t * (b.latitude - a.latitude), longitude: a.longitude + t * (b.longitude - a.longitude) },
        offRouteM: off,
        travelledM,
        remainingM: Math.max(0, totalM - travelledM),
        totalM,
      };
    }
  }
  return best;
}

/** The part of the route still ahead of the driver, starting where they are on it. */
export function remainingPath(path: Coordinates[], progress: RouteProgress): Coordinates[] {
  return [progress.point, ...path.slice(progress.segment + 1)];
}

export type ManeuverKind = 'straight' | 'slight-left' | 'slight-right' | 'left' | 'right' | 'u-turn' | 'arrive';

export interface Maneuver {
  kind: ManeuverKind;
  distanceM: number; // from the driver to the maneuver
}

// A bend shallower than this is the road curving, not a turn to announce.
const TURN_DEG = 30;
// A change of direction smaller than this at one point is drawing noise, not the start of a bend.
const BEND_START_DEG = 8;
// A junction is often drawn as several short segments; the bends within this distance of
// the first are added up and announced as one turn.
const JUNCTION_M = 25;

/** Signed change of direction at path[j], degrees in (-180, 180]; positive is a right turn. */
function bendAt(path: Coordinates[], j: number): number {
  return ((bearingDeg(path[j], path[j + 1]) - bearingDeg(path[j - 1], path[j]) + 540) % 360) - 180;
}

/** The next turn along the route ahead of the driver, or the arrival if none is left. */
export function nextManeuver(path: Coordinates[], cum: number[], progress: RouteProgress): Maneuver {
  let j = progress.segment + 1;
  while (j < path.length - 1) {
    if (cum[j + 1] - cum[j] < 0.5 || cum[j] - cum[j - 1] < 0.5) {
      j++; // a repeated point has no direction
      continue;
    }
    if (Math.abs(bendAt(path, j)) < BEND_START_DEG) {
      j++;
      continue;
    }
    // The whole bend: every change of direction within JUNCTION_M of where it starts.
    let turn = 0;
    let k = j;
    while (k < path.length - 1 && cum[k] - cum[j] <= JUNCTION_M) {
      if (cum[k + 1] - cum[k] >= 0.5 && cum[k] - cum[k - 1] >= 0.5) turn += bendAt(path, k);
      k++;
    }
    const size = Math.abs(turn);
    if (size >= TURN_DEG) {
      const distance = Math.max(0, cum[j] - progress.travelledM);
      if (size >= 150) return { kind: 'u-turn', distanceM: distance };
      const side = turn < 0 ? 'left' : 'right';
      return { kind: size < 50 ? (`slight-${side}` as ManeuverKind) : side, distanceM: distance };
    }
    j = k;
  }
  return { kind: 'arrive', distanceM: progress.remainingM };
}

export function maneuverText(kind: ManeuverKind): string {
  switch (kind) {
    case 'left':
      return 'Turn left';
    case 'right':
      return 'Turn right';
    case 'slight-left':
      return 'Keep left';
    case 'slight-right':
      return 'Keep right';
    case 'u-turn':
      return 'Make a U-turn';
    case 'arrive':
      return 'Arrive at your destination';
    default:
      return 'Continue straight';
  }
}

export function formatDistance(m: number): string {
  if (m >= 1000) return `${(m / 1000).toFixed(1)} km`;
  if (m >= 100) return `${Math.round(m / 50) * 50} m`;
  return `${Math.max(10, Math.round(m / 10) * 10)} m`;
}
