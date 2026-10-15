import type { Coordinates } from '@/api/routeService';

/**
 * SEAM: the incident/marker layer.
 *
 * The map legend and the marker layer both read these types, so nothing can drift between
 * what the legend advertises and what actually renders. There is no data source yet -- the
 * backend has no incident, weather, flood or CCTV feed -- so src/data/incidentSource.ts
 * ships an empty implementation. Wiring real data means replacing that one export; no UI
 * file has to change.
 */

/** One legend row and one marker glyph. Adding a kind is the only change the UI needs. */
export type IncidentKind =
  | 'accident' // "Accident / Event"
  | 'road_closure'
  | 'traffic' // a segment the recorded data has in the heavy band
  | 'weather_rain' // "Weather (Rain)"
  | 'flood'
  | 'camera'; // "Camera (CCTV)"

/**
 * Traffic band on a road segment. These are the legend's three bands; the backend's
 * CongestionLevel uses 'clear' where the legend says "Light", and adds 'unknown'.
 */
export type TrafficBand = 'heavy' | 'moderate' | 'light';

export interface MapIncident {
  id: string;
  kind: IncidentKind;
  position: Coordinates;
  title: string;
  description?: string | null;
  severity?: TrafficBand | null;
  /** ISO strings; null means currently active, or an unknown window. */
  startsAt?: string | null;
  endsAt?: string | null;
  /** Provenance, for a future "data as of ..." note. */
  source?: string | null;
}

export interface IncidentQuery {
  /** Current viewport, so a real source can filter by bounding box. */
  bounds?: { north: number; south: number; east: number; west: number } | null;
  /** The planned departure time, matching planRoute's departAt. */
  at?: Date | null;
  kinds?: IncidentKind[];
}

/** The one thing a real data source has to implement. */
export interface IncidentSource {
  list(query: IncidentQuery): Promise<MapIncident[]>;
}
