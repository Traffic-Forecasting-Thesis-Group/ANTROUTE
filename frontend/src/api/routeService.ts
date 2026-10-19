export interface StudyNode {
  id: string;
  city: string;
  latitude: number;
  longitude: number;
  label: string;
}

export interface HistoricalStudyStatus {
  ready: boolean;
  required_hours_per_entity: number;
  message: string;
  requirements: { display_period: string; dynamic_resolution: string; allow_estimates: boolean; timezone: string };
  audit: { feature_audit: Record<string, { status: string; local_evidence: string; required_evidence: string }> } | null;
}

export interface BoundaryGeometry {
  type: 'Polygon' | 'MultiPolygon';
  coordinates: number[][][] | number[][][][];
}

export interface WeatherSnapshot {
  source: string;
  stale: boolean;
  age_seconds: number;
  cities: Record<string, { localtime: string; description: string[]; precip_mm: number; condition: string; retrieved_at: string }>;
}

export interface RoadSelection {
  id: string;
  name: string;
  geometry: [number, number][];
  opposite_roads?: RoadSelection[];
}

export interface NetworkInfo {
  cities: string[];
  nodes: StudyNode[];
  weather_conditions: string[];
  road_ids: string[];
  boundaries: Record<string, BoundaryGeometry>;
  node_count: number;
  road_count: number;
  data_quality: { mode: string; limitations: string[]; built_at: string; estimation_registry: Record<string, { classification: string; basis: string; source: string }> };
}

export interface Conditions {
  departure_date: string;
  time_of_day: number;
  weather: string;
  closed_road_ids: string[];
}

export interface DatasetInfo {
  nodes: number; directed_segments: number; segments_by_city: Record<string, number>; traffic_locations: number;
  largest_strongly_connected_nodes: number; strongly_connected_components: number;
  hours: number; potential_segment_hour_rows: number;
  retained_weather_grids: number; assigned_weather_grids: number; note: string;
}

export interface DatasetMap {
  bands: { label: string; color: string; lines: [number, number][][] }[];
  segment_count: number;
}

export interface RouteResult {
  route: { node_ids: string[]; road_ids: string[]; geometry: [number, number][]; road_names: string[]; feature_sources: Record<string, Record<string, string>> };
  conditions: Conditions;
  weather_observation: WeatherSnapshot | null;
  evaluation: {
    distance_travel_m: number;
    path_cost: number;
    computational_time_ms: number;
    convergence_rule: {
      converged: boolean;
      iterations: number;
      max_iterations: number;
      tolerance: number;
      patience: number;
      stop_reason: string;
      best_cost_by_iteration: (number | null)[];
    };
  };
}

export type Algorithm = 'antroute' | 'cheng';

export interface TripEvent {
  at_minute: number;
  close_road_ids?: string[];
  weather?: string;
}

export interface TripSearch {
  kind: 'initial' | 'reroute';
  at_node: string;
  at_minute: number;
  trigger: string;
  affected_roads: number;
  iterations: number;
  iterations_to_convergence: number;
  stop_reason: string;
  route_cost: number;
  routing_time_ms: number;
  best_cost_by_iteration: (number | null)[];
}

/** One feature's part of a route's path cost; contributions across the ten features sum to the path cost. */
export interface FeatureContribution {
  feature: string; weight: number; contribution: number; share: number; normalized_mean: number;
  route_value: number | number[] | { mean: number; max: number } | Record<string, number>; unit: string;
  classification: string; basis: string; reference: string; data_sources: Record<string, number>;
}

export interface TripResult {
  algorithm: { name: Algorithm; description: string };
  contributions: FeatureContribution[];
  route: { node_ids: string[]; road_ids: string[]; geometry: [number, number][]; road_names: string[] };
  trip: { departure: string; travel_time_s: number; stops: string[]; stop_arrival_minutes: number[];
    changes_without_replanning: { at_minute: number; change: string }[] };
  evaluation: {
    distance_travel_m: number;
    path_cost: number;
    computational_time_ms: number;
    initial_routing_time_ms: number;
    dynamic_rerouting_time_ms: number;
    reroutes: number;
    mean_rerouting_time_ms: number | null;
    convergence: { iterations_to_convergence: number; mean_iterations_to_convergence: number; searches: TripSearch[] };
  };
}


const baseURL =(import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseURL}${path}`, init);
  const body = await response.json();
  if (!response.ok) {
    if (typeof body.detail === 'string') throw new Error(body.detail);
    // Request-format errors list each problem; show them instead of a generic message.
    const problems = Array.isArray(body.detail)
      ? body.detail.map((item: { loc?: (string | number)[]; msg?: string }) => `${item.loc?.slice(1).join('.') || 'request'}: ${item.msg}`).join('; ')
      : '';
    throw new Error(`Invalid route inputs${problems ? ` (${problems})` : ''}. If the backend was started before the app was updated, restart it.`);
  }
  return body as T;
}

export const loadNetwork = (signal: AbortSignal) => request<NetworkInfo>('/network?include_nodes=false', { signal });
export const loadStudyStatus = (signal: AbortSignal) => request<HistoricalStudyStatus>('/study/status', { signal });
export const searchPlaces = (query: string, signal: AbortSignal) => request<{ results: StudyNode[] }>(`/places/search?query=${encodeURIComponent(query)}`, { signal });
export const snapNode = (latitude: number, longitude: number) => request<StudyNode>(`/places/nearest?latitude=${latitude}&longitude=${longitude}`);
export const snapRoad = (latitude: number, longitude: number) => request<RoadSelection>(`/roads/nearest?latitude=${latitude}&longitude=${longitude}`);
export const loadWeather = (departureDate: string, hour: number, signal?: AbortSignal) => request<WeatherSnapshot>(`/conditions/weather?departure_date=${departureDate}&time_of_day=${hour}`, { signal });
export const refreshWeather = () => request<WeatherSnapshot>('/conditions/weather/refresh', { method: 'POST' });
export const loadDatasetInfo = (signal: AbortSignal) => request<DatasetInfo>('/dataset/info', { signal });
/** Every road segment included in the study, grouped by city; it does not depend on date, hour or weather. */
export const loadDatasetMap = (city: string | null, signal: AbortSignal) =>
  request<DatasetMap>('/dataset/map', { method: 'POST', signal, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ city }) });
export async function downloadSnapshot(conditions: Conditions, city: string | null, signal: AbortSignal) {
  const response = await fetch(`${baseURL}/dataset/export`, { method: 'POST', signal,
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ conditions, city }) });
  if (!response.ok) throw new Error('Could not export the snapshot. Check the study date and settings.');
  const blob = await response.blob();
  if (signal.aborted) return;
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url; link.download = `antroute_${conditions.departure_date}_${conditions.time_of_day}.csv`;
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
// ACO parameters are fixed by the study; the server's defaults always apply.
export const planRoute = (origin: string, destination: string, conditions: Conditions, signal: AbortSignal) =>
  request<RouteResult>('/routes/plan', {
    method: 'POST', signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ origin, destination, conditions }),
  });
export interface StretchFacts {
  distance_m: number; cost: number; roads: number; intersections: number; signals: number;
  travel_time_s: number; max_flood_risk: number; road_names: string[];
  /** Length-weighted averages over the stretch. */
  flood_risk: number; congestion: number; capacity_vph: number; weather_cost: number; time_of_day_load: number;
  road_types: string[];
}

/** One place where a reference route leaves ANTROUTE's route and rejoins it. */
export interface RouteStretch {
  reference: 'shortest_distance' | 'lowest_cost';
  verdict: 'costlier_alternative' | 'missed_by_search';
  chosen: StretchFacts & { geometry: [number, number][] };
  alternative: StretchFacts & { geometry: [number, number][] };
  marker: [number, number];
  distance_saved_m: number; cost_difference: number;
  feature_differences: Record<string, number>;
  ant_routes_using_alternative: number; ant_routes_using_first_road: number;
}

export interface RouteExplanation {
  chosen: StretchFacts; ant_routes: number;
  references: Partial<Record<RouteStretch['reference'], StretchFacts & { same_as_chosen: boolean }>>;
  stretches: RouteStretch[];
}

/** Explains ANTROUTE's departure plan for one leg; reference routes are for comparison only. */
export const explainRoute = (origin: string, destination: string, conditions: Conditions, signal: AbortSignal) =>
  request<RouteExplanation>('/routes/explain', {
    method: 'POST', signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ origin, destination, conditions }),
  });

export const planTrip = (algorithm: Algorithm, origin: string, destinations: string[], conditions: Conditions,
  departureMinute: number, events: TripEvent[], signal: AbortSignal) =>
  request<TripResult>('/routes/trip', {
    method: 'POST', signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ algorithm, origin, destinations, conditions, departure_minute: departureMinute, events }),
  });
