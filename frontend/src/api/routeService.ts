export interface StudyNode {
  id: string;
  city: string;
  latitude: number;
  longitude: number;
  label: string;
}

export interface HistoricalStudyStatus {
  ready: boolean;
  contract_version?: number;
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
  serving_mode?: 'live' | 'historical';
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
  /** Live mode: the saved HERE/weather capture (Asia/Manila ISO time) that scores every road. */
  live_at?: string | null;
}

export interface LiveStatus {
  mode: 'live';
  ready: boolean;
  message: string;
  captures: number;
  first_capture: string | null;
  latest_capture: string | null;
  release: { status: string; calibration_id: string; valid_hours: number } | null;
  latest?: {
    weather_at?: string; live_matched?: number; segments?: number; closed?: number; flood_reported?: number;
    lane_blocking?: number; weather?: Record<string, string>; city_load?: Record<string, number>; unavailable?: string;
  };
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

/** Search outcome of one ACO run; a time-limited run still returns its best route so far. */
export type StopReason = 'converged' | 'iteration_limit' | 'time_limit';

/** Everything needed to reproduce and explain one selected route (contract version 2). */
export interface Decision {
  search_id: string; algorithm: Algorithm; mode: string; seed: number; origin: string; destination: string;
  road_ids: string[]; snapshot_id: string; conditions: Conditions; calibration_id: string; graph_built_at: string;
  contract_version: number; native_objective: string; native_cost: number; common_evaluation_cost: number | null;
  stop_reason: StopReason;
}

export interface RouteResult {
  route: { node_ids: string[]; road_ids: string[]; geometry: [number, number][]; road_names: string[]; feature_sources: Record<string, Record<string, string>> };
  conditions: Conditions;
  weather_observation: WeatherSnapshot | null;
  decision: Decision;
  evaluation: {
    distance_travel_m: number;
    path_cost: number;
    search_time_ms: number;
    response_time_ms: number;
    convergence_rule: {
      converged: boolean;
      iterations: number;
      max_iterations: number;
      patience: number;
      stop_reason: StopReason;
      iterations_to_convergence: number | null;
      first_best_found_iteration: number;
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
  snapshot_id: string;
  affected_roads: number;
  iterations: number;
  iterations_to_convergence: number | null;  // N_conv; null when the search did not converge.
  best_found_at: number;
  stop_reason: StopReason | 'search_exhausted' | 'no_feasible_route';
  route_cost: number | null;  // Candidate cost in the algorithm's native objective.
  incumbent_cost: number | null;  // The current route's remainder under the same objective; null when invalid.
  decision: 'initial' | 'adopted_incumbent_invalid' | 'adopted_better' | 'kept_incumbent' | 'kept_incumbent_search_failed';
  search_time_ms: number;
  routing_time_ms: number;  // Recalculation: snapshot rebuild, rescoring and search.
  best_cost_by_iteration: (number | null)[];
}

/** One criterion's part of a route's path cost; contributions across the ten criteria sum to the path cost. */
export interface FeatureContribution {
  feature: string; weight: number; contribution: number; share: number;
  /** The route's additive criterion total in `unit`; mean_severity is its length-weighted mean (per km) where defined. */
  route_value: number; unit: string; mean_severity: number | null; development_max: number;
  classification: string; basis: string; reference: string; data_sources: Record<string, number>;
}

export interface TripResult {
  algorithm: { name: Algorithm; description: string };
  decision: Decision;
  contributions: FeatureContribution[];
  route: { node_ids: string[]; road_ids: string[]; geometry: [number, number][]; road_names: string[] };
  trip: { departure: string; travel_time_s: number; stops: string[]; stop_arrival_minutes: number[];
    changes_without_replanning: { at_minute: number; change: string }[] };
  evaluation: {
    distance_travel_m: number;
    path_cost: number;  // Common evaluation cost: the ten-criterion cost of each road when entered.
    search_time_ms: number;  // ACO search only, all searches of the trip.
    initial_search_time_ms: number;
    response_time_ms: number;  // Service time of the whole request.
    recalculation_time_ms: number;  // Re-routing: snapshot rebuild, rescoring and search.
    computational_time_ms: number;
    initial_routing_time_ms: number;
    dynamic_rerouting_time_ms: number;
    detection_delay_s: number[];  // Simulated time from each event to its detection at a node.
    reroutes: number;
    routes_adopted: number;
    mean_rerouting_time_ms: number | null;
    convergence: { iterations_to_convergence: number | null; converged: boolean; stop_reason: TripSearch['stop_reason'];
      first_best_found_iteration: number; mean_iterations_to_convergence: number | null;
      searches_not_converged: number; searches: TripSearch[] };
  };
}

// One primary outcome in scripts/evaluate_sop.py: d = baseline − ANTROUTE over request medians, Wilcoxon, Holm (m = 2).
export interface SopTest {
  n: number; test: string; proposed_median: number | null; baseline_median: number | null; mean_difference: number | null;
  median_difference: number | null; n_nonzero?: number; W_plus?: number; W_minus?: number; W?: number;
  p: number | null; p_holm: number | null; significant: boolean | null;
}
export interface SopResults {
  generated_at: string; scipy_version: string; alpha: number;
  manifest: { frozen_at: string; calibration_id: string; contract_version: number };
  primary_measures: Record<string, { label: string; unit: string; precision: number }>;
  analysis: {
    requests: number; complete_requests: number; incomplete_requests: string[]; seeds: number[];
    outcomes: Record<Algorithm, Record<string, number>>; primary: Record<string, SopTest>;
  };
}

const baseURL =(import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

/** A failed search or request, with the server's outcome (no_feasible_route, search_time_limit, search_exhausted, data_unavailable, ...). */
export class RouteError extends Error {
  constructor(message: string, public outcome: string) { super(message); }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseURL}${path}`, init);
  const body = await response.json();
  if (!response.ok) {
    if (typeof body.detail === 'string') throw new RouteError(body.detail, response.status === 503 ? 'data_unavailable' : 'invalid_request');
    if (body.detail && typeof body.detail.outcome === 'string') throw new RouteError(body.detail.message, body.detail.outcome);
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
export const loadLiveStatus = (signal?: AbortSignal) => request<LiveStatus>('/conditions/live', { signal });
export const loadSopResults = (signal: AbortSignal) => request<SopResults>('/study/results', { signal });
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
  link.href = url; link.download = conditions.live_at
    ? `antroute_live_${conditions.live_at.slice(0, 16).replace(':', '')}.csv`
    : `antroute_${conditions.departure_date}_${conditions.time_of_day}.csv`;
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
  travel_time_s: number; max_flood_severity: number; road_names: string[];
  /** Length-weighted means over the stretch (capacity: length / total capacity burden). */
  flood_severity: number; congestion: number; capacity_vph: number; weather_severity: number; time_profile_severity: number;
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
  chosen: StretchFacts; ant_routes: number; decision: Decision;
  references: Partial<Record<RouteStretch['reference'], StretchFacts & { same_as_chosen: boolean }>>;
  stretches: RouteStretch[];
}

/** Explains the selected ANTROUTE decision on its own snapshot (no new search); reference routes are for comparison only. */
export const explainRoute = (searchId: string, signal: AbortSignal) =>
  request<RouteExplanation>('/routes/explain', {
    method: 'POST', signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ search_id: searchId }),
  });

export const planTrip = (algorithm: Algorithm, origin: string, destinations: string[], conditions: Conditions,
  departureMinute: number, events: TripEvent[], signal: AbortSignal) =>
  request<TripResult>('/routes/trip', {
    method: 'POST', signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ algorithm, origin, destinations, conditions, departure_minute: departureMinute, events }),
  });
