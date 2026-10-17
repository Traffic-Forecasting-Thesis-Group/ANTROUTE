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

const baseURL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseURL}${path}`, init);
  const body = await response.json();
  if (!response.ok) {
    throw new Error(typeof body.detail === 'string' ? body.detail : 'Invalid route inputs. Check the selected nodes and conditions.');
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
export const planRoute = (origin: string, destination: string, conditions: Conditions, signal: AbortSignal) =>
  request<RouteResult>('/routes/plan', {
    method: 'POST', signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ origin, destination, conditions }),
  });
