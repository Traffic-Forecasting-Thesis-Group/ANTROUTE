import apiClient from './client';

// 'unknown' when the server has no congestion data loaded.
export type CongestionLevel = 'clear' | 'moderate' | 'heavy' | 'unknown';
export type RouteModel = 'antroute' | 'baseline';

/**
 * Which algorithm produced a route. The baseline is Improved ACO (Cheng 2023): 'iaco' when
 * it found the route itself, 'shortest_distance' when it found none and the paper's
 * shortest-distance comparator stood in, 'mixed' when a multi-stop trip needed both.
 * Absent on ANTRoute's placeholder routes, which no algorithm produced.
 */
export type RouteAlgorithm = 'antroute' | 'iaco' | 'shortest_distance' | 'mixed';

/** Caption for every travel time the app shows: ETA is not a thesis metric and was never scored. */
export const DURATION_NOTE = 'estimated, not evaluated';

export interface RouteOption {
  label: string;
  via: string;
  /** The ETA engine's travel time -- estimated, not evaluated (see duration_note). */
  duration_min: number;
  duration_note?: string;
  distance_km: number;
  congestion_level: CongestionLevel;
  /** "Estimated from weather, road type and traffic reports" when the level rests mostly on roads far from any camera. */
  congestion_note?: string | null;
  event_note?: string | null;
  path?: Coordinates[];
  algorithm?: RouteAlgorithm | null;
  /** Why a baseline route is not IACO's own, when it is not. */
  fallback_reason?: string | null;
  /** Mean predicted congestion risk (0-1) along the route, measured the same way for both models. */
  mean_risk?: number | null;
  /** Share of the route's length in the observed or propagated risk tier; below 0.5 congestion_level is 'unknown'. */
  risk_coverage?: number | null;
  /**
   * Share of the route's length per risk tier: observed (camera subgraph, model prediction),
   * propagated (within 1.5 km of it), assumed (farther: weather, flood, incidents, clock, road class).
   */
  risk_sources?: { observed: number; propagated: number; assumed: number } | null;
  /** The recorded inputs along this route: cameras, incidents, weather, flood, vehicle counts. */
  inputs?: RouteInputs | null;
}

export interface CameraReading {
  camera: string;
  node_id: number;
  lat: number;
  lon: number;
  /** The model's mean risk on the camera's edges, 0-1. */
  model_risk: number | null;
  /** The human congestion label at the camera in that window. */
  human_label: 'Light' | 'Medium' | 'Heavy' | null;
  /** Mean YOLO vehicle count over the window. */
  vehicle_count: number | null;
  /** Metres from the route, on a route's inputs. */
  distance_m?: number | null;
}

export interface IncidentReading {
  location: string;
  intersection: string;
  kind: string;
  lanes: number;
  /** Base impact times linear decay, 0-1. */
  strength: number;
  reported_at: string;
  minutes_ago: number | null;
  lat: number | null;
  lon: number | null;
  distance_m?: number | null;
}

export interface WeatherReading {
  available: boolean;
  where?: string | null;
  hour?: string | null;
  temp_c?: number | null;
  precip_mm?: number | null;
  precip_mm_max?: number | null;
  humidity_pct?: number | null;
  source?: string | null;
  note?: string | null;
}

/**
 * The recorded inputs along one route, for comparing alternatives. "Recorded": read from one
 * window of recorded data, not a live feed.
 */
export interface RouteInputs {
  recorded: boolean;
  window: string;
  cameras: CameraReading[];
  incidents: IncidentReading[];
  weather: WeatherReading;
  flood: { max_level?: number; share_of_route_flood_prone?: number };
  vehicle_count_mean: number | null;
}

export interface WindowInfo {
  window: string;
  window_end: string | null;
  day: string;
  session: 'AM' | 'PM';
  label: string;
  split: string;
  /** True for the test split (May 25): the model never saw it in training or validation. */
  held_out: boolean;
  split_note: string;
}

/** The "Inputs used" panel: every recorded input of the window the routes were planned on. */
export interface RecordedInputs {
  recorded: boolean;
  /** 'uploaded' when the inputs come from the user's CSV rather than the server's recorded data. */
  source?: 'recorded' | 'uploaded';
  source_name?: string;
  matched_window: WindowInfo;
  cameras: CameraReading[];
  incidents: IncidentReading[];
  weather: WeatherReading;
  flood: { mean_level?: number | null; share_of_roads_flood_prone?: number | null; source?: string };
  vehicle_counts_available: boolean;
}

export interface RecordedSession {
  day: string;
  session: 'AM' | 'PM';
  label: string;
  split: string;
  held_out: boolean;
  note: string;
  labelled_edges: number;
  windows: { start: string; end: string | null }[];
}

export interface WhatIfCase {
  key: string;
  label: string;
  description: string;
  camera_mean: number;
  camera_max: number;
  other_mean: number;
  other_max: number;
  other_moved: number;
}

export interface WhatIf {
  model: string;
  windows: number;
  note: string;
  cases: WhatIfCase[];
  route_effect: {
    n_trips: number | null;
    lanes: number | null;
    active_minutes: number | null;
    systems: Record<
      string,
      {
        routes_changed_share: number;
        event_exposure_old: number;
        event_exposure_recalculated: number;
        congestion_exposure_old: number;
        congestion_exposure_recalculated: number;
      }
    >;
  } | null;
}

export interface Coordinates {
  latitude: number;
  longitude: number;
}

export interface DestinationInput {
  name: string;
  lat?: number | null;
  lng?: number | null;
}

export interface ComparisonMetricRow {
  metric: string;
  antroute: string;
  baseline: string;
  /** Relative difference, positive when ANTRoute is better; null when undefined. */
  improvementPct: number | null;
  higherIsBetter: boolean;
  /** Wilcoxon signed-rank p-value; routing evaluation only. */
  pValue: number | null;
  significant: boolean | null;
}

/** Mean Route Optimality (%) per system over the evaluated test trips (thesis Equation 1). */
export interface RouteOptimalitySummary {
  antroute: number;
  baseline: number;
  nTrials: number;
}

/**
 * The thesis routing evaluation (Section 3.9, Appendix 3): ANTRoute against the baseline
 * Improved ACO on Route Optimality (both routes timed in Google Maps), congestion exposure and
 * event exposure, with route length as a descriptive extra, by scripts/evaluate_routing.py.
 * Δ% follows Equations 7-8 (positive = ANTRoute better). ETA is not evaluated.
 */
export interface ComparisonMetrics {
  source: 'routing' | 'forecast';
  title: string;
  baselineName: string;
  description: string;
  metrics: ComparisonMetricRow[];
  routeOptimality: RouteOptimalitySummary | null;
  /** The same table per scenario: "overall" (every trip) first, then each scenario. */
  scenarios: ScenarioEvaluation[];
}

export interface ScenarioEvaluation {
  scenario: string;
  label: string;
  nTrials: number;
  metrics: ComparisonMetricRow[];
}

/**
 * The routing evaluation of one planned trip. Only the test trips were timed in Google Maps,
 * so 'evaluated' carries that trip's own Route Optimality and exposures; 'other_time' means
 * the same stops were evaluated at another departure time; 'not_evaluated' means no test trip
 * has these stops. `message` says which, in words for the user.
 */
export interface TripEvaluation {
  status: 'evaluated' | 'other_time' | 'not_evaluated';
  message: string;
  tripId: string | null;
  baselineName: string;
  routeOptimality: { antroute: number; baseline: number } | null;
  metrics: ComparisonMetricRow[];
}

interface TripEvaluationResponse {
  status: 'evaluated' | 'other_time' | 'not_evaluated';
  message: string;
  trip_id?: string | null;
  baseline_name: string;
  route_optimality?: { antroute: number; baseline: number } | null;
  metrics: ComparisonMetricsResponse['metrics'];
}

export interface RoutePlan {
  routes: RouteOption[];
  /** Which recorded traffic the routes were planned on, or that none is loaded. */
  trafficNote: string;
  /** The recorded inputs of the window the routes were planned on. */
  recordedInputs: RecordedInputs | null;
  /** True when the user chose the recorded day and session (simulation mode). */
  simulation: boolean;
  /** Why `routes` is empty, when the server says so (e.g. no baseline route for this trip). */
  notice: string | null;
}

interface PlanRouteResult {
  routes: (Omit<RouteOption, 'path'> & { path?: { lat: number; lng: number }[] })[];
  departure_time: string;
  traffic_note?: string;
  notice?: string | null;
  recorded_inputs?: RecordedInputs | null;
  simulation?: boolean;
}

interface ComparisonMetricsResponse {
  source: 'routing' | 'forecast';
  title: string;
  baseline_name: string;
  description: string;
  metrics: {
    metric: string;
    antroute: string;
    baseline: string;
    improvement_pct: number | null;
    higher_is_better: boolean;
    p_value?: number | null;
    significant?: boolean | null;
  }[];
  route_optimality?: { antroute: number; baseline: number; n_trials: number } | null;
  scenarios?: { scenario: string; label: string; n_trials: number; metrics: MetricRowResponse[] }[];
}

type MetricRowResponse = ComparisonMetricsResponse['metrics'][number];

function toMetricRow(m: MetricRowResponse): ComparisonMetricRow {
  return {
    metric: m.metric,
    antroute: m.antroute,
    baseline: m.baseline,
    improvementPct: m.improvement_pct,
    higherIsBetter: m.higher_is_better,
    pValue: m.p_value ?? null,
    significant: m.significant ?? null,
  };
}

/**
 * POST /routes/plan — returns up to 3 route options (best / least traffic /
 * shortest) for an origin and one or more destinations, in the order given.
 * Pass real coordinates (GPS for origin, search results for destinations)
 * whenever available; text without them is geocoded on the backend, and a
 * place it can't find comes back as an error rather than a guessed route.
 *
 * `model` picks which model computes the routes (ANTRoute or the baseline). The baseline
 * may legitimately return no routes, with `notice` saying why, rather than an error.
 * `departAt` is when the driver leaves; null means leaving now. The backend
 * routes both models on the traffic recorded at that time of day.
 */
export async function planRoute(
  origin: string,
  destinations: DestinationInput[],
  originCoords?: Coordinates | null,
  model: RouteModel = 'antroute',
  departAt: Date | null = null,
  recordedWindow: string | null = null
): Promise<RoutePlan> {
  try {
    const { data } = await apiClient.post<PlanRouteResult>('/routes/plan', {
      origin,
      origin_lat: originCoords?.latitude ?? null,
      origin_lng: originCoords?.longitude ?? null,
      destinations: destinations.map((d) => ({
        name: d.name,
        lat: d.lat ?? null,
        lng: d.lng ?? null,
      })),
      model,
      depart_at: departAt ? departAt.toISOString() : null,
      recorded_window: recordedWindow,
    });
    return {
      routes: data.routes.map((r) => ({
        ...r,
        path: (r.path ?? []).map((p) => ({ latitude: p.lat, longitude: p.lng })),
      })),
      trafficNote: data.traffic_note ?? '',
      recordedInputs: data.recorded_inputs ?? null,
      simulation: data.simulation ?? false,
      notice: data.notice ?? null,
    };
  } catch (error: any) {
    if (error.response) {
      throw new Error(
        error.response.data?.detail || error.response.data?.message || 'Could not compute a route.'
      );
    }
    if (error.code === 'ECONNABORTED') {
      throw new Error('The server took too long to plan this route. Please try again.');
    }
    if (error.request) {
      throw new Error('Could not reach the server. Check your connection and try again.');
    }
    throw error;
  }
}

function failure(error: any, fallback: string): Error {
  if (error.response) return new Error(error.response.data?.detail || fallback);
  if (error.request) return new Error('Could not reach the server. Check your connection and try again.');
  return new Error(fallback);
}

/** GET /routes/recorded-sessions — every recorded day and session, for simulation mode. */
export async function getRecordedSessions(): Promise<RecordedSession[]> {
  try {
    const { data } = await apiClient.get<{ sessions: RecordedSession[] }>('/routes/recorded-sessions');
    return data.sessions;
  } catch (error: any) {
    throw failure(error, 'Could not load the recorded sessions.');
  }
}

/** GET /routes/recorded-inputs — the recorded inputs of one window (a RecordedSession window start). */
export async function getRecordedInputs(window: string): Promise<RecordedInputs> {
  try {
    const { data } = await apiClient.get<RecordedInputs>('/routes/recorded-inputs', { params: { window } });
    return data;
  } catch (error: any) {
    throw failure(error, 'Could not load the recorded inputs.');
  }
}

/** GET /routes/what-if — precomputed cases: the checkpoint with rain, flood, incidents or cameras changed. */
export async function getWhatIf(): Promise<WhatIf> {
  try {
    const { data } = await apiClient.get<WhatIf>('/routes/what-if');
    return data;
  } catch (error: any) {
    throw failure(error, 'Could not load the what-if table.');
  }
}

/**
 * POST /routes/trip-evaluation — the evaluation of this trip (same stops and departure as
 * planRoute): its own metrics when it is one of the test trips timed in Google Maps.
 */
export async function getTripEvaluation(
  origin: string,
  destinations: DestinationInput[],
  originCoords?: Coordinates | null,
  departAt: Date | null = null
): Promise<TripEvaluation> {
  try {
    const { data } = await apiClient.post<TripEvaluationResponse>('/routes/trip-evaluation', {
      origin,
      origin_lat: originCoords?.latitude ?? null,
      origin_lng: originCoords?.longitude ?? null,
      destinations: destinations.map((d) => ({ name: d.name, lat: d.lat ?? null, lng: d.lng ?? null })),
      depart_at: departAt ? departAt.toISOString() : null,
    });
    return {
      status: data.status,
      message: data.message,
      tripId: data.trip_id ?? null,
      baselineName: data.baseline_name,
      routeOptimality: data.route_optimality
        ? { antroute: data.route_optimality.antroute, baseline: data.route_optimality.baseline }
        : null,
      metrics: data.metrics.map((m) => ({
        metric: m.metric,
        antroute: m.antroute,
        baseline: m.baseline,
        improvementPct: m.improvement_pct,
        higherIsBetter: m.higher_is_better,
        pValue: m.p_value ?? null,
        significant: m.significant ?? null,
      })),
    };
  } catch (error: any) {
    if (error.response) {
      throw new Error(error.response.data?.detail || 'Could not load this trip’s evaluation.');
    }
    if (error.request) {
      throw new Error('Could not reach the server. Check your connection and try again.');
    }
    throw new Error('Could not load this trip’s evaluation.');
  }
}

/** GET /routes/comparison-metrics — the thesis routing evaluation (never the CRS forecast scores). */
export async function getComparisonMetrics(): Promise<ComparisonMetrics> {
  try {
    const { data } = await apiClient.get<ComparisonMetricsResponse>('/routes/comparison-metrics');
    return {
      source: data.source,
      title: data.title,
      baselineName: data.baseline_name,
      description: data.description,
      metrics: data.metrics.map(toMetricRow),
      scenarios: (data.scenarios ?? []).map((sc) => ({
        scenario: sc.scenario,
        label: sc.label,
        nTrials: sc.n_trials,
        metrics: sc.metrics.map(toMetricRow),
      })),
      routeOptimality: data.route_optimality
        ? {
            antroute: data.route_optimality.antroute,
            baseline: data.route_optimality.baseline,
            nTrials: data.route_optimality.n_trials,
          }
        : null,
    };
  } catch (error: any) {
    if (error.response) {
      // 404 carries the server's reason, e.g. that the evaluation hasn't been run yet.
      throw new Error(error.response.data?.detail || 'Could not load comparison metrics.');
    }
    if (error.request) {
      throw new Error('Could not reach the server. Check your connection and try again.');
    }
    throw new Error('Could not load comparison metrics.');
  }
}