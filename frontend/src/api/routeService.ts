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

export interface RouteOption {
  label: string;
  via: string;
  duration_min: number;
  distance_km: number;
  congestion_level: CongestionLevel;
  event_note?: string | null;
  path?: Coordinates[];
  algorithm?: RouteAlgorithm | null;
  /** Why a baseline route is not IACO's own, when it is not. */
  fallback_reason?: string | null;
  /** Mean predicted congestion risk (0-1) along the route, measured the same way for both models. */
  mean_risk?: number | null;
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

/**
 * Evaluation results. `source` says what is compared: 'routing' is ANTRoute against the
 * baseline Improved ACO (scripts/evaluate_routing.py); 'forecast' is the congestion model
 * against a naive forecaster, shown until the routing evaluation has been run.
 */
export interface ComparisonMetrics {
  source: 'routing' | 'forecast';
  title: string;
  baselineName: string;
  description: string;
  metrics: ComparisonMetricRow[];
}

export interface RoutePlan {
  routes: RouteOption[];
  /** Which recorded traffic the routes were planned on, or that none is loaded. */
  trafficNote: string;
  /** Why `routes` is empty, when the server says so (e.g. no baseline route for this trip). */
  notice: string | null;
}

interface PlanRouteResult {
  routes: (Omit<RouteOption, 'path'> & { path?: { lat: number; lng: number }[] })[];
  departure_time: string;
  traffic_note?: string;
  notice?: string | null;
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
  departAt: Date | null = null
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
    });
    return {
      routes: data.routes.map((r) => ({
        ...r,
        path: (r.path ?? []).map((p) => ({ latitude: p.lat, longitude: p.lng })),
      })),
      trafficNote: data.traffic_note ?? '',
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

/** GET /routes/comparison-metrics — ANTRoute vs. baseline evaluation results. */
export async function getComparisonMetrics(): Promise<ComparisonMetrics> {
  try {
    const { data } = await apiClient.get<ComparisonMetricsResponse>('/routes/comparison-metrics');
    return {
      source: data.source,
      title: data.title,
      baselineName: data.baseline_name,
      description: data.description,
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
      // 404 carries the server's reason, e.g. that the evaluation hasn't been run yet.
      throw new Error(error.response.data?.detail || 'Could not load comparison metrics.');
    }
    if (error.request) {
      throw new Error('Could not reach the server. Check your connection and try again.');
    }
    throw new Error('Could not load comparison metrics.');
  }
}