import apiClient from './client';
import { waitForRouteJob } from './routeJobs';

export type CongestionLevel = 'clear' | 'moderate' | 'heavy';
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

export interface RoutePlan {
  routes: RouteOption[];
  /** Why `routes` is empty, when the server says so (e.g. no baseline route for this trip). */
  notice: string | null;
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
  improvementPct: number;
  higherIsBetter: boolean;
}

/**
 * ANTRoute's congestion forecast against a naive forecaster (`baselineName`). This is
 * forecast accuracy, not a routing comparison: the routing baseline is Improved ACO, and
 * its routes come from planRoute(..., 'baseline').
 */
export interface ComparisonMetrics {
  routeOptimalityPct: { antroute: number; baseline: number };
  metrics: ComparisonMetricRow[];
  baselineName: string;
}

interface PlanRouteResult {
  routes: (Omit<RouteOption, 'path'> & { path?: { lat: number; lng: number }[] })[];
  notice?: string | null;
}

interface ComparisonMetricsResponse {
  route_optimality_pct: { antroute: number; baseline: number };
  baseline_name?: string;
  metrics: {
    metric: string;
    antroute: string;
    baseline: string;
    improvement_pct: number;
    higher_is_better: boolean;
  }[];
}

/**
 * POST /routes/jobs + status polling — returns route options (best / least traffic /
 * shortest) for an origin and one or more destinations. Pass real coordinates
 * (GPS for origin, search results for destinations) whenever available —
 * they take priority over text on the backend, which otherwise falls back to
 * a placeholder KNOWN_PLACES dictionary.
 *
 * `model` picks which model computes the routes (ANTRoute or the baseline). The baseline
 * may legitimately return no routes, with `notice` saying why, rather than an error.
 */
export async function planRoute(
  origin: string,
  destinations: DestinationInput[],
  optimizeStopOrder: boolean = true,
  originCoords?: Coordinates | null,
  model: RouteModel = 'antroute',
  signal?: AbortSignal
): Promise<RoutePlan> {
  try {
    const data = await waitForRouteJob<PlanRouteResult>(apiClient, {
      origin,
      origin_lat: originCoords?.latitude ?? null,
      origin_lng: originCoords?.longitude ?? null,
      destinations: destinations.map((d) => ({
        name: d.name,
        lat: d.lat ?? null,
        lng: d.lng ?? null,
      })),
      optimize_stop_order: optimizeStopOrder,
      model,
    }, { signal });
    return {
      routes: data.routes.map((r) => ({
        ...r,
        path: (r.path ?? []).map((p) => ({ latitude: p.lat, longitude: p.lng })),
      })),
      notice: data.notice ?? null,
    };
  } catch (error: any) {
    if (signal?.aborted || error?.name === 'AbortError' || error?.code === 'ERR_CANCELED') {
      throw error;
    }
    if (error.response) {
      throw new Error(
        error.response.data?.detail || error.response.data?.message || 'Could not compute a route.'
      );
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
      routeOptimalityPct: data.route_optimality_pct,
      baselineName: data.baseline_name ?? "Always 'Medium'",
      metrics: data.metrics.map((m) => ({
        metric: m.metric,
        antroute: m.antroute,
        baseline: m.baseline,
        improvementPct: m.improvement_pct,
        higherIsBetter: m.higher_is_better,
      })),
    };
  } catch (error: any) {
    if (error.request && !error.response) {
      throw new Error('Could not reach the server. Check your connection and try again.');
    }
    throw new Error('Could not load comparison metrics.');
  }
}
