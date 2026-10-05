import apiClient from './client';

export type CongestionLevel = 'clear' | 'moderate' | 'heavy';
export type RouteModel = 'antroute' | 'baseline';

export interface RouteOption {
  label: string;
  via: string;
  duration_min: number;
  distance_km: number;
  congestion_level: CongestionLevel;
  event_note?: string | null;
  path?: Coordinates[];
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

export interface ComparisonMetrics {
  routeOptimalityPct: { antroute: number; baseline: number };
  metrics: ComparisonMetricRow[];
}

interface PlanRouteResult {
  routes: (Omit<RouteOption, 'path'> & { path?: { lat: number; lng: number }[] })[];
}

interface ComparisonMetricsResponse {
  route_optimality_pct: { antroute: number; baseline: number };
  metrics: {
    metric: string;
    antroute: string;
    baseline: string;
    improvement_pct: number;
    higher_is_better: boolean;
  }[];
}

/**
 * POST /routes/plan — returns the top 3 route options (best / least traffic /
 * shortest) for an origin and one or more destinations. Pass real coordinates
 * (GPS for origin, search results for destinations) whenever available —
 * they take priority over text on the backend, which otherwise falls back to
 * a placeholder KNOWN_PLACES dictionary.
 *
 * `model` picks which model computes the routes (ANTRoute or the baseline).
 */
export async function planRoute(
  origin: string,
  destinations: DestinationInput[],
  optimizeStopOrder: boolean = true,
  originCoords?: Coordinates | null,
  model: RouteModel = 'antroute'
): Promise<RouteOption[]> {
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
      optimize_stop_order: optimizeStopOrder,
      model,
    });
    return data.routes.map((r) => ({
      ...r,
      path: (r.path ?? []).map((p) => ({ latitude: p.lat, longitude: p.lng })),
    }));
  } catch (error: any) {
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