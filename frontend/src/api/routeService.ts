import apiClient from './client';

export type CongestionLevel = 'clear' | 'moderate' | 'heavy';

export interface RouteOption {
  label: string;
  via: string;
  duration_min: number;
  distance_km: number;
  congestion_level: CongestionLevel;
  event_note?: string | null;
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

interface PlanRouteResult {
  routes: RouteOption[];
}

/**
 * POST /routes/plan — returns the top 3 route options (best / least traffic /
 * shortest) for an origin and one or more destinations. Pass real coordinates
 * (GPS for origin, search results for destinations) whenever available —
 * they take priority over text on the backend, which otherwise falls back to
 * a placeholder KNOWN_PLACES dictionary.
 */
export async function planRoute(
  origin: string,
  destinations: DestinationInput[],
  optimizeStopOrder: boolean = true,
  originCoords?: Coordinates | null
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
    });
    return data.routes;
  } catch (error: any) {
    if (error.response) {
      throw new Error(error.response.data?.message || 'Could not compute a route.');
    }
    if (error.request) {
      throw new Error('Could not reach the server. Check your connection and try again.');
    }
    throw error;
  }
}
