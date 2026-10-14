import type { RouteOption } from '@/api/routeService';

/** Names the algorithm that actually produced a baseline route, which is not always IACO. */
export function baselineMethodLabel(route?: RouteOption): string {
  switch (route?.algorithm) {
    case 'shortest_distance':
      return 'Shortest distance (IACO found no route)';
    case 'mixed':
      return 'IACO + shortest distance';
    default:
      return 'Improved ACO (Cheng 2023)';
  }
}
