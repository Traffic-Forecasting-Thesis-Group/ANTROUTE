import type { CongestionLevel } from '@/api/routeService';

/**
 * Colour values that the imperative Google Maps layers need as plain strings (a
 * google.maps.Polyline takes strokeColor, not a CSS class). These mirror the tokens in
 * tailwind.config.js -- change both together.
 */

/** Route polyline colours. */
export const ANT_COLOR = '#3B6BE6';
export const BASELINE_COLOR = '#F7C441';

/**
 * Opacity of the part of the route already driven. The old React Native code appended a
 * hex alpha ('80') to the colour; the Maps JS API ignores alpha inside strokeColor, so
 * this is applied through strokeOpacity instead.
 */
export const TRAVELLED_OPACITY = 0.5;
export const ROUTE_OPACITY = 1;

export const CONGESTION_COLORS: Record<CongestionLevel, string> = {
  clear: '#10b981',
  moderate: '#f59e0b',
  heavy: '#ef4444',
  unknown: '#d1d5db',
};

/** How full the congestion bar reads for each band. */
export const CONGESTION_WIDTH: Record<CongestionLevel, string> = {
  clear: '30%',
  moderate: '60%',
  heavy: '85%',
  unknown: '0%',
};

export const CONGESTION_LABEL: Record<CongestionLevel, string> = {
  clear: 'Light traffic',
  moderate: 'Moderate traffic',
  heavy: 'Heavy traffic',
  unknown: 'Traffic unknown',
};
