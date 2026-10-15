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

/**
 * Chart series colours (ANTRoute vs the Improved ACO baseline).
 *
 * Validated with the dataviz palette checker on a light surface: lightness band, chroma
 * floor, CVD separation (ΔE 31.0 protan / 29.1 tritan), normal-vision separation (35.7) and
 * ≥3:1 contrast all pass. The blue/purple pairing these mockups first used failed badly --
 * ΔE 4.5 for deuteranopia and 13.1 for normal vision, i.e. the two bars were effectively the
 * same colour for a red-green colourblind reader. The amber keeps the baseline's map identity
 * while being dark enough to read as a fill on white.
 */
export const CHART_ANT = '#3B6BE6';
export const CHART_BASELINE = '#D97706';

/** Chart furniture: hairline grid and ink for labels (text never wears the series colour). */
export const CHART_GRID = '#e2e8f0';
export const CHART_SURFACE = '#ffffff';
