import {
  AlertTriangle,
  Camera,
  CircleX,
  CloudRain,
  Gauge,
  Waves,
  type LucideIcon,
} from 'lucide-react';

import { ANT_COLOR, BASELINE_COLOR, CONGESTION_COLORS } from '@/lib/theme';
import type { IncidentKind, TrafficBand } from '@/types/markers';

/**
 * One description of every legend row, read by BOTH the legend panel and the marker layer,
 * so the legend can never advertise a glyph or colour the map does not actually draw.
 */

export interface LegendEntry {
  label: string;
  color: string;
  icon: LucideIcon;
}

export const INCIDENT_LEGEND: Record<IncidentKind, LegendEntry> = {
  accident: { label: 'Accident / Event', color: '#ef4444', icon: AlertTriangle },
  road_closure: { label: 'Road Closure', color: '#dc2626', icon: CircleX },
  // A segment the uploaded data records in the heavy band, as opposed to the route-colour
  // bands below, which read live traffic.
  traffic: { label: 'Heavy Traffic Segment', color: CONGESTION_COLORS.heavy, icon: Gauge },
  weather_rain: { label: 'Weather (Rain)', color: '#0ea5e9', icon: CloudRain },
  flood: { label: 'Flood', color: '#2563eb', icon: Waves },
  camera: { label: 'Camera (CCTV)', color: '#1d4ed8', icon: Camera },
};

export const TRAFFIC_LEGEND: Record<TrafficBand, { label: string; color: string }> = {
  heavy: { label: 'Heavy Traffic', color: CONGESTION_COLORS.heavy },
  moderate: { label: 'Moderate Traffic', color: CONGESTION_COLORS.moderate },
  light: { label: 'Light Traffic', color: CONGESTION_COLORS.clear },
};

export interface RouteLegendEntry {
  label: string;
  color: string;
  dashed: boolean;
}

export const ROUTE_LEGEND: RouteLegendEntry[] = [
  { label: 'Recommended Route', color: ANT_COLOR, dashed: false },
  { label: 'Alternative Route', color: '#8b5cf6', dashed: true },
  { label: 'Baseline Route', color: BASELINE_COLOR, dashed: false },
];

/** Colour of an alternative (non-recommended) ANTRoute option, by its index. */
export const ALTERNATIVE_COLORS = ['#8b5cf6', '#a855f7', '#c084fc'];

export function alternativeColor(index: number): string {
  return ALTERNATIVE_COLORS[(index - 1) % ALTERNATIVE_COLORS.length];
}
