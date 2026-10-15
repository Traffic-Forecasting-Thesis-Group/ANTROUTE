import type { RouteInputs, WeatherReading } from '@/api/routeService';

/**
 * Formatting for the recorded inputs (cameras, incidents, weather, flood) that the Route
 * Information modal shows. All of it describes recorded data read from one window, never a
 * live feed.
 */

export function weatherLine(w: WeatherReading | undefined | null): string {
  if (!w) return '–';
  if (!w.available) return w.note ?? 'No weather recorded.';
  const parts = [
    w.temp_c != null ? `${w.temp_c} °C` : null,
    w.precip_mm != null ? `${w.precip_mm} mm rain` : null,
    w.humidity_pct != null ? `${w.humidity_pct}% humidity` : null,
  ].filter(Boolean);
  return parts.length > 0 ? parts.join(', ') : 'Recorded';
}

/** A short weather word for a table cell: rain if any fell, else the temperature. */
export function weatherCell(w: WeatherReading | undefined | null): string {
  if (!w || !w.available) return '–';
  const mm = w.precip_mm;
  if (mm != null && mm > 0) {
    if (mm < 2.5) return 'Light rain';
    if (mm < 7.6) return 'Moderate rain';
    return 'Heavy rain';
  }
  return w.temp_c != null ? `${w.temp_c} °C` : 'No rain';
}

/** Whether any rain fell, for colouring the cell. */
export function isRainy(w: WeatherReading | undefined | null): boolean {
  return Boolean(w?.available && (w.precip_mm ?? 0) > 0);
}

/** The incidents on a stretch, as one cell: the kinds reported there. */
export function eventCell(inputs: RouteInputs | undefined | null): string | null {
  const incidents = inputs?.incidents ?? [];
  if (incidents.length === 0) return null;
  const kinds = Array.from(new Set(incidents.map((i) => i.kind).filter(Boolean)));
  if (kinds.length === 0) return `${incidents.length} incident(s)`;
  return kinds.join(', ');
}

