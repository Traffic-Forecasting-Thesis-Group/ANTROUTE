import type {
  CameraReading,
  IncidentReading,
  RecordedInputs,
  WindowInfo,
} from '@/api/routeService';

/**
 * The uploaded input-features CSV (traffic_system/scripts/build_input_features_csv.py): one row per
 * 30-minute window and camera intersection, with the inputs the system used then -- the cameras'
 * human labels and vehicle counts, incidents, hourly weather, flood level and the road text signals.
 *
 * The app reads it in the browser and shows it as the "Inputs used" panel and the map markers for
 * the window being planned. The routes themselves still come from the server's stored model scores
 * for that window; nothing in the file is sent to the server.
 */

export const REQUIRED_COLUMNS = ['window_start', 'camera', 'lat', 'lon'] as const;

export interface FeatureRow {
  window_start: string;
  window_end: string;
  day: string;
  session: 'AM' | 'PM';
  split: string;
  camera: string;
  lat: number;
  lon: number;
  human_label: 'Light' | 'Medium' | 'Heavy' | null;
  vehicle_count_mean: number | null;
  model_risk: number | null;
  temp_c: number | null;
  precip_mm: number | null;
  humidity_pct: number | null;
  flood_level: number | null;
  incident_count: number;
  incident_strength: number;
  incident_lanes: number;
  incident_places: string;
}

export interface UploadedInputs {
  fileName: string;
  rowCount: number;
  /** window_start -> rows of that window, in file order */
  windows: Record<string, FeatureRow[]>;
  windowStarts: string[];
  /** Columns beyond the required ones that this app does not read (the model's f_* vector, text signals). */
  extraColumns: string[];
}

/** RFC 4180 CSV: quoted fields, doubled quotes, CRLF, a leading byte-order mark. */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let quoted = false;
  const src = text.replace(/^﻿/, '');
  for (let i = 0; i < src.length; i++) {
    const c = src[i];
    if (quoted) {
      if (c === '"' && src[i + 1] === '"') {
        field += '"';
        i++;
      } else if (c === '"') {
        quoted = false;
      } else {
        field += c;
      }
    } else if (c === '"') {
      quoted = true;
    } else if (c === ',') {
      row.push(field);
      field = '';
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && src[i + 1] === '\n') i++;
      row.push(field);
      field = '';
      if (row.length > 1 || row[0] !== '') rows.push(row);
      row = [];
    } else {
      field += c;
    }
  }
  if (field !== '' || row.length > 0) {
    row.push(field);
    if (row.length > 1 || row[0] !== '') rows.push(row);
  }
  return rows;
}

const num = (v: string | undefined): number | null => {
  if (v === undefined || v.trim() === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};
const LABELS = new Set(['Light', 'Medium', 'Heavy']);

export class UploadError extends Error {}

/** Parses and validates the file; throws an UploadError whose message says what to fix. */
export function parseInputFeatures(fileName: string, text: string): UploadedInputs {
  const table = parseCsv(text);
  if (table.length < 2) throw new UploadError('The file has no data rows.');
  const header = table[0].map((h) => h.trim());
  const missing = REQUIRED_COLUMNS.filter((c) => !header.includes(c));
  if (missing.length > 0) {
    throw new UploadError(
      `Missing column${missing.length > 1 ? 's' : ''}: ${missing.join(', ')}. Use the May 25 input-features file.`
    );
  }
  const at = (name: string) => header.indexOf(name);
  const col = (r: string[], name: string) => (at(name) >= 0 ? r[at(name)] : undefined);

  const windows: Record<string, FeatureRow[]> = {};
  let rowCount = 0;
  for (let n = 1; n < table.length; n++) {
    const r = table[n];
    const start = (col(r, 'window_start') ?? '').trim();
    const lat = num(col(r, 'lat'));
    const lon = num(col(r, 'lon'));
    if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(start)) {
      throw new UploadError(`Row ${n + 1}: window_start "${start}" is not a date and time like 2026-05-25T17:20:00.`);
    }
    if (lat === null || lon === null) {
      throw new UploadError(`Row ${n + 1}: lat and lon must be numbers.`);
    }
    const label = (col(r, 'human_label') ?? '').trim();
    const row: FeatureRow = {
      window_start: start,
      window_end: (col(r, 'window_end') ?? '').trim(),
      day: (col(r, 'day') ?? start.slice(0, 10)).trim(),
      session: ((col(r, 'session') ?? '').trim() === 'AM' || Number(start.slice(11, 13)) < 12 ? 'AM' : 'PM'),
      split: (col(r, 'split') ?? '').trim(),
      camera: (col(r, 'camera') ?? '').trim(),
      lat,
      lon,
      human_label: LABELS.has(label) ? (label as FeatureRow['human_label']) : null,
      vehicle_count_mean: num(col(r, 'vehicle_count_mean')),
      model_risk: num(col(r, 'model_risk')),
      temp_c: num(col(r, 'temp_c')),
      precip_mm: num(col(r, 'precip_mm')),
      humidity_pct: num(col(r, 'humidity_pct')),
      flood_level: num(col(r, 'flood_level')),
      incident_count: num(col(r, 'incident_count')) ?? 0,
      incident_strength: num(col(r, 'incident_strength')) ?? 0,
      incident_lanes: num(col(r, 'incident_lanes')) ?? 0,
      incident_places: (col(r, 'incident_places') ?? '').trim(),
    };
    (windows[start] ??= []).push(row);
    rowCount++;
  }
  const known = new Set([
    ...REQUIRED_COLUMNS,
    'window_end', 'day', 'session', 'split', 'human_label', 'vehicle_count_mean', 'model_risk', 'temp_c',
    'precip_mm', 'humidity_pct', 'flood_level', 'incident_count', 'incident_strength', 'incident_lanes',
    'incident_places', 'node_id', 'human_score',
  ]);
  return {
    fileName,
    rowCount,
    windows,
    windowStarts: Object.keys(windows).sort(),
    extraColumns: header.filter((h) => !known.has(h)),
  };
}

const clock = (iso: string) => {
  const h = Number(iso.slice(11, 13));
  const m = iso.slice(14, 16);
  return `${h % 12 || 12}:${m} ${h < 12 ? 'AM' : 'PM'}`;
};

const mean = (xs: (number | null)[]): number | null => {
  const ok = xs.filter((x): x is number => x !== null);
  return ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : null;
};

/** One uploaded window as the same shape the server's "Inputs used" panel uses. */
export function toRecordedInputs(upload: UploadedInputs, windowStart: string): RecordedInputs | null {
  const rows = upload.windows[windowStart];
  if (!rows?.length) return null;
  const first = rows[0];
  const date = new Date(`${first.day}T00:00:00`);
  const label = `${date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}, ${clock(windowStart)}`;
  const held = first.split === 'test';
  const matched: WindowInfo = {
    window: windowStart,
    window_end: first.window_end || null,
    day: first.day,
    session: first.session,
    label: first.window_end ? `${label} - ${clock(first.window_end)}` : label,
    split: first.split || 'test',
    held_out: held,
    split_note: held ? 'held out: the model never saw this session in training or validation' : '',
  };
  const cameras: CameraReading[] = rows.map((r) => ({
    camera: r.camera,
    node_id: 0,
    lat: r.lat,
    lon: r.lon,
    model_risk: r.model_risk,
    human_label: r.human_label,
    vehicle_count: r.vehicle_count_mean,
  }));
  const incidents: IncidentReading[] = rows
    .filter((r) => r.incident_count > 0)
    .map((r) => ({
      location: r.incident_places || r.camera,
      intersection: r.camera,
      kind: 'incident',
      lanes: r.incident_lanes,
      strength: r.incident_strength,
      reported_at: windowStart,
      minutes_ago: null,
      lat: r.lat,
      lon: r.lon,
    }));
  const flood = mean(rows.map((r) => r.flood_level));
  return {
    recorded: true,
    source: 'uploaded',
    source_name: upload.fileName,
    matched_window: matched,
    cameras,
    incidents,
    weather: {
      available: rows.some((r) => r.temp_c !== null),
      where: 'mean over the cameras in the uploaded file',
      hour: windowStart.slice(0, 13).replace('T', ' ') + ':00',
      temp_c: mean(rows.map((r) => r.temp_c)),
      precip_mm: mean(rows.map((r) => r.precip_mm)),
      humidity_pct: mean(rows.map((r) => r.humidity_pct)),
      source: 'uploaded file',
    },
    flood: {
      mean_level: flood,
      share_of_roads_flood_prone: rows.filter((r) => (r.flood_level ?? 0) > 0).length / rows.length,
      source: 'uploaded file (Project NOAH level at each camera)',
    },
    vehicle_counts_available: rows.some((r) => r.vehicle_count_mean !== null),
  };
}
