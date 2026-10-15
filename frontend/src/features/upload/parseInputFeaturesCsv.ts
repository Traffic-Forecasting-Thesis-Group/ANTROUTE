import apiClient from '@/api/client';
import { SIMULATION_DATE } from '@/lib/constants';
import type { IncidentKind, MapIncident, TrafficBand } from '@/types/markers';

/**
 * Reads an uploaded CSV of the simulation day's input features and turns it into map markers.
 *
 * Two shapes are accepted, because both exist in this project:
 *
 *   1. The events list -- one row per event, and nothing else:
 *      event_id,event_type,event_severity,segment_id,start_time,end_time,city
 *      EV1,accident,2,21834288>88772467,08:05,08:55,San Juan
 *
 *   2. The day export (test_day_YYYY-MM-DD.csv) -- one row per segment per minute, carrying
 *      all four input-feature families the model is given:
 *        traffic   congestion_level (free / moderate / heavy)
 *        events    event_flag, event_type, event_severity
 *        weather   ws_precip_mm (and the rest of the ws_* columns)
 *        flood     flood_depth_cm, on segments graded by flood_hazard_level
 *      A feature holding over consecutive minutes is one condition, so the minutes are
 *      grouped back into periods: "this segment was heavily congested 07:10-07:45",
 *      "this city had moderate rain 13:00-16:20". Each period is one marker, drawn solid
 *      while the planned departure falls inside it and dimmed outside (see isInForce).
 *
 * Both shapes locate a row by a segment, written "source>target" as two graph node ids. Node
 * ids mean nothing in the browser, so the server resolves them (GET /routes/node-coordinates)
 * unless the file already carries lat/lon. A row whose nodes the graph does not have is
 * reported as a problem rather than dropped silently or placed at a guessed point.
 *
 * Nothing here changes a route. The features are drawn on the map; the planner is untouched.
 */

export interface ParsedInputFeatures {
  incidents: MapIncident[];
  /** Conditions the file described, including any that could not be placed. */
  rowCount: number;
  /** One line naming what was found, per feature family. */
  summary: string | null;
  problems: string[];
}

/** Shape 1: the events list. */
const EVENT_COLUMNS = [
  'event_id',
  'event_type',
  'event_severity',
  'segment_id',
  'start_time',
  'end_time',
] as const;

/** Shape 2: the day export. The feature columns beyond these are optional. */
const DAY_COLUMNS = ['timestamp', 'segment_id'] as const;

/**
 * Rain from this rate upwards is worth a marker; above the second it reads as heavy.
 * (2.5 and 7.6 mm/h are the usual light / moderate / heavy rainfall boundaries.)
 */
const RAIN_MM = 2.5;
const RAIN_HEAVY_MM = 7.6;

/**
 * Standing water from this depth upwards is worth a marker; above the second it is the depth
 * drivers avoid. flood_hazard_level is NOT used for this: it holds one value per segment for
 * the whole day -- it is the segment's hazard zone, not water on the road. flood_depth_cm is
 * the column that actually moves, so it is what the periods are cut from.
 */
const FLOOD_CM = 5;
const FLOOD_DEEP_CM = 15;

/** Two sampled minutes of the same condition this far apart still count as one period. */
const CONDITION_GAP_MS = 10 * 60_000;
/** A condition shorter than this is sampling flicker, not a period worth a marker. */
const CONDITION_MIN_MS = 10 * 60_000;

/** The legend's categories. Anything that blocks the road is a closure; the rest are events. */
function toKind(eventType: string): IncidentKind {
  return eventType.trim().toLowerCase() === 'road_closure' ? 'road_closure' : 'accident';
}

/** 1 / 2 / 3 in the file, lightest first. */
function toSeverity(value: string | number): TrafficBand | null {
  switch (String(value).trim()) {
    case '1':
      return 'light';
    case '2':
      return 'moderate';
    case '3':
      return 'heavy';
    default:
      return null;
  }
}

function titleCase(eventType: string): string {
  return eventType
    .trim()
    .split('_')
    .filter(Boolean)
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(' ');
}

/** "HH:MM" on the simulation day, as a local ISO string. */
function atSimulationTime(hhmm: string): string | null {
  const [h, m] = hhmm.trim().split(':').map(Number);
  if (!Number.isFinite(h) || !Number.isFinite(m)) return null;
  const d = new Date();
  d.setFullYear(SIMULATION_DATE.year, SIMULATION_DATE.month - 1, SIMULATION_DATE.day);
  d.setHours(h, m, 0, 0);
  return d.toISOString();
}

/** "2026-05-25 08:05:00" (or with a T) as local milliseconds; NaN when unreadable. */
function readTimestamp(value: string): number {
  return new Date(value.trim().replace(' ', 'T')).getTime();
}

function clockTime(ms: number): string {
  const d = new Date(ms);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

function plural(n: number, noun: string): string {
  return `${n.toLocaleString()} ${noun}${n === 1 ? '' : 's'}`;
}

/**
 * Minimal CSV reader: handles quoted fields and CRLF, which is all these formats need.
 * Rows are handed over one at a time -- the day export is tens of megabytes, and holding
 * every row at once costs several times the file itself.
 */
function forEachRow(text: string, onRow: (row: string[]) => void): void {
  let row: string[] = [];
  let value = '';
  let quoted = false;

  const endRow = () => {
    row.push(value);
    if (row.some((v) => v.trim() !== '')) onRow(row);
    row = [];
    value = '';
  };

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          value += '"';
          i++;
        } else {
          quoted = false;
        }
      } else {
        value += c;
      }
      continue;
    }
    if (c === '"') quoted = true;
    else if (c === ',') {
      row.push(value);
      value = '';
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      endRow();
    } else {
      value += c;
    }
  }
  endRow();
}

function readCsv(text: string): string[][] {
  const rows: string[][] = [];
  forEachRow(text, (row) => rows.push(row));
  return rows;
}

/** The header line alone, read without pulling the whole file into memory. */
async function readHeader(file: File): Promise<string[]> {
  const head = await file.slice(0, 64 * 1024).text();
  const line = head.split(/\r?\n/, 1)[0] ?? '';
  return readCsv(line)[0]?.map((h) => h.trim().toLowerCase()) ?? [];
}

/** Coordinates for the source node of each "source>target" segment, from the server's graph. */
async function resolveSegments(
  segments: Iterable<string>
): Promise<Record<string, { lat: number; lng: number }>> {
  const nodeIds = new Set<string>();
  for (const segment of segments) {
    for (const part of segment.split('>')) {
      const id = part.trim();
      if (/^\d+$/.test(id)) nodeIds.add(id);
    }
  }
  if (nodeIds.size === 0) return {};
  const { data } = await apiClient.get<{ nodes: Record<string, { lat: number; lng: number }> }>(
    '/routes/node-coordinates',
    { params: { ids: [...nodeIds].join(',') } }
  );
  return data.nodes ?? {};
}

const UNREACHABLE = 'Could not reach the server to place these features on the map.';

export async function parseInputFeaturesCsv(file: File): Promise<ParsedInputFeatures> {
  // Read the columns before the rest of the file: the day export is tens of megabytes, and
  // which of the two shapes it is decides how its rows have to be read.
  const columns = await readHeader(file);
  const has = (c: string) => columns.includes(c);

  if (EVENT_COLUMNS.every(has)) return parseEventList(file);
  if (DAY_COLUMNS.every(has)) return parseDayExport(file);

  // Neither shape. Name what the events list is missing -- it is the smaller, likelier file.
  const missing = EVENT_COLUMNS.filter((c) => !has(c));
  return {
    incidents: [],
    rowCount: 0,
    summary: null,
    problems: [
      `This file is missing the column(s): ${missing.join(', ')}. ` +
        'Upload either the events file (one row per event: event_id, event_type, ' +
        'event_severity, segment_id, start_time, end_time, city) or the day export, which ' +
        'carries traffic, events, weather and flood per segment per minute (timestamp, ' +
        'segment_id, congestion_level, event_type, ws_precip_mm, flood_hazard_level, …).',
    ],
  };
}

/** Shape 1: one row per event. */
async function parseEventList(file: File): Promise<ParsedInputFeatures> {
  const rows = readCsv(await file.text());
  if (rows.length < 2) {
    return {
      incidents: [],
      rowCount: 0,
      summary: null,
      problems: ['The file has no rows under its header.'],
    };
  }

  const header = rows[0].map((h) => h.trim().toLowerCase());
  const at = (row: string[], column: string) => (row[header.indexOf(column)] ?? '').trim();
  const body = rows.slice(1);
  const problems: string[] = [];

  let coords: Record<string, { lat: number; lng: number }>;
  try {
    coords = await resolveSegments(body.map((row) => at(row, 'segment_id')));
  } catch {
    return { incidents: [], rowCount: body.length, summary: null, problems: [UNREACHABLE] };
  }

  const incidents: MapIncident[] = [];
  for (const row of body) {
    const id = at(row, 'event_id') || `row ${incidents.length + 1}`;
    const segment = at(row, 'segment_id');
    const [from] = segment.split('>').map((s) => s.trim());
    const point = coords[from];
    if (!point) {
      problems.push(`${id}: segment ${segment || '(blank)'} is not on the road network.`);
      continue;
    }
    const eventType = at(row, 'event_type');
    const city = at(row, 'city');
    const starts = at(row, 'start_time');
    const ends = at(row, 'end_time');

    incidents.push({
      id,
      kind: toKind(eventType),
      position: { latitude: point.lat, longitude: point.lng },
      title: titleCase(eventType) || 'Event',
      description: [city, starts && `${starts}–${ends}`].filter(Boolean).join(' · '),
      severity: toSeverity(at(row, 'event_severity')),
      startsAt: atSimulationTime(starts),
      endsAt: atSimulationTime(ends),
      source: file.name,
    });
  }

  return {
    incidents,
    rowCount: body.length,
    summary: `${plural(incidents.length, 'event')} on the map.`,
    problems,
  };
}

/** Which input-feature family a period belongs to. */
type Family = 'event' | 'traffic' | 'rain' | 'flood';

/** One condition holding on one place over a stretch of consecutive minutes. */
interface Period {
  family: Family;
  kind: IncidentKind;
  label: string;
  segment: string;
  city: string;
  /** The road's name, when the file gives one. */
  road: string;
  /** Running mean of the rows' own coordinates; null when the file carries none. */
  latSum: number;
  lngSum: number;
  points: number;
  /** Highest level/intensity reached, however the family measures it. */
  peak: number;
  /** A second reading worth naming, e.g. flood depth in cm. */
  peakDetail: number;
  first: number;
  last: number;
}

/** What each family contributes to the marker and how its periods are cut. */
const FAMILY_RULES: Record<Family, { gapMs: number; minMs: number }> = {
  // An event's minutes are written out contiguously, so any break is a different event,
  // and even a short one is real rather than flicker.
  event: { gapMs: 0, minMs: 0 },
  traffic: { gapMs: CONDITION_GAP_MS, minMs: CONDITION_MIN_MS },
  rain: { gapMs: CONDITION_GAP_MS, minMs: CONDITION_MIN_MS },
  flood: { gapMs: CONDITION_GAP_MS, minMs: CONDITION_MIN_MS },
};

/**
 * Shape 2: one row per segment per minute, carrying every input feature.
 *
 * Each family is read off the row, and consecutive minutes of the same condition on the same
 * place are accumulated into one period. Traffic and flood are per segment; rain is per city,
 * which is the level the weather columns actually vary at.
 */
async function parseDayExport(file: File): Promise<ParsedInputFeatures> {
  const text = await file.text();

  let header: string[] | null = null;
  const index: Record<string, number> = {};
  /** Periods still being extended, by family + place + (for events) which event. */
  const open = new Map<string, Period>();
  const closed: Period[] = [];
  const segments = new Set<string>();
  const minutes = new Set<number>();

  let rowsRead = 0;
  let previousTime = NaN;
  /** The export's sampling interval, learned from the first two distinct timestamps. */
  let step = 60_000;

  const note = (
    family: Family,
    kind: IncidentKind,
    label: string,
    place: string,
    when: number,
    level: number,
    detail: number,
    row: { segment: string; city: string; road: string; lat: number; lng: number }
  ) => {
    const key = `${family}|${place}`;
    const current = open.get(key);
    // A gap wider than the family tolerates ends the period and starts a new one.
    if (current && when - current.last > step + FAMILY_RULES[family].gapMs) {
      closed.push(current);
      open.delete(key);
    }
    const period = open.get(key);
    if (period) {
      period.last = Math.max(period.last, when);
      period.peak = Math.max(period.peak, level);
      period.peakDetail = Math.max(period.peakDetail, detail);
      if (Number.isFinite(row.lat) && Number.isFinite(row.lng)) {
        period.latSum += row.lat;
        period.lngSum += row.lng;
        period.points++;
      }
      return;
    }
    const placed = Number.isFinite(row.lat) && Number.isFinite(row.lng) && row.lat !== 0;
    open.set(key, {
      family,
      kind,
      label,
      segment: row.segment,
      city: row.city,
      road: row.road,
      latSum: placed ? row.lat : 0,
      lngSum: placed ? row.lng : 0,
      points: placed ? 1 : 0,
      peak: level,
      peakDetail: detail,
      first: when,
      last: when,
    });
  };

  forEachRow(text, (row) => {
    if (header === null) {
      header = row.map((h) => h.trim().toLowerCase());
      header.forEach((name, i) => {
        if (!(name in index)) index[name] = i;
      });
      return;
    }
    rowsRead++;

    const at = (column: string) => {
      const i = index[column];
      return i === undefined ? '' : (row[i] ?? '').trim();
    };
    const num = (column: string) => Number(at(column));

    const when = readTimestamp(at('timestamp'));
    if (!Number.isFinite(when)) return;
    if (Number.isFinite(previousTime) && when > previousTime) {
      step = Math.min(step, when - previousTime);
    }
    previousTime = when;
    minutes.add(when);

    const segment = at('segment_id');
    const city = at('city');
    const lat = num('lat');
    const lng = Number(at('lon') || at('lng'));
    const where = { segment, city, road: at('name'), lat, lng };
    segments.add(segment);

    // --- events -----------------------------------------------------------------------
    // event_flag marks the minutes an event is in force; an export without it writes "none"
    // in event_type on the rest.
    const eventType = at('event_type');
    const flag = at('event_flag');
    const hasEvent =
      eventType !== '' &&
      eventType.toLowerCase() !== 'none' &&
      !(flag === '0' || flag.toLowerCase() === 'false');
    if (hasEvent) {
      const severity = at('event_severity');
      note(
        'event',
        toKind(eventType),
        titleCase(eventType) || 'Event',
        `${eventType}|${severity}|${segment}`,
        when,
        Number(severity) || 0,
        0,
        where
      );
    }

    // --- traffic ----------------------------------------------------------------------
    // Only the heavy band gets a marker: moderate is the common case on these roads, and the
    // legend row it would answer to is "Heavy Traffic Segment".
    if (at('congestion_level').toLowerCase() === 'heavy') {
      note('traffic', 'traffic', 'Heavy Traffic', segment, when, 3, num('avg_speed_kph'), where);
    }

    // --- weather ----------------------------------------------------------------------
    // The ws_* columns vary by city, not by segment, so rain is grouped by city and the
    // marker sits at the mean of that city's segments.
    const precip = num('ws_precip_mm');
    if (Number.isFinite(precip) && precip >= RAIN_MM) {
      note('rain', 'weather_rain', 'Rain', city || segment, when, precip, precip, where);
    }

    // --- flood ------------------------------------------------------------------------
    const depth = num('flood_depth_cm');
    if (Number.isFinite(depth) && depth >= FLOOD_CM) {
      note('flood', 'flood', 'Flood', segment, when, depth, num('flood_hazard_level'), where);
    }
  });

  for (const period of open.values()) closed.push(period);

  if (closed.length === 0) {
    return {
      incidents: [],
      rowCount: 0,
      summary: null,
      problems: [
        rowsRead === 0
          ? 'The file has no rows under its header.'
          : `No row in this day export carries traffic, event, weather or flood data ` +
            `(${plural(rowsRead, 'row')} read).`,
      ],
    };
  }

  // A condition that barely registered is sampling flicker, not a period worth a marker.
  const periods = closed.filter(
    (p) => p.last + step - p.first >= FAMILY_RULES[p.family].minMs
  );

  // Only ask the server about the places whose coordinates the file did not carry.
  const unplaced = periods.filter((p) => p.points === 0);
  let coords: Record<string, { lat: number; lng: number }> = {};
  if (unplaced.length > 0) {
    try {
      coords = await resolveSegments(unplaced.map((p) => p.segment));
    } catch {
      return { incidents: [], rowCount: periods.length, summary: null, problems: [UNREACHABLE] };
    }
  }

  const incidents: MapIncident[] = [];
  const problems: string[] = [];
  const counts: Record<Family, number> = { event: 0, traffic: 0, rain: 0, flood: 0 };

  for (const period of periods) {
    let latitude: number;
    let longitude: number;
    if (period.points > 0) {
      latitude = period.latSum / period.points;
      longitude = period.lngSum / period.points;
    } else {
      const [from] = period.segment.split('>').map((s) => s.trim());
      const point = coords[from];
      if (!point) {
        problems.push(
          `${period.label}: segment ${period.segment || '(blank)'} is not on the road network.`
        );
        continue;
      }
      latitude = point.lat;
      longitude = point.lng;
    }

    // The last sample sits one step short of the end (08:54 for an 08:05-08:55 event), so the
    // window closes at the end of that sample rather than at its start.
    const endsAt = period.last + step;
    const window = `${clockTime(period.first)}–${clockTime(endsAt)}`;
    counts[period.family]++;

    incidents.push({
      id: `${period.family}-${period.label}-${period.segment || period.city}-${clockTime(period.first)}`,
      kind: period.kind,
      position: { latitude, longitude },
      title: period.label,
      description: [placeOf(period), detailOf(period), window].filter(Boolean).join(' · '),
      severity: bandOf(period),
      startsAt: new Date(period.first).toISOString(),
      endsAt: new Date(endsAt).toISOString(),
      source: file.name,
    });
  }

  const found = (['traffic', 'event', 'rain', 'flood'] as const)
    .filter((f) => counts[f] > 0)
    .map((f) => `${counts[f]} ${LABELS[f]}`)
    .join(', ');

  return {
    incidents,
    rowCount: periods.length,
    summary:
      `${found || 'nothing'} on the map, from ${plural(segments.size, 'segment')} ` +
      `over ${plural(minutes.size, 'minute')}.`,
    problems,
  };
}

const LABELS: Record<Family, string> = {
  traffic: 'heavy-traffic',
  event: 'event',
  rain: 'rain',
  flood: 'flood',
};

/** The legend band a period reads as. */
function bandOf(period: Period): TrafficBand | null {
  switch (period.family) {
    case 'traffic':
      return 'heavy';
    case 'rain':
      return period.peak >= RAIN_HEAVY_MM ? 'heavy' : 'moderate';
    case 'flood':
      return period.peak >= FLOOD_DEEP_CM ? 'heavy' : 'moderate';
    default:
      // An event grades 1 / 2 / 3.
      return toSeverity(period.peak);
  }
}

/** Road and city, as much of each as the file gave. */
function placeOf(period: Period): string {
  return [period.road, period.city].filter(Boolean).join(', ');
}

/** The one extra reading worth putting in the marker's tooltip. */
function detailOf(period: Period): string {
  switch (period.family) {
    case 'rain':
      return `peak ${period.peak.toFixed(1)} mm`;
    case 'flood':
      return (
        `peak ${period.peak.toFixed(1)} cm` +
        (period.peakDetail > 0 ? `, hazard level ${period.peakDetail}` : '')
      );
    case 'traffic':
      return '';
    default:
      return period.peak > 0 ? `severity ${period.peak}` : '';
  }
}

/** True when `at` falls inside the period's window; one with no window always counts. */
export function isInForce(incident: MapIncident, at: Date): boolean {
  if (!incident.startsAt || !incident.endsAt) return true;
  const from = new Date(incident.startsAt).getTime();
  const to = new Date(incident.endsAt).getTime();
  const when = at.getTime();
  return when >= from && when <= to;
}
