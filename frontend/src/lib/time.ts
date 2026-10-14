/** Clock and departure-time helpers, extracted from the old HomeScreen. */

export function formatClock(date: Date): string {
  let hours = date.getHours();
  const minutes = date.getMinutes().toString().padStart(2, '0');
  const ampm = hours >= 12 ? 'PM' : 'AM';
  hours = hours % 12 || 12;
  return `${hours}:${minutes} ${ampm}`;
}

/** departAt null means leaving now. */
export function getArrivalTime(durationMin: number, departAt: Date | null): string {
  const arrival = departAt ? new Date(departAt) : new Date();
  arrival.setMinutes(arrival.getMinutes() + durationMin);
  return formatClock(arrival);
}

export function startOfDay(date: Date): Date {
  const d = new Date(date);
  d.setHours(0, 0, 0, 0);
  return d;
}

export function daysFromToday(date: Date): number {
  return Math.round((startOfDay(date).getTime() - startOfDay(new Date()).getTime()) / 86400000);
}

/** The current minute -- the earliest a planned departure can be. */
export function currentMinute(): Date {
  const d = new Date();
  d.setSeconds(0, 0);
  return d;
}

/** Keep a planned departure between now and the last minute of tomorrow. */
export function clampDeparture(date: Date): Date {
  const earliest = currentMinute();
  const latest = startOfDay(new Date());
  latest.setDate(latest.getDate() + 2);
  latest.setMinutes(-1);
  if (date < earliest) return earliest;
  if (date > latest) return latest;
  return date;
}

export function formatDeparture(date: Date): string {
  return `${daysFromToday(date) === 1 ? 'Tomorrow ' : ''}${formatClock(date)}`;
}

/**
 * The <input type="date"> / <input type="time"> value for a Date, in the browser's local
 * zone. toISOString() would shift by the UTC offset and land on the wrong day.
 */
export function toDateInputValue(date: Date): string {
  const y = date.getFullYear();
  const m = (date.getMonth() + 1).toString().padStart(2, '0');
  const d = date.getDate().toString().padStart(2, '0');
  return `${y}-${m}-${d}`;
}

export function toTimeInputValue(date: Date): string {
  const h = date.getHours().toString().padStart(2, '0');
  const m = date.getMinutes().toString().padStart(2, '0');
  return `${h}:${m}`;
}

/** Applies a "YYYY-MM-DD" date input onto an existing Date, keeping its time of day. */
export function withDateInput(base: Date, value: string): Date {
  const [y, m, d] = value.split('-').map(Number);
  if (!y || !m || !d) return base;
  const next = new Date(base);
  next.setFullYear(y, m - 1, d);
  return next;
}

/** Applies an "HH:MM" time input onto an existing Date, keeping its day. */
export function withTimeInput(base: Date, value: string): Date {
  const [h, min] = value.split(':').map(Number);
  if (Number.isNaN(h) || Number.isNaN(min)) return base;
  const next = new Date(base);
  next.setHours(h, min, 0, 0);
  return next;
}
