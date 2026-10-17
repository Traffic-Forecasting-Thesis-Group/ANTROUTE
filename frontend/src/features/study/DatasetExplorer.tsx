import { useEffect, useRef, useState } from 'react';
import { downloadSnapshot, loadDatasetInfo, loadDatasetMap } from '../../api/routeService';
import type { Conditions, DatasetInfo, DatasetMap } from '../../api/routeService';

interface Props {
  conditions: Conditions;
  onMap: (value: DatasetMap | null) => void;
  disabled: boolean;
}

export default function DatasetExplorer({ conditions, onMap, disabled }: Props) {
  const [info, setInfo] = useState<DatasetInfo | null>(null);
  const [city, setCity] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const active = useRef<AbortController | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    loadDatasetInfo(controller.signal).then(setInfo).catch((reason: Error) => {
      if (reason.name !== 'AbortError') setError('Dataset coverage is unavailable.');
    });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    active.current?.abort(); setBusy(false); onMap(null); setError('');
    return () => active.current?.abort();
  }, [conditions, city, onMap]);

  const run = async (task: 'map' | 'export') => {
    active.current?.abort();
    const controller = new AbortController(); active.current = controller;
    setBusy(true); setError('');
    try {
      if (task === 'map') {
        const map = await loadDatasetMap(conditions, city || null, controller.signal);
        if (!controller.signal.aborted) onMap(map);
      } else await downloadSnapshot(conditions, city || null, controller.signal);
    } catch (reason) {
      if (reason instanceof Error && reason.name !== 'AbortError') setError(reason.message);
    } finally { if (!controller.signal.aborted) setBusy(false); }
  };
  return <section className="card">
    <h2>Network dataset</h2>
    <p>Inspect every retained road segment, rather than only the selected route. The map and CSV use the date, hour, weather and closures above.</p>
    {info && <>
      <p><strong>{info.directed_segments.toLocaleString()} segments · {info.hours.toLocaleString()} hours · {info.potential_segment_hour_rows.toLocaleString()} possible segment-hour rows</strong></p>
      <p className="muted">{info.retained_weather_grids} retained weather grids; {info.assigned_weather_grids} assigned to road locations. {info.traffic_locations} hourly traffic locations (city × road type). Largest mutually reachable component: {info.largest_strongly_connected_nodes.toLocaleString()} of {info.nodes.toLocaleString()} nodes. Map completeness has not been verified.</p>
      <div className="feature-list">{Object.entries(info.segments_by_city).map(([name, count]) => <span key={name}>{name}: {count.toLocaleString()} segments</span>)}</div>
    </>}
    <div className="dataset-actions">
      <label>Dataset / map coverage<select value={city} onChange={event => setCity(event.target.value)}>
        <option value="">All five cities</option>{['San Juan', 'Manila', 'Mandaluyong', 'Makati', 'Pasig'].map(name => <option key={name}>{name}</option>)}
      </select></label>
      <button type="button" disabled={disabled || busy} onClick={() => run('map')}>Show network congestion on map</button>
      <button type="button" disabled={disabled || busy} onClick={() => run('export')}>Download hourly snapshot CSV</button>
    </div>
    {busy && <p role="status">Preparing the selected network snapshot…</p>}
    {error && <p role="alert" className="error">{error}</p>}
    <p className="muted">CSV rows contain all ten feature values, segment IDs, estimated path costs, and the weather grid and traffic location each segment reads. Full-year hourly CSVs are available through the dataset export script. Traffic values are estimated, not measured traffic.</p>
  </section>;
}
