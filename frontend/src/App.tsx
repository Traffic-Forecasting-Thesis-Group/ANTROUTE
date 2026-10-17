import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { loadNetwork, loadStudyStatus, loadWeather, planRoute, snapNode, snapRoad } from './api/routeService';
import type { HistoricalStudyStatus as StudyStatus, NetworkInfo, RoadSelection, RouteResult, StudyNode, WeatherSnapshot } from './api/routeService';
import RouteMap from './features/map/RouteMap';
import PlaceInput from './features/trip/PlaceInput';
import HistoricalStudyStatus from './features/study/HistoricalStudyStatus';

const features = ['Distance', 'Traffic Congestion', 'Travel Time', 'Number of Intersections', 'Traffic Signal', 'Road Type', 'Time of Day', 'Weather', 'Flood Risk', 'Road Capacity'];
const cities = ['San Juan', 'Manila', 'Mandaluyong', 'Makati', 'Pasig'];
const emptyGeometry: [number, number][] = [];
const emptyBoundaries = {};

function CostHistory({ values }: { values: (number | null)[] }) {
  const costs = values.filter((value): value is number => value !== null);
  if (!costs.length) return null;
  const high = Math.max(...costs), low = Math.min(...costs);
  const points = values.flatMap((value, i) => value === null ? [] : [
    `${20 + i / Math.max(values.length - 1, 1) * 560},${110 - (value - low) / (high - low || 1) * 90}`,
  ]).join(' ');
  return <figure>
    <figcaption>Best path cost by iteration</figcaption>
    <svg viewBox="0 0 600 140" role="img" aria-label={`Best path cost over ${values.length} iterations, ending at ${costs[costs.length - 1].toFixed(4)}`}>
      <path d="M20 15V115H585" fill="none" stroke="#cbd5e1" />
      <polyline points={points} fill="none" stroke="#4475F2" strokeWidth="3" />
      {costs.length === 1 && <circle cx={points.split(',')[0]} cy="110" r="4" fill="#4475F2" />}
      <text x="20" y="135" fontSize="11">Iteration 1</text>
      <text x="510" y="135" fontSize="11">Iteration {values.length}</text>
    </svg>
    <p className="muted">Cost range: {low.toFixed(4)} – {high.toFixed(4)}</p>
  </figure>;
}

export default function App() {
  const [studyStatus, setStudyStatus] = useState<StudyStatus | null>(null);
  const [network, setNetwork] = useState<NetworkInfo | null>(null);
  const [origin, setOrigin] = useState<StudyNode | null>(null);
  const [destination, setDestination] = useState<StudyNode | null>(null);
  const [departureDate, setDepartureDate] = useState('2025-06-01');
  const [hour, setHour] = useState(8);
  const [weather, setWeather] = useState('clear');
  const [snapshot, setSnapshot] = useState<WeatherSnapshot | null>(null);
  const [closures, setClosures] = useState<RoadSelection[]>([]);
  const [bothDirections, setBothDirections] = useState(true);
  const [pickMode, setPickMode] = useState<'origin' | 'destination' | 'closure'>('origin');
  const [result, setResult] = useState<RouteResult | null>(null);
  const [error, setError] = useState('');
  const [weatherError, setWeatherError] = useState('');
  const [busy, setBusy] = useState(false);
  const [picking, setPicking] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(true);
  const activeRequest = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    loadStudyStatus(controller.signal).then(status => {
      setStudyStatus(status);
      if (!status.ready) return null;
      return loadNetwork(controller.signal);
    }).then(async data => {
      if (!data) return;
      setNetwork(data);
      setWeather(data.weather_conditions.includes('historical') ? 'historical' : data.weather_conditions[0]);

    }).catch((reason: Error) => {
      if (reason.name !== 'AbortError') setError(reason.message);
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false);
    });
    return () => { controller.abort(); activeRequest.current?.abort(); };
  }, []);

  useEffect(() => {
    if (!network) return;
    const controller = new AbortController();
    setSnapshot(null);
    loadWeather(departureDate, hour, controller.signal).then(value => { setSnapshot(value); setWeatherError(''); })
      .catch((reason: Error) => { if (reason.name !== 'AbortError') setWeatherError('Historical provider weather is unavailable for this timestamp.'); });
    return () => controller.abort();
  }, [network, departureDate, hour]);

  const invalidate = () => { setResult(null); setError(''); };
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!origin || !destination) { setError('Choose an origin and destination from the search results or map.'); return; }
    if (origin.id === destination.id) { setError('Choose different origin and destination locations.'); return; }
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    setBusy(true); setResult(null); setError('');
    try {
      const next = await planRoute(origin.id, destination.id, {
        departure_date: departureDate, time_of_day: hour, weather, closed_road_ids: closures.map(road => road.id),
      }, controller.signal);
      setResult(next);
    } catch (reason) {
      if (reason instanceof Error && reason.name !== 'AbortError') setError(reason.message);
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  };

  const pickPoint = async (latitude: number, longitude: number) => {
    setPicking(true); setError('');
    try {
      if (pickMode === 'closure') {
        const road = await snapRoad(latitude, longitude);
        const selected = bothDirections ? [road, ...(road.opposite_roads ?? [])] : [road];
        setClosures(previous => [...new Map([...previous, ...selected].map(item => [item.id, item])).values()]);
      } else {
        const node = await snapNode(latitude, longitude);
        if (pickMode === 'origin') { setOrigin(node); setPickMode('destination'); }
        else setDestination(node);
      }
      invalidate();
    } catch (reason) { if (reason instanceof Error) setError(reason.message); }
    finally { setPicking(false); }
  };

  const updateWeather = async () => {
    setRefreshing(true); setWeatherError('');
    try {
      const observed = await loadWeather(departureDate, hour);
      setSnapshot(observed); invalidate();
    } catch (reason) { if (reason instanceof Error) setWeatherError(reason.message); }
    finally { setRefreshing(false); }
  };

  const disabled = busy || picking || refreshing || !network;
  const evaluation = result?.evaluation;
  const convergence = evaluation?.convergence_rule;
  const cityWeather = snapshot?.cities[origin?.city ?? 'Manila'];

  if (studyStatus && !studyStatus.ready) return <HistoricalStudyStatus status={studyStatus} />;

  return <>
    <header><div className="brand">ANTROUTE</div><span>Ant Colony Optimization</span></header>
    <main>
      <div className="page-title"><div><h1>ACO Route Planner</h1><p>{cities.join(' · ')}</p></div><span className="badge">Five-city study</span></div>
      {network && <div className="data-notice"><strong>Historical routing with documented estimates</strong><span>{network.node_count.toLocaleString()} nodes · {network.road_count.toLocaleString()} directed road segments. Traffic conditions are modeled, not measured historical traffic. Road inventory is a 2026 surrogate; all estimation methods are disclosed below.</span></div>}
      <div className="layout">
        <section className="card planner">
          <h2>Plan a route</h2>
          {loading && <p role="status">Loading the study network…</p>}
          <form onSubmit={submit}>
            <fieldset disabled={disabled}>
              <PlaceInput label="Origin" value={origin} onSelect={node => { setOrigin(node); invalidate(); }} />
              <PlaceInput label="Destination" value={destination} onSelect={node => { setDestination(node); invalidate(); }} />
              <h3>Dynamic conditions</h3>
              <label>Study date<input aria-label="Study date" type="date" min="2025-06-01" max="2026-05-31" required value={departureDate} onChange={event => { setDepartureDate(event.target.value); invalidate(); }} /></label>
              <label>Time of Day (Asia/Manila)<select aria-label="Time of Day (Asia/Manila)" value={hour} onChange={event => { setHour(Number(event.target.value)); invalidate(); }}>{Array.from({ length: 24 }, (_, value) => <option key={value} value={value}>{String(value).padStart(2, '0')}:00</option>)}</select></label>
              <label>Weather Conditions<select aria-label="Weather Conditions" required value={weather} onChange={event => { setWeather(event.target.value); invalidate(); }}>{network?.weather_conditions.map(value => <option key={value} value={value} disabled={value === 'current' && (!snapshot || snapshot.stale)}>{value === 'historical' ? 'Historical provider weather' : `${value.replaceAll('_', ' ')} scenario`}</option>)}</select></label>
              {cityWeather && <p className="muted weather-status">{origin?.city ?? 'Manila'}: {cityWeather.description.join(', ')} · {cityWeather.precip_mm} mm precipitation<br />Provider timestamp: {cityWeather.localtime} {snapshot?.stale ? '(refresh required)' : ''}. Routing effects are estimated.</p>}
              <button type="button" className="secondary" onClick={updateWeather}>{refreshing ? 'Refreshing…' : 'Reload historical weather'}</button>
              {weatherError && <p className="error" role="status">{weatherError}</p>}
              <h3>Road Closure</h3>
              <p className="muted">Select “Close road” above the map, then click a road segment. These are hypothetical closure scenarios, not verified historical closure records.</p>
              <label className="checkbox"><input type="checkbox" checked={bothDirections} onChange={event => setBothDirections(event.target.checked)} />Close both mapped directions</label>
              <ul className="closure-list">{closures.map(road => <li key={road.id}><span>{road.name}</span><button type="button" aria-label={`Reopen ${road.name}`} onClick={() => { setClosures(previous => previous.filter(item => item.id !== road.id)); invalidate(); }}>Reopen</button></li>)}</ul>
              {!closures.length && <p className="muted">No closures entered.</p>}
              <button type="submit">{busy ? 'Running ACO…' : 'Calculate route'}</button>
            </fieldset>
          </form>
          {error && <p className="error" role="alert">{error}</p>}
        </section>
        <section className="card map-card">
          <h2>Route within the study area</h2>
          <div className="map-toolbar" role="group" aria-label="Map selection mode">{(['origin', 'destination', 'closure'] as const).map(mode => <button type="button" key={mode} disabled={disabled} aria-pressed={pickMode === mode} onClick={() => setPickMode(mode)}>{mode === 'closure' ? 'Close road' : `Set ${mode}`}</button>)}</div>
          <p className="muted" role="status">{picking ? 'Finding the nearest study road…' : `Click the map to ${pickMode === 'closure' ? 'enter a road closure' : `choose the ${pickMode}`}.`}</p>
          <div className="map"><RouteMap geometry={result?.route.geometry ?? emptyGeometry} boundaries={network?.boundaries ?? emptyBoundaries} origin={origin} destination={destination} closures={closures} onPick={pickPoint} disabled={disabled} /></div>
          <p className="muted">{result ? `${result.route.road_ids.length} road segments in the best route found by ACO.` : 'Search road names or select points on the map.'} Blue boundaries mark the study area; red segments are closed.</p>
          <p className="attribution">Roads and boundaries © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap contributors</a>. Flood hazard source: Project NOAH.</p>
        </section>
      </div>
      <section className="card evaluation">
        <h2>Evaluation</h2>
        {evaluation && convergence ? <>
          <div className="metric-grid">
            <div className="metric"><span>Distance Travel</span><strong>{(evaluation.distance_travel_m / 1000).toFixed(3)} km</strong></div>
            <div className="metric"><span>Path Cost</span><strong>{evaluation.path_cost.toFixed(4)}</strong></div>
            <div className="metric"><span>Computational Time</span><strong>{evaluation.computational_time_ms.toFixed(1)} ms</strong></div>
            <div className="metric"><span>Convergence Rule</span><strong>{convergence.converged ? 'Satisfied' : 'Iteration limit'}</strong><small>Iteration {convergence.iterations} / {convergence.max_iterations}</small></div>
          </div>
          <p className="muted">Convergence: relative improvement ≤ {convergence.tolerance} for {convergence.patience} consecutive iterations. Satisfying this rule does not establish global optimality.</p>
          <CostHistory values={convergence.best_cost_by_iteration} />
          <details><summary>Route roads and input sources</summary><p>{result.route.road_names.filter((name, index, array) => index === 0 || name !== array[index - 1]).join(' → ')}</p><pre>{JSON.stringify(result.route.feature_sources, null, 2)}</pre></details>
        </> : <p className="muted">Distance traveled, path cost, computational time, and convergence results appear after a completed run.</p>}
      </section>
      <section className="card"><h2>Routing features</h2><div className="feature-list">{features.map(feature => <span key={feature}>{feature}</span>)}</div>
        <details className="source-details"><summary>Where each feature and estimate comes from</summary>
          <div className="audit-table"><table><thead><tr><th>Feature</th><th>Classification</th><th>Basis and assumptions</th><th>Reference</th></tr></thead><tbody>
            {Object.entries(network?.data_quality.estimation_registry ?? {}).map(([name, entry]) => <tr key={name}><th scope="row">{name.replaceAll('_', ' ')}</th><td>{entry.classification.replaceAll('_', ' ')}</td><td>{entry.basis}</td><td>{entry.source.startsWith('https://') ? <a href={entry.source} target="_blank" rel="noreferrer">Source</a> : entry.source}</td></tr>)}
          </tbody></table></div>
        </details>
        <details className="source-details"><summary>Data sources and estimate limitations</summary><ul>{network?.data_quality.limitations.map(item => <li key={item}>{item}</li>)}</ul></details>
      </section>
    </main>
  </>;
}
