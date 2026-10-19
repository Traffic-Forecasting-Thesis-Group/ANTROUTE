import { useEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { explainRoute, loadNetwork, loadStudyStatus, loadWeather, planTrip, snapNode, snapRoad } from './api/routeService';
import type { HistoricalStudyStatus as StudyStatus, NetworkInfo, RoadSelection, StudyNode, TripEvent, TripResult, WeatherSnapshot } from './api/routeService';
import RouteMap, { BASELINE_COLOR, ROUTE_COLOR } from './features/map/RouteMap';
import PlaceInput from './features/trip/PlaceInput';
import CostBreakdown from './features/trip/CostBreakdown';
import HistoricalStudyStatus from './features/study/HistoricalStudyStatus';
import DatasetExplorer from './features/study/DatasetExplorer';
import type { DatasetMap, RouteExplanation } from './api/routeService';
import RouteExplanationPanel, { COSTLIER_COLOR, MISSED_COLOR } from './features/map/RouteExplanation';

const features = ['Distance', 'Traffic Congestion', 'Travel Time', 'Number of Intersections', 'Traffic Signal', 'Road Type', 'Time of Day', 'Weather', 'Flood Risk', 'Road Capacity'];
const cities = ['San Juan', 'Manila', 'Mandaluyong', 'Makati', 'Pasig'];
const emptyGeometry: [number, number][] = [];
const emptyBoundaries = {};
const seconds = (milliseconds: number) => `${(milliseconds / 1000).toFixed(2)} s`;
const pad = (value: number) => String(value).padStart(2, '0');
const MAX_STOPS = 5;
const closeIcon = <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>;
const swapIcon = <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 4v15M8 4L4.5 7.5M8 4l3.5 3.5M16 20V5M16 20l-3.5-3.5M16 20l3.5-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>;
const capitalize = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);
/** The named road the trip spends the most segments on, for the route card title. */
const mainRoad = (trip: TripResult) => {
  const counts = new Map<string, number>();
  trip.route.road_names.filter(name => !/^\d/.test(name)).forEach(name => counts.set(name, (counts.get(name) ?? 0) + 1));
  return [...counts].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'unnamed roads';
};

function CostHistory({ values, caption, color, marker }: { values: (number | null)[]; caption: string; color: string; marker: number }) {
  const costs = values.filter((value): value is number => value !== null);
  if (!costs.length) return null;
  const high = Math.max(...costs), low = Math.min(...costs);
  const x = (i: number) => 20 + i / Math.max(values.length - 1, 1) * 560;
  const points = values.flatMap((value, i) => value === null ? [] : [`${x(i)},${110 - (value - low) / (high - low || 1) * 90}`]).join(' ');
  return <figure>
    <figcaption>{caption}</figcaption>
    <svg viewBox="0 0 600 140" role="img" aria-label={`${caption}: best cost over ${values.length} iterations, final best first found at iteration ${marker}`}>
      <path d="M20 15V115H585" fill="none" stroke="#cbd5e1" />
      <line x1={x(marker - 1)} x2={x(marker - 1)} y1="15" y2="115" stroke={color} strokeDasharray="4 4" />
      <polyline points={points} fill="none" stroke={color} strokeWidth="3" />
      <text x="20" y="135" fontSize="11">Iteration 1</text>
      <text x={Math.min(x(marker - 1) + 4, 470)} y="28" fontSize="11">Converged at {marker}</text>
      <text x="510" y="135" fontSize="11">Iteration {values.length}</text>
    </svg>
    <p className="muted">Best cost (own objective): {low.toFixed(4)} – {high.toFixed(4)}</p>
  </figure>;
}

export default function App() {
  const [studyStatus, setStudyStatus] = useState<StudyStatus | null>(null);
  const [network, setNetwork] = useState<NetworkInfo | null>(null);
  const [origin, setOrigin] = useState<StudyNode | null>(null);
  const [originKey, setOriginKey] = useState(0);
  // Stops are visited in this order. Keys stay with a stop when it is moved, and change to clear it.
  const [stops, setStops] = useState<{ key: number; node: StudyNode | null }[]>([{ key: 1, node: null }]);
  const nextStopKey = useRef(2);
  const chosenStops = stops.flatMap(stop => stop.node ? [stop.node] : []);
  const [departureDate, setDepartureDate] = useState('2025-06-01');
  const [hour, setHour] = useState(8);
  const [weather, setWeather] = useState('clear');
  const [snapshot, setSnapshot] = useState<WeatherSnapshot | null>(null);
  const [closures, setClosures] = useState<RoadSelection[]>([]);
  const [bothDirections, setBothDirections] = useState(true);
  const [pickMode, setPickMode] = useState<'origin' | 'destination' | 'closure'>('origin');
  const [departureMinute, setDepartureMinute] = useState(0);
  const [closureMinute, setClosureMinute] = useState<number | null>(null);
  const [weatherChange, setWeatherChange] = useState('');
  const [weatherChangeMinute, setWeatherChangeMinute] = useState(5);
  const [result, setResult] = useState<TripResult | null>(null);
  const [error, setError] = useState('');
  const [weatherError, setWeatherError] = useState('');
  const [busy, setBusy] = useState(false);
  const [picking, setPicking] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [datasetMap, setDatasetMap] = useState<DatasetMap | null>(null);
  const conditions = useMemo(() => ({ departure_date: departureDate, time_of_day: hour, weather,
    closed_road_ids: closures.map(road => road.id) }), [departureDate, hour, weather, closures]);
  const activeRequest = useRef<AbortController | null>(null);
  const [baseline, setBaseline] = useState<TripResult | null>(null);
  const [baselineBusy, setBaselineBusy] = useState(false);
  const [baselineError, setBaselineError] = useState('');
  const [explanation, setExplanation] = useState<RouteExplanation | null>(null);
  const [explanationNote, setExplanationNote] = useState('');
  const [focusedStretch, setFocusedStretch] = useState<number | null>(null);

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

  const invalidate = () => {
    activeRequest.current?.abort(); activeRequest.current = null;
    setResult(null); setError(''); setBaseline(null); setBaselineError(''); setBusy(false); setBaselineBusy(false); setExplanation(null); setFocusedStretch(null);
  };
  // Both algorithms drive the same trip. The baseline runs after ANTROUTE rather than alongside
  // it, so the two never compete for the server and computational times stay comparable.
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!origin || chosenStops.length !== stops.length) { setError('Choose the start location and every destination from the search results or map.'); return; }
    const sequence = [origin, ...chosenStops];
    if (sequence.some((node, index) => index > 0 && node.id === sequence[index - 1].id)) { setError('Each destination must differ from the location before it.'); return; }
    const destinationIds = chosenStops.map(node => node.id);
    invalidate();
    const controller = new AbortController();
    activeRequest.current = controller;
    const closedIds = closures.map(road => road.id);
    const tripConditions = { departure_date: departureDate, time_of_day: hour, weather,
      closed_road_ids: closureMinute === null ? closedIds : [] };
    const events: TripEvent[] = [];
    if (closureMinute !== null && closedIds.length) events.push({ at_minute: closureMinute, close_road_ids: closedIds });
    if (weatherChange) events.push({ at_minute: weatherChangeMinute, weather: weatherChange });
    setBusy(true);
    try {
      setResult(await planTrip('antroute', origin.id, destinationIds, tripConditions, departureMinute, events, controller.signal));
    } catch (reason) {
      if (reason instanceof Error && reason.name !== 'AbortError') setError(reason.message);
      return;
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
    setBaselineBusy(true);
    try {
      setBaseline(await planTrip('cheng', origin.id, destinationIds, tripConditions, departureMinute, events, controller.signal));
    } catch (reason) {
      if (reason instanceof Error && reason.name !== 'AbortError') setBaselineError(reason.message);
    } finally {
      if (!controller.signal.aborted) setBaselineBusy(false);
    }
    // Explains the departure plan of the first leg, after both timed runs so it never competes with them.
    try {
      setExplanation(await explainRoute(origin.id, destinationIds[0], tripConditions, controller.signal));
      setExplanationNote(destinationIds.length > 1 || events.length
        ? 'This explains the plan made at departure for the first destination; later legs and mid-trip re-routing are not compared.' : '');
    } catch {
      setExplanation(null);
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
        // The map stays in the mode the user chose; only the toolbar changes it.
        if (pickMode === 'origin') setOrigin(node);
        else setStops(previous => {
          // A map click fills the first empty destination, or adds one while there is room.
          const empty = previous.findIndex(stop => !stop.node);
          if (empty >= 0) return previous.map((stop, index) => index === empty ? { ...stop, node } : stop);
          return previous.length < MAX_STOPS ? [...previous, { key: nextStopKey.current++, node }] : previous;
        });
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
  const day = new Date(`${departureDate}T00:00:00`);
  const departureLabel = Number.isNaN(day.getTime()) ? 'Choose a date' : `${day.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })} (${[0, 6].includes(day.getDay()) ? 'weekend' : 'weekday'} traffic)`;
  const scenarioWeathers =network?.weather_conditions.filter(value => value !== 'historical' && value !== 'current') ?? [];
  const runs = [result && { name: 'ANTROUTE', color: ROUTE_COLOR, trip: result },
    baseline && { name: 'Baseline IACO', color: BASELINE_COLOR, trip: baseline }]
    .filter((run): run is { name: string; color: string; trip: TripResult } => !!run);
  const cityWeather = snapshot?.cities[origin?.city ?? 'Manila'];

  if (studyStatus && !studyStatus.ready) return <HistoricalStudyStatus status={studyStatus} />;

  return <>
    <header className="topbar">
      <a className="logo" href="#planner" aria-label="ANTRoute home"><img src="/logo-no-bg.png" alt="" /><span>ANT<b>Route</b></span></a>
      <nav aria-label="Sections"><a href="#planner">Planner</a><a href="#evaluation">Evaluation</a><a href="#features">Routing features</a><a href="#dataset">Dataset</a></nav>
      <span className="topbar-note">Ant Colony Optimization</span>
    </header>
    <div className="shell">
        <aside className="sidebar planner" id="planner">
          <h2>Plan a route</h2>
          {loading && <p role="status">Loading the study network…</p>}
          <form onSubmit={submit}>
            <fieldset disabled={disabled}>
              <div className="form-section first">
                <p className="field-label" aria-hidden="true">Origin</p>
                <PlaceInput key={`origin-${originKey}`} label="Start location" marker="origin" placeholder="Enter start location" value={origin}
                  onSelect={node => { setOrigin(node); invalidate(); }}
                  actions={<button type="button" className="icon-button" aria-label="Clear start location" onClick={() => { setOrigin(null); setOriginKey(key => key + 1); invalidate(); }}>{closeIcon}</button>} />
                <p className="field-label" aria-hidden="true">{stops.length > 1 ? 'Destinations' : 'Destination'}</p>
                {stops.map((stop, index) => <PlaceInput key={stop.key} label={`Destination ${index + 1}`} marker={index + 1} placeholder="Enter destination" value={stop.node}
                  onSelect={node => { setStops(previous => previous.map(item => item.key === stop.key ? { ...item, node } : item)); invalidate(); }}
                  actions={<>
                    <button type="button" className="icon-button" disabled={stops.length < 2} aria-label={`Move destination ${index + 1} ${index === stops.length - 1 ? 'up' : 'down'}`} title="Change order"
                      onClick={() => { setStops(previous => { const next = [...previous]; const other = index === previous.length - 1 ? index - 1 : index + 1; [next[index], next[other]] = [next[other], next[index]]; return next; }); invalidate(); }}>{swapIcon}</button>
                    <button type="button" className="icon-button" aria-label={`Remove destination ${index + 1}`}
                      onClick={() => { setStops(previous => previous.length > 1 ? previous.filter(item => item.key !== stop.key) : [{ key: nextStopKey.current++, node: null }]); invalidate(); }}>{closeIcon}</button>
                  </>} />)}
                {stops.length < MAX_STOPS && <button type="button" className="add-stop" onClick={() => setStops(previous => [...previous, { key: nextStopKey.current++, node: null }])}>
                  <span aria-hidden="true">+</span>Add Destination</button>}
              </div>
              <div className="form-section">
                <h3>Departure</h3>
                <div className="field-row">
                  <label>Date<input aria-label="Study date" type="date" min="2025-06-01" max="2026-05-31" required value={departureDate} onChange={event => { if (event.target.value) { setDepartureDate(event.target.value); invalidate(); } }} /></label>
                  <label>Time<input aria-label="Departure time" type="time" step={60} required value={`${pad(hour)}:${pad(departureMinute)}`} onChange={event => {
                    const [hours, minutes] = event.target.value.split(':').map(Number);
                    if (Number.isInteger(hours) && Number.isInteger(minutes)) { setHour(hours); setDepartureMinute(minutes); invalidate(); }
                  }} /></label>
                </div>
                <p className="hint">{departureLabel} · Asia/Manila. Traffic and weather update as the trip crosses each hour.</p>
              </div>
              <div className="form-section">
                <h3>Weather</h3>
                <label>At departure<select aria-label="Weather Conditions" required value={weather} onChange={event => { setWeather(event.target.value); invalidate(); }}>{network?.weather_conditions.map(value => <option key={value} value={value} disabled={value === 'current' && (!snapshot || snapshot.stale)}>{value === 'historical' ? 'Historical provider weather' : `${capitalize(value.replaceAll('_', ' '))} scenario`}</option>)}</select></label>
                <div className="field-row">
                  <label>Change during trip<select value={weatherChange} onChange={event => { setWeatherChange(event.target.value); invalidate(); }}>
                    <option value="">No change</option>{scenarioWeathers.map(value => <option key={value} value={value}>To {value.replaceAll('_', ' ')}</option>)}
                  </select></label>
                  <label>At minute<input type="number" min={0} max={1440} step={1} required disabled={!weatherChange} value={weatherChangeMinute} onChange={event => { setWeatherChangeMinute(Number(event.target.value)); invalidate(); }} /></label>
                </div>
                {cityWeather && <p className="hint">{origin?.city ?? 'Manila'} at {cityWeather.localtime.slice(11)}: {cityWeather.description.join(', ')}, {cityWeather.precip_mm} mm. <button type="button" className="link-button" onClick={updateWeather}>{refreshing ? 'Reloading…' : 'Reload'}</button></p>}
                {weatherError && <p className="error" role="status">{weatherError}</p>}
              </div>
              <div className="form-section">
                <h3>Road closures</h3>
                <p className="hint">Choose “Close road” above the map, then click a road. Closures are hypothetical scenarios.</p>
                <label className="checkbox"><input type="checkbox" checked={bothDirections} onChange={event => setBothDirections(event.target.checked)} />Close both directions</label>
                {closures.length > 0 ? <ul className="closure-list">{closures.map(road => <li key={road.id}><span>{road.name}</span><button type="button" aria-label={`Reopen ${road.name}`} onClick={() => { setClosures(previous => previous.filter(item => item.id !== road.id)); invalidate(); }}>Reopen</button></li>)}</ul>
                  : <p className="hint">No roads closed.</p>}
                {closures.length > 0 && <div className="field-row">
                  <label>Takes effect<select value={closureMinute === null ? 'departure' : 'during'} onChange={event => { setClosureMinute(event.target.value === 'departure' ? null : 5); invalidate(); }}>
                    <option value="departure">At departure</option><option value="during">During the trip</option>
                  </select></label>
                  <label>At minute<input type="number" min={0} max={1440} step={1} required disabled={closureMinute === null} value={closureMinute ?? 0} onChange={event => { setClosureMinute(Number(event.target.value)); invalidate(); }} /></label>
                </div>}
              </div>
              <button type="submit">{busy ? 'Running ANTROUTE…' : baselineBusy ? 'Running baseline…' : 'Calculate route'}</button>
            </fieldset>
          </form>
          {error && <p className="error" role="alert">{error}</p>}
        </aside>
      <main className="content">
        <div className="page-title"><div><h1>ROUTE PLANNER</h1><p>{cities.join(' · ')}</p></div><span className="badge">Five-city study</span></div>
        <section className="card map-card" id="route-map">
          <div className="map-head">
            <h2>Route within the study area</h2>
            <div className="map-toolbar" role="group" aria-label="Map selection mode">{(['origin', 'destination', 'closure'] as const).map(mode => <button type="button" key={mode} disabled={disabled} aria-pressed={pickMode === mode} onClick={() => setPickMode(mode)}>{mode === 'closure' ? 'Close road' : `Set ${mode}`}</button>)}</div>
          </div>
          <p className="muted" role="status">{picking ? 'Finding the nearest study road…' : `Click the map to ${pickMode === 'closure' ? 'enter a road closure' : `choose the ${pickMode}`}.`}</p>
          <div className="map">
            <RouteMap geometry={result?.route.geometry ?? emptyGeometry} baselineGeometry={baseline?.route.geometry ?? emptyGeometry} boundaries={network?.boundaries ?? emptyBoundaries} origin={origin} destinations={chosenStops} closures={closures} datasetMap={datasetMap} explanation={explanation} focusedStretch={focusedStretch} onFocusStretch={setFocusedStretch} onPick={pickPoint} disabled={disabled} />
            <ul className="map-legend" aria-label="Map legend">
              <li><i className="line" style={{ background: ROUTE_COLOR }} />ANTROUTE route{result ? ` · ${(result.evaluation.distance_travel_m / 1000).toFixed(2)} km` : ''}</li>
              <li><i className="line" style={{ background: BASELINE_COLOR }} />Baseline IACO{baseline ? ` · ${(baseline.evaluation.distance_travel_m / 1000).toFixed(2)} km` : baselineBusy ? ' · running…' : ''}</li>
              <li><i className="line" style={{ background: '#dc2626' }} />Road closure</li>
              <li><i className="dot" style={{ background: '#15803d' }} />Start location</li>
              <li><i className="dot" style={{ background: '#be123c' }} />Destinations, in order</li>
              {explanation && explanation.stretches.length > 0 && <><li><i className="line dashed" style={{ borderColor: MISSED_COLOR }} />Better road the ants never found</li><li><i className="line dashed" style={{ borderColor: COSTLIER_COLOR }} />Shorter road, worse on the ten features</li></>}
            </ul>
          </div>
          {datasetMap && <><p className="muted">{datasetMap.segment_count.toLocaleString()} road segments included in the study area, colored by city.</p><div className="feature-list">{datasetMap.bands.map(band => <span key={band.label} style={{ color: band.color }}>{band.label}: {band.lines.length.toLocaleString()}</span>)}</div></>}
          <p className="muted">{result ? `${result.route.road_ids.length} road segments driven by ANTROUTE, including any re-routing.` : 'Search road names or select points on the map.'} Blue boundaries mark the study area; red segments are closed.</p>
          {explanation && <RouteExplanationPanel explanation={explanation} note={explanationNote} hour={hour} onShow={index => { setFocusedStretch(index); document.getElementById('route-map')?.scrollIntoView({ behavior: 'smooth' }); }} />}
          <p className="attribution">Roads and boundaries © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap contributors</a>. Flood hazard source: Project NOAH.</p>
        </section>
      {runs.length > 0 && <section className="card route-summary">
        <h2>Route summary</h2>
          <div className="route-cards">
            {runs.map((run, index) => <article key={run.name} className={index === 0 ? 'route-card proposed' : 'route-card'}>
              <span className="route-badge" style={{ color: run.color }}>{index === 0 ? 'Proposed route' : 'Baseline route'}</span>
              <h4>{run.name}</h4>
              <p className="route-via">Via {mainRoad(run.trip)}</p>
              <dl>
                <div><dt>Travel time</dt><dd>{(run.trip.trip.travel_time_s / 60).toFixed(0)} min</dd></div>
                <div><dt>Distance</dt><dd>{(run.trip.evaluation.distance_travel_m / 1000).toFixed(1)} km</dd></div>
                <div><dt>Path cost</dt><dd>{run.trip.evaluation.path_cost.toFixed(1)}</dd></div>
              </dl>
              <a className={index === 0 ? 'button' : 'button ghost'} href="#trip-measures">View details</a>
            </article>)}
            {!baseline && <article className="route-card pending"><span className="route-badge" style={{ color: BASELINE_COLOR }}>Baseline route</span><h4>Baseline IACO</h4><p className="route-via">{baselineBusy ? 'Running on the same trip…' : baselineError ? 'Not available' : '–'}</p></article>}
          </div>
      </section>}
      <section className="card evaluation" id="evaluation">
        <h2>Evaluation of this trip</h2>
        {runs.length ? <>
          <p className="muted" id="trip-measures">Both algorithms drive the same trip, departure and mid-trip changes. Path cost is the CRITIC ten-feature cost of each road when entered, for both. Convergence rate is the iteration at which the initial search first found its final best route (fewer = faster). Lower is better for every measure.</p>
          <div className="audit-table"><table><thead><tr><th>Measure</th>{runs.map(run => <th key={run.name}><span className="legend-dot" style={{ background: run.color }} />{run.name}</th>)}{!baseline && <th>Baseline IACO</th>}</tr></thead><tbody>
            {([
              ['Distance travelled', trip => `${(trip.evaluation.distance_travel_m / 1000).toFixed(3)} km`],
              ['Path cost', trip => trip.evaluation.path_cost.toFixed(4)],
              ['Computational time', trip => seconds(trip.evaluation.computational_time_ms)],
              ['Convergence rate', trip => `${trip.evaluation.convergence.iterations_to_convergence} iteration${trip.evaluation.convergence.iterations_to_convergence === 1 ? '' : 's'} (all searches: ${trip.evaluation.convergence.mean_iterations_to_convergence.toFixed(1)})`],
            ] as [string, (trip: TripResult) => string][]).map(([name, show]) => <tr key={name}><th scope="row">{name}</th>
              {runs.map(run => <td key={run.name}>{show(run.trip)}</td>)}{!baseline && <td>{baselineBusy ? 'Running…' : '–'}</td>}</tr>)}
          </tbody></table></div>
          {baselineError && <p className="error" role="alert">Baseline: {baselineError}</p>}
          <CostBreakdown runs={runs} />
          <div className="chart-pair">{runs.map(run => <CostHistory key={run.name} caption={`${run.name}: initial search`} color={run.color}
            values={run.trip.evaluation.convergence.searches[0].best_cost_by_iteration} marker={run.trip.evaluation.convergence.iterations_to_convergence} />)}</div>
          <h3>Routing and re-routing log</h3>
          <div className="audit-table"><table><thead><tr><th>Algorithm</th><th>Trip minute</th><th>Reason</th><th>Roads affected</th><th>Iterations to convergence</th><th>Routing time</th></tr></thead><tbody>
            {runs.flatMap(run => run.trip.evaluation.convergence.searches.map((search, index) => <tr key={`${run.name}-${index}`}>
              <td>{run.name}</td><td>{search.at_minute.toFixed(1)}</td><td>{search.trigger === 'departure' ? 'Initial routing at departure' : search.kind === 'initial' ? `Initial routing: ${search.trigger.replace(/^leg/, 'Leg')}` : search.trigger}</td>
              <td>{search.affected_roads ? search.affected_roads.toLocaleString() : '–'}</td><td>{search.iterations_to_convergence} of {search.iterations}</td><td>{seconds(search.routing_time_ms)}</td></tr>))}
          </tbody></table></div>
          {runs.some(run => run.trip.trip.changes_without_replanning.length > 0) && <p className="muted">Changes that did not require re-routing: {runs.map(run => run.trip.trip.changes_without_replanning.map(change => `${run.name} at minute ${change.at_minute.toFixed(1)} (${change.change})`).join('; ')).filter(Boolean).join('; ')}.</p>}
          {result && <details><summary>Roads driven by ANTROUTE</summary><p>{result.route.road_names.filter((name, index, array) => index === 0 || name !== array[index - 1]).join(' → ')}</p></details>}
        </> : <p className="muted">Distance travelled, path cost, computational time and convergence rate appear for both algorithms after a completed trip.</p>}
      </section>
      <section className="card" id="features"><h2>Routing features</h2><div className="feature-list">{features.map(feature => <span key={feature}>{feature}</span>)}</div>
        <details className="source-details"><summary>Where each feature and estimate comes from</summary>
          <div className="audit-table"><table><thead><tr><th>Feature</th><th>Classification</th><th>Basis and assumptions</th><th>Reference</th></tr></thead><tbody>
            {Object.entries(network?.data_quality.estimation_registry ?? {}).map(([name, entry]) => <tr key={name}><th scope="row">{name.replaceAll('_', ' ')}</th><td>{entry.classification.replaceAll('_', ' ')}</td><td>{entry.basis}</td><td>{entry.source.startsWith('https://') ? <a href={entry.source} target="_blank" rel="noreferrer">Source</a> : entry.source}</td></tr>)}
          </tbody></table></div>
        </details>
        <details className="source-details"><summary>Data sources and estimate limitations</summary><ul>{network?.data_quality.limitations.map(item => <li key={item}>{item}</li>)}</ul></details>
      </section>
      <div id="dataset">{network && <DatasetExplorer conditions={conditions} onMap={setDatasetMap} disabled={disabled} />}</div>
      </main>
    </div>
  </>;
}
