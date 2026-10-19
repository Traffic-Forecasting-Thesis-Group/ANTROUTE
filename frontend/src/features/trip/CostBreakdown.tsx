import type { FeatureContribution, TripResult } from '../../api/routeService';

const label = (text: string) => text.replaceAll('_', ' ');

/** The route's raw value for one feature, in the unit a reader expects. */
function routeValue(item: FeatureContribution): string {
  const value = item.route_value;
  if (Array.isArray(value)) return `${value.map(hour => `${String(hour).padStart(2, '0')}:00`).join(', ')} · load avg ${item.normalized_mean.toFixed(2)}`;
  if (typeof value === 'number') {
    if (item.feature === 'distance') return `${(value / 1000).toFixed(2)} km`;
    if (item.feature === 'travel_time') return `${(value / 60).toFixed(1)} min`;
    return `${value} ${value === 1 ? item.unit.replace(/s$/, '') : item.unit}`;
  }
  if ('mean' in value && 'max' in value && Object.keys(value).length === 2) {
    if (item.feature === 'road_capacity') return `avg ${Math.round(value.mean).toLocaleString()} veh/h`;
    return `avg ${value.mean.toFixed(3)} · max ${value.max.toFixed(3)}`;
  }
  const mix = Object.entries(value as Record<string, number>);
  return mix.slice(0, 3).map(([name, count]) => `${label(name)} ${count}`).join(', ') + (mix.length > 3 ? ', …' : '');
}

function sources(item: FeatureContribution): string {
  return Object.entries(item.data_sources).map(([name, count]) => `${label(name)} (${count})`).join('; ');
}

/** Each feature's share of path cost for every algorithm, with the data source behind it. */
export default function CostBreakdown({ runs }: { runs: { name: string; color: string; trip: TripResult }[] }) {
  const withData = runs.filter(run => run.trip.contributions?.length);
  if (!withData.length) return null;
  const [first] = withData;
  return <>
    <h3>What made up the path cost</h3>
    <p className="muted">Path cost = Σ over the roads driven of (CRITIC weight × normalized feature value). Each row is one feature's part of that sum, so the
      contributions add up to the path cost. Values are scored when each road is entered. Data sources count the roads (or road-end intersections) on the route that took each source.</p>
    <div className="audit-table"><table className="cost-breakdown"><thead><tr><th>Feature</th><th>CRITIC weight</th>
      {withData.map(run => <th key={run.name}><span className="legend-dot" style={{ background: run.color }} />{run.name}: route value · contribution</th>)}
      <th>Where it came from</th></tr></thead><tbody>
      {first.trip.contributions.map((item, row) => <tr key={item.feature}>
        <th scope="row">{label(item.feature)}</th>
        <td>{item.weight.toFixed(3)}</td>
        {withData.map(run => {
          const own = run.trip.contributions[row];
          return <td key={run.name}>
            <div>{routeValue(own)}</div>
            <div className="share"><span className="share-bar" style={{ width: `${Math.max(own.share * 100, 1)}%`, background: run.color }} />
              <span>{own.contribution.toFixed(2)} ({(own.share * 100).toFixed(1)}%)</span></div>
          </td>;
        })}
        <td><strong>{label(item.classification)}</strong>
          <div className="muted">{withData.map(run => `${run.name}: ${sources(run.trip.contributions[row])}`).join(' | ')}</div>
          <details><summary>Basis</summary><p>{item.basis}</p>
            {item.reference && (item.reference.startsWith('https://')
              ? <a href={item.reference} target="_blank" rel="noreferrer">Reference</a> : <p>{item.reference}</p>)}</details></td>
      </tr>)}
      <tr><th scope="row">Path cost</th><td>–</td>{withData.map(run => <td key={run.name}><strong>{run.trip.contributions.reduce((sum, item) => sum + item.contribution, 0).toFixed(2)}</strong></td>)}<td /></tr>
    </tbody></table></div>
  </>;
}
