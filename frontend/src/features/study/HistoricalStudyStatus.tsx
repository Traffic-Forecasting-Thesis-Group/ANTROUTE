import type { HistoricalStudyStatus as Status } from '../../api/routeService';

const labels: Record<string, string> = {
  distance: 'Distance', congestion: 'Congestion', travel_time: 'Travel Time',
  intersections: 'Intersections', signals: 'Traffic Signals', road_type: 'Road Type',
  time_profile: 'Time Profile', weather: 'Weather', flood_exposure: 'Flood Exposure', capacity_burden: 'Capacity Burden',
};

export default function HistoricalStudyStatus({ status }: { status: Status }) {
  return <>
    <header><div className="brand">ANTROUTE</div><span>Historical ACO Study</span></header>
    <main>
      <div className="page-title"><div><h1>Historical study data requirements</h1><p>{status.requirements.display_period} · Asia/Manila</p></div><span className="badge">Hourly · {status.requirements.allow_estimates ? 'Documented estimates allowed' : 'No estimates'}</span></div>
      <section className="card" role="status"><h2>Historical dataset is not ready</h2>
        <p>{status.message}</p>
        <p>{status.requirements.allow_estimates ? 'Rebuild or verify the documented estimation release before routing. Estimates must remain labeled and traceable.' : 'Route calculation is disabled until verified historical inputs are acquired.'}</p>
        <p><strong>{status.required_hours_per_entity.toLocaleString()} hourly timestamps</strong> are required across the study period for each covered dynamic-data entity. Road inventories need documented historical validity intervals.</p>
        <p>Scope: San Juan, Manila, Mandaluyong, Makati, and Pasig.</p>
        <button disabled type="button">Route calculation unavailable</button>
      </section>
      <section className="card"><h2>Ten-feature data audit</h2>
        <div className="audit-table"><table><thead><tr><th>Feature</th><th>Local evidence and gaps</th><th>Required historical evidence</th></tr></thead>
          <tbody>{Object.entries(status.audit?.feature_audit ?? {}).map(([key, entry]) => <tr key={key}><th scope="row">{labels[key] ?? key}</th><td>{entry.local_evidence}</td><td>{entry.required_evidence}</td></tr>)}</tbody>
        </table></div>
        <p className="muted">Source coverage, measurement status, and every permitted estimation method must be disclosed. A model output is not an observed historical record.</p>
      </section>
    </main>
  </>;
}
