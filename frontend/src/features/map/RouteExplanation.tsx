import type { RouteExplanation as Explanation, RouteStretch, StretchFacts } from '../../api/routeService';

export const MISSED_COLOR = '#ea580c';
export const COSTLIER_COLOR = '#475569';
export const stretchColor = (stretch: RouteStretch) => stretch.verdict === 'missed_by_search' ? MISSED_COLOR : COSTLIER_COLOR;
export const stretchLetter = (index: number) => String.fromCharCode(65 + index);

const metres = (value: number) => value >= 1000 ? `${(value / 1000).toFixed(2)} km` : `${Math.round(value)} m`;
const names = (facts: StretchFacts) => facts.road_names.length ? facts.road_names.slice(0, 3).join(', ') + (facts.road_names.length > 3 ? ', …' : '') : 'unnamed road';
// What a higher cost on each feature means on the road, in words.
const WORSE: Record<string, string> = {
  distance: 'longer distance', congestion: 'more congested length', travel_time: 'longer travel time',
  intersections: 'more intersections', signals: 'more traffic signals', road_type: 'more length on less preferred road classes',
  time_profile: 'more length on roads that are usually busy at this hour',
  weather: 'more length in bad weather', flood_exposure: 'more length in mapped flood hazard areas', capacity_burden: 'a higher capacity burden',
};

type Row = [string, (facts: StretchFacts) => string, ((road: StretchFacts, taken: StretchFacts) => boolean)?];
// The third item marks this road's value when it is worse than ANTROUTE's.
const ROWS: Row[] = [
  ['Roads', facts => names(facts)],
  ['Length', facts => metres(facts.distance_m), (road, taken) => road.distance_m > taken.distance_m],
  ['Road type', facts => facts.road_types.join(', ').replaceAll('_', ' ')],
  ['Intersections', facts => String(facts.intersections), (road, taken) => road.intersections > taken.intersections],
  ['Traffic signals', facts => String(facts.signals), (road, taken) => road.signals > taken.signals],
  ['Congestion', facts => facts.congestion.toFixed(3), (road, taken) => road.congestion > taken.congestion],
  ['Weather', facts => facts.weather_severity.toFixed(2), (road, taken) => road.weather_severity > taken.weather_severity],
  ['Travel time', facts => `${(facts.travel_time_s / 60).toFixed(1)} min`, (road, taken) => road.travel_time_s > taken.travel_time_s],
  ['Flood severity', facts => `${facts.flood_severity.toFixed(2)} (max ${facts.max_flood_severity.toFixed(2)})`, (road, taken) => road.flood_severity > taken.flood_severity],
  ['Capacity', facts => `${Math.round(facts.capacity_vph).toLocaleString()} veh/h`, (road, taken) => road.capacity_vph < taken.capacity_vph],
  ['Path cost', facts => facts.cost.toFixed(2), (road, taken) => road.cost > taken.cost],
];

const brief = (text: string) => text.replace(/ \(max [^)]*\)/, '');
const WORSE_ROW: Record<string, string> = {
  Length: 'a longer length', Intersections: 'more intersections', 'Traffic signals': 'more traffic signals', Congestion: 'heavier congestion',
  'Travel time': 'a longer travel time', 'Flood severity': 'a higher length-weighted flood severity', Capacity: 'lower capacity',
  Weather: 'a higher weather severity',
};

export const roadName = (stretch: RouteStretch) => stretch.alternative.road_names[0] ?? 'Unnamed road';

/** The road's title and a plain reason it was not on ANTROUTE's route. */
export function stretchReason(stretch: RouteStretch, antRoutes: number): { title: string; detail: string } {
  const length = stretch.distance_saved_m > 0 ? `${metres(stretch.distance_saved_m)} shorter` : `${metres(-stretch.distance_saved_m)} longer`;
  const title = `${roadName(stretch)}: not chosen`;
  if (stretch.verdict === 'missed_by_search') {
    const used = stretch.ant_routes_using_alternative === 0 ? `none of the ${antRoutes} ant routes reached it, so the colony never learned it was better`
      : `although ${stretch.ant_routes_using_alternative} of ${antRoutes} ant routes used it, none of them completed a cheaper whole route through it than the one the colony kept`;
    // Same comparisons that mark the table's red cells, so the sentence and the table agree.
    const worse = ROWS.filter(([name, , isWorse]) => name !== 'Path cost' && isWorse?.(stretch.alternative, stretch.chosen))
      .map(([name, show]) => `${WORSE_ROW[name] ?? `worse ${name.toLowerCase()}`} (${brief(show(stretch.alternative))} vs ${brief(show(stretch.chosen))})`);
    const tradeoff = worse.length ? `It has ${worse.join(', ')}, but that is outweighed: overall` : 'Nothing on this road makes it worse:';
    return { title, detail: `${tradeoff} it is ${length} and scores ${Math.abs(stretch.cost_difference).toFixed(2)} lower on the ten features than the way ANTROUTE went. It was left out because ${used}. That is a gap in the ACO search, not a judgement on the road.` };
  }
  const causes = Object.entries(stretch.feature_differences).filter(([, value]) => value > 1e-6).slice(0, 3).map(([name]) => WORSE[name] ?? name);
  return { title, detail: `This road is ${length}, but it scores ${stretch.cost_difference.toFixed(2)} higher on the ten features, mainly because of ${causes.join(', ') || 'small differences across features'}. ANTROUTE's way scores lower overall, so the ants preferred it.` };
}

/** What is on the road ANTROUTE left out, next to what it took instead, and why. */
export function StretchCard({ stretch, letter, antRoutes, hour }: { stretch: RouteStretch; letter: string; antRoutes: number; hour?: number }) {
  const reason = stretchReason(stretch, antRoutes);
  return <div className="stretch-card">
    <strong>{letter}. {reason.title}</strong>
    <table><thead><tr><th /><th>This road</th><th>ANTROUTE took</th></tr></thead><tbody>
      {ROWS.map(([name, show, worse]) => <tr key={name}><th scope="row">{name}{name === 'Congestion' && hour !== undefined ? ` at ${String(hour).padStart(2, '0')}:00` : ''}</th>
        <td className={worse?.(stretch.alternative, stretch.chosen) ? 'worse' : undefined}>{show(stretch.alternative)}</td>
        <td>{show(stretch.chosen)}</td></tr>)}
    </tbody></table>
    <p><b>Why it was not chosen:</b> {reason.detail}</p>
  </div>;
}

/** Every road left out, matching the lettered markers on the map. */
export default function RouteExplanationPanel({ explanation, note, hour, onShow }: { explanation: Explanation; note?: string; hour?: number; onShow?: (index: number) => void }) {
  const { stretches, references } = explanation;
  return <div className="route-explanation">
    <h3>Roads ANTROUTE did not take</h3>
    <p className="muted">Dashed roads on the map are shorter or cheaper options on the same date and hour, taken from the shortest-distance and lowest-cost routes (Dijkstra, used only for this comparison).{' '}
      <span style={{ color: MISSED_COLOR }}>Orange</span>: better on the ten features but never found by the ants.{' '}
      <span style={{ color: COSTLIER_COLOR }}>Grey</span>: shorter, but worse on the ten features. Red values are where the road is worse than ANTROUTE's way.{note ? ` ${note}` : ''}</p>
    {stretches.length === 0
      ? <p className="muted">ANTROUTE's route matches {references.lowest_cost?.same_as_chosen ? 'the lowest-cost route' : 'the reference routes'}{references.shortest_distance?.same_as_chosen ? ' and the shortest-distance route' : ''}; no shorter road was left out.</p>
      : <ol className="stretch-list">{stretches.map((stretch, index) => <li key={index} style={{ borderColor: stretchColor(stretch) }}>
        <span className="stretch-letter" style={{ background: stretchColor(stretch) }}>{stretchLetter(index)}</span>
        <div><StretchCard stretch={stretch} letter={stretchLetter(index)} antRoutes={explanation.ant_routes} hour={hour} />
          {onShow && <button type="button" className="link-button" onClick={() => onShow(index)}>Show on map</button>}</div>
      </li>)}</ol>}
  </div>;
}
