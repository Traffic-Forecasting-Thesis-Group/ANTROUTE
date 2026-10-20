import { useEffect, useState } from 'react';
import { loadSopResults, type SopResults } from '../../api/routeService';

const pValue = (p: number | null) => p === null ? '–' : p < 0.0001 ? '< 0.0001' : p.toFixed(4);
const ranks = (value?: number) => value === undefined ? '–' : value.toFixed(1);
const NAMES = { antroute: 'ANTROUTE', cheng: 'Baseline' } as const;

/** Primary comparison from scripts/evaluate_sop.py: paired d = baseline − ANTROUTE over request medians, Wilcoxon, Holm (m = 2). */
export default function StatisticalComparison() {
  const [results, setResults] = useState<SopResults | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    loadSopResults(controller.signal).then(setResults).catch(reason => { if (!controller.signal.aborted) setError(reason.message); });
    return () => controller.abort();
  }, []);
  if (error) return <section className="card" id="statistics"><h2>Statistical comparison</h2><p className="muted">{error}</p></section>;
  if (!results?.analysis) return null;
  const { analysis } = results;
  return <section className="card" id="statistics">
    <h2>Statistical comparison</h2>
    <p className="muted">Frozen experiment manifest of {analysis.requests} requests on held-out test-period snapshots; {analysis.complete_requests} have a route
      from both algorithms on every seed ({analysis.seeds.join(', ')}) and form the pairs. Each request's value is the median over its seeds.
      Paired difference d<sub>i</sub> = C<sub>baseline,i</sub> − C<sub>ANTROUTE,i</sub>, so a positive d favours ANTROUTE. Two-sided Wilcoxon signed-rank
      (zero differences dropped, tied ranks averaged), W = min(W⁺, W⁻), Holm-adjusted over the two primary outcomes; significant when the adjusted
      p &lt; α = {results.alpha}. Traffic is an assumed scenario: this is a modeled comparison, not empirical validation.</p>
    <div className="audit-table"><table><thead><tr><th>Primary outcome</th><th>n</th><th>ANTROUTE median</th><th>Baseline median</th><th>Median d</th>
      <th>W⁺</th><th>W⁻</th><th>W</th><th>p</th><th>Holm p</th><th>Decision</th></tr></thead><tbody>
      {Object.entries(results.primary_measures).map(([measure, { label, unit }]) => {
        const test = analysis.primary[measure];
        const show = (value: number | null) => value === null ? '–' : `${value.toFixed(unit ? 2 : 4)}${unit ? ` ${unit}` : ''}`;
        return <tr key={measure}><th scope="row">{label}</th><td>{test.n}</td>
          <td>{show(test.proposed_median)}</td><td>{show(test.baseline_median)}</td><td>{show(test.median_difference)}</td>
          <td>{ranks(test.W_plus)}</td><td>{ranks(test.W_minus)}</td><td>{ranks(test.W)}</td>
          <td>{pValue(test.p)}</td><td>{pValue(test.p_holm)}</td>
          <td>{test.significant === null ? test.test : test.significant ? <strong>Significant · reject H₀</strong> : 'Not significant · fail to reject H₀'}</td></tr>;
      })}
    </tbody></table></div>
    <p className="muted">All requests by outcome: {(Object.keys(NAMES) as (keyof typeof NAMES)[]).map(algorithm =>
      `${NAMES[algorithm]} ${Object.entries(analysis.outcomes[algorithm] ?? {}).map(([status, count]) => `${status.replaceAll('_', ' ')} ${count}`).join(', ')}`).join('; ')}.
      Failures are reported, never counted as zero cost or time.</p>
    <p className="muted">From outputs/sop_results.json, generated {new Date(results.generated_at).toLocaleString()} (scipy {results.scipy_version}).</p>
  </section>;
}
