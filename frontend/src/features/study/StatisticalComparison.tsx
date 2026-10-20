import { useEffect, useState } from 'react';
import { loadSopResults, type SopResults } from '../../api/routeService';

const pValue = (p: number | null) => p === null ? '–' : p < 0.0001 ? '< 0.0001' : p.toFixed(4);
const ranks = (value?: number) => value === undefined ? '–' : value.toFixed(1);

/** SOP 3 from scripts/evaluate_sop.py: paired d = baseline − ANTROUTE, Wilcoxon signed-rank, Holm-adjusted p-values. */
export default function StatisticalComparison() {
  const [results, setResults] = useState<SopResults | null>(null);
  const [error, setError] = useState('');
  const [family, setFamily] = useState('all');
  useEffect(() => {
    const controller = new AbortController();
    loadSopResults(controller.signal).then(setResults).catch(reason => { if (!controller.signal.aborted) setError(reason.message); });
    return () => controller.abort();
  }, []);
  if (error) return <section className="card" id="statistics"><h2>Statistical comparison</h2><p className="muted">{error}</p></section>;
  if (!results) return null;
  const tests = results.sop3[family];
  return <section className="card" id="statistics">
    <h2>Statistical comparison</h2>
    <p className="muted">Paired by trip and traffic condition over {results.trips} trips × {Object.keys(results.scenarios).length} conditions × {results.seeds} seeds (seeds averaged).
      Paired difference d<sub>i</sub> = C<sub>baseline,i</sub> − C<sub>ANTROUTE,i</sub>, so a positive d favours ANTROUTE. Two-sided Wilcoxon signed-rank test with
      zero differences excluded and tied ranks averaged, W = min(W⁺, W⁻). The four p-values of each condition are Holm-adjusted; a difference is significant
      when the adjusted p &lt; α = {results.alpha}. A single trip above cannot be tested on its own.</p>
    <label>Traffic condition<select value={family} onChange={event => setFamily(event.target.value)}>
      <option value="all">All conditions</option>
      {Object.entries(results.scenarios).map(([key, name]) => <option key={key} value={key}>{name}</option>)}
    </select></label>
    <div className="audit-table"><table><thead><tr><th>Measure</th><th>n</th><th>ANTROUTE mean</th><th>Baseline mean</th><th>Mean d</th>
      <th>W⁺</th><th>W⁻</th><th>W</th><th>p</th><th>Holm p</th><th>Decision</th></tr></thead><tbody>
      {results.sop_measures['3'].map(measure => {
        const { label, unit, scale } = results.measures[measure];
        const test = tests[measure];
        const show = (value?: number) => value === undefined ? '–' : `${(value * scale).toFixed(unit === 'iterations' ? 2 : unit ? 3 : 4)}${unit && unit !== 'iterations' ? ` ${unit}` : ''}`;
        return <tr key={measure}><th scope="row">{label.split(':')[0]}</th><td>{test.n}</td>
          <td>{show(test.proposed_mean)}</td><td>{show(test.baseline_mean)}</td><td>{show(test.mean_difference)}</td>
          <td>{ranks(test.W_plus)}</td><td>{ranks(test.W_minus)}</td><td>{ranks(test.W)}</td>
          <td>{pValue(test.p)}</td><td>{pValue(test.p_holm)}</td>
          <td>{test.significant === null ? test.test : test.significant ? <strong>Significant · reject H₀</strong> : 'Not significant · fail to reject H₀'}</td></tr>;
      })}
    </tbody></table></div>
    {results.not_converged && <p className="muted">Runs that reached the iteration limit without converging: ANTROUTE {results.not_converged[family].antroute},
      baseline {results.not_converged[family].cheng}. {tests.iterations_to_convergence?.pairs_excluded_not_converged
        ? `${tests.iterations_to_convergence.pairs_excluded_not_converged} pair(s) were left out of the convergence-rate test.` : ''}</p>}
    <p className="muted">From outputs/sop_results.json, generated {new Date(results.generated_at).toLocaleString()}. Re-run with python -m scripts.evaluate_sop.</p>
  </section>;
}
