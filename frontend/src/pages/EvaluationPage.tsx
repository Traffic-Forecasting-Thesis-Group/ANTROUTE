import { useCallback, useEffect, useState } from 'react';
import { RotateCcw } from 'lucide-react';

import { getComparisonMetrics, type ComparisonMetrics } from '@/api/routeService';
import PageHeader from '@/app/PageHeader';
import { Button, Card, EmptyState, Spinner } from '@/components/ui';
import { cn } from '@/lib/cn';
import MetricsTable, { MetricsFootnote } from '@/features/metrics/MetricsTable';

/**
 * The thesis routing evaluation over every test trip timed in Google Maps (Section 3.9,
 * Appendix 3): ANTRoute against the baseline Improved ACO on Route Optimality and the
 * congestion and event exposures, overall and per scenario. Travel time is not among them.
 * A single trip's own evaluation is on the Comparison page.
 */
export default function EvaluationPage() {
  const [data, setData] = useState<ComparisonMetrics | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [scenario, setScenario] = useState('overall');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setData(await getComparisonMetrics());
    } catch (e: any) {
      setError(e?.message || 'Could not load the evaluation.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const shown = data?.scenarios.find((s) => s.scenario === scenario) ?? data?.scenarios[0];
  const total = data?.scenarios[0]?.nTrials ?? 0;

  return (
    <>
      <PageHeader
        title="Evaluation"
        subtitle={data?.description}
        actions={
          <Button variant="secondary" onClick={load} disabled={loading}>
            {loading ? <Spinner /> : <RotateCcw size={15} aria-hidden />}
            Refresh
          </Button>
        }
      />

      <div className="flex-1 overflow-y-auto p-6">
        {data && data.scenarios.length > 0 && (
          <div className="mb-5 flex flex-wrap gap-2">
            {data.scenarios.map((s) => {
              const isActive = s.scenario === shown?.scenario;
              return (
                <button
                  key={s.scenario}
                  type="button"
                  onClick={() => setScenario(s.scenario)}
                  className={cn(
                    'rounded-full border px-3.5 py-1.5 text-xs font-medium transition-colors',
                    isActive
                      ? 'border-brand bg-brand text-white'
                      : 'border-slate-300 bg-white text-slate-600 hover:bg-slate-50'
                  )}
                >
                  {s.label} ({s.nTrials})
                </button>
              );
            })}
          </div>
        )}

        {loading && !data ? (
          <div className="flex justify-center py-20">
            <Spinner className="h-8 w-8 text-brand" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center gap-4 py-16">
            <EmptyState title="Could not load the evaluation" description={error} />
            <Button onClick={load}>Retry</Button>
          </div>
        ) : shown ? (
          <Card className="max-w-4xl p-6">
            <h2 className="text-lg font-bold text-slate-900">
              {shown.scenario === 'overall' ? `All ${total} Test Trips` : `${shown.label} Trips`}
            </h2>
            <p className="mb-5 mt-1 text-sm leading-relaxed text-slate-500">
              ANTRoute against the baseline ({data?.baselineName}) on {shown.nTrials} test
              trips, both routed on the same stops and departure times and both timed in
              Google Maps.
            </p>

            <MetricsTable rows={shown.metrics} baselineName={data?.baselineName} />
            <MetricsFootnote />
          </Card>
        ) : (
          <EmptyState
            title="No evaluation available"
            description="The server has no scored test trips loaded yet."
          />
        )}
      </div>
    </>
  );
}
