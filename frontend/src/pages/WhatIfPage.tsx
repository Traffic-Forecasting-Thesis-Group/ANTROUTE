import { useCallback, useEffect, useState } from 'react';
import { RotateCcw } from 'lucide-react';

import { getWhatIf, type WhatIf } from '@/api/routeService';
import PageHeader from '@/app/PageHeader';
import { Button, Card, EmptyState, Spinner } from '@/components/ui';

const pct = (x: number) => `${(x * 100).toFixed(x >= 0.995 || x === 0 ? 0 : 1)}%`;
const num = (x: number) => x.toFixed(3);

/**
 * What-if cases, precomputed: the real checkpoint is re-run with one input changed at a time
 * (rain, flood, new incidents, cameras off, ...) and the table shows how far the Congestion Risk
 * Score moves. It shows the model accepts weather and events; it is offline re-inference, not a
 * live run, and the cases are not a forecast.
 */
export default function WhatIfPage() {
  const [data, setData] = useState<WhatIf | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setData(await getWhatIf());
    } catch (e: any) {
      setError(e?.message || 'Could not load the what-if table.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const effect = data?.route_effect;

  return (
    <>
      <PageHeader
        title="What-if"
        subtitle="Change one input of the model and see how far the risk moves. Precomputed from the checkpoint."
        actions={
          <Button variant="secondary" onClick={load} disabled={loading}>
            {loading ? <Spinner /> : <RotateCcw size={15} aria-hidden />}
            Refresh
          </Button>
        }
      />

      <div className="flex-1 overflow-y-auto p-6">
        {loading && !data ? (
          <div className="flex justify-center py-20">
            <Spinner className="h-8 w-8 text-brand" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center gap-4 py-16">
            <EmptyState title="Could not load the what-if table" description={error} />
            <Button onClick={load}>Retry</Button>
          </div>
        ) : data ? (
          <div className="flex max-w-5xl flex-col gap-6">
            <Card className="p-6">
              <h2 className="text-lg font-bold text-slate-900">Risk response by input</h2>
              <p className="mb-5 mt-1 text-sm leading-relaxed text-slate-500">
                {data.model}, {data.windows} recorded windows. {data.note}
              </p>

              <div className="overflow-hidden rounded-lg border border-slate-200">
                <table className="w-full border-collapse text-sm">
                  <thead>
                    <tr className="bg-slate-50 text-left">
                      <th className="px-4 py-2.5 font-semibold text-slate-600">Case</th>
                      <th className="px-4 py-2.5 text-right font-semibold text-slate-600">
                        Camera roads
                      </th>
                      <th className="px-4 py-2.5 text-right font-semibold text-slate-600">
                        Other roads
                      </th>
                      <th className="px-4 py-2.5 text-right font-semibold text-slate-600">
                        Other roads moved
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.cases.map((c) => (
                      <tr key={c.key} className="border-t border-slate-200 align-top">
                        <td className="px-4 py-2.5">
                          <p className="font-medium text-slate-800">{c.label}</p>
                          <p className="text-xs text-slate-500">{c.description}</p>
                        </td>
                        <td className="px-4 py-2.5 text-right tabular-nums text-slate-800">
                          {num(c.camera_mean)}
                          <span className="block text-[11px] text-slate-400">max {num(c.camera_max)}</span>
                        </td>
                        <td className="px-4 py-2.5 text-right tabular-nums text-slate-800">
                          {num(c.other_mean)}
                          <span className="block text-[11px] text-slate-400">max {num(c.other_max)}</span>
                        </td>
                        <td className="px-4 py-2.5 text-right tabular-nums text-slate-800">
                          {pct(c.other_moved)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-3 text-xs leading-relaxed text-slate-500">
                Mean absolute change in risk (0&ndash;1 scale) when only that input changes. Camera
                roads are the edges the model is supervised on; other roads have no camera, so a
                change there shows weather, flood, incidents and time of day reaching them.
              </p>
            </Card>

            {effect && (
              <Card className="p-6">
                <h2 className="text-lg font-bold text-slate-900">What a new incident does to a route</h2>
                <p className="mb-5 mt-1 text-sm leading-relaxed text-slate-500">
                  An MMDA incident ({effect.lanes} lanes, active {effect.active_minutes} minutes) is
                  injected on the route of {effect.n_trips} trips and each system plans again.
                </p>
                <div className="overflow-hidden rounded-lg border border-slate-200">
                  <table className="w-full border-collapse text-sm">
                    <thead>
                      <tr className="bg-slate-50 text-left">
                        <th className="px-4 py-2.5 font-semibold text-slate-600">System</th>
                        <th className="px-4 py-2.5 text-right font-semibold text-slate-600">
                          Routes changed
                        </th>
                        <th className="px-4 py-2.5 text-right font-semibold text-slate-600">
                          Event exposure, before → after
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(effect.systems).map(([system, s]) => (
                        <tr key={system} className="border-t border-slate-200">
                          <td className="px-4 py-2.5 text-slate-700">{system === 'antroute' ? 'ANTRoute' : 'Baseline'}</td>
                          <td className="px-4 py-2.5 text-right tabular-nums text-slate-800">
                            {pct(s.routes_changed_share)}
                          </td>
                          <td className="px-4 py-2.5 text-right tabular-nums text-slate-800">
                            {s.event_exposure_old.toFixed(2)} → {s.event_exposure_recalculated.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-3 text-xs leading-relaxed text-slate-500">
                  The baseline has no incident input, so it keeps its route.
                </p>
              </Card>
            )}
          </div>
        ) : null}
      </div>
    </>
  );
}
