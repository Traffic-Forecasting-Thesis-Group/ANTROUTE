import { useState } from 'react';

import { cn } from '@/lib/cn';
import DynamicAdaptabilityPanel from '@/features/evaluation/DynamicAdaptabilityPanel';
import RoutePerformancePanel from '@/features/evaluation/RoutePerformancePanel';
import TripSummaryRail from '@/features/evaluation/TripSummaryRail';

type Tab = 'performance' | 'adaptability';

/**
 * Evaluation of the trip planned on Home.
 *
 * Route Performance answers research question 1 -- Route Optimality against Google Maps and
 * congestion exposure along the route -- for this trip, with the whole test set's scenario
 * results beside it. Dynamic Adaptability answers research question 2 from the condition-update
 * harness: recalculation response time, whether the route changed, and route quality before
 * and after the conditions changed.
 */
export default function EvaluationPage() {
  const [tab, setTab] = useState<Tab>('performance');

  return (
    <div className="flex flex-1 bg-slate-50">
      <TripSummaryRail />

      <div className="min-w-0 flex-1 overflow-y-auto px-6 py-6">
        <div className="mb-5 inline-flex rounded-lg border border-slate-300 bg-white p-0.5">
          <button
            type="button"
            onClick={() => setTab('performance')}
            className={cn(
              'rounded-md px-4 py-1.5 text-sm font-medium transition-colors',
              tab === 'performance' ? 'bg-brand text-white' : 'text-slate-600 hover:bg-slate-50'
            )}
          >
            Route Performance
          </button>
          <button
            type="button"
            onClick={() => setTab('adaptability')}
            className={cn(
              'rounded-md px-4 py-1.5 text-sm font-medium transition-colors',
              tab === 'adaptability' ? 'bg-brand text-white' : 'text-slate-600 hover:bg-slate-50'
            )}
          >
            Dynamic Adaptability
          </button>
        </div>

        {tab === 'performance' ? <RoutePerformancePanel /> : <DynamicAdaptabilityPanel />}
      </div>
    </div>
  );
}
