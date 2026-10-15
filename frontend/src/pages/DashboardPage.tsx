import { useState } from 'react';
import { CarFront, Search } from 'lucide-react';

import { Button, ErrorText, Spinner } from '@/components/ui';
import MapCard from '@/features/map/MapCard';
import DeparturePicker from '@/features/trip/DeparturePicker';
import RouteResultsList from '@/features/trip/RouteResultsList';
import TripDetailsPanel from '@/features/trip/TripDetailsPanel';
import { useTripPlanner } from '@/features/trip/TripPlannerContext';
import UploadFilePanel from '@/features/upload/UploadFilePanel';
import { parseInputFeaturesCsv } from '@/features/upload/parseInputFeaturesCsv';
import type { UploadStatus } from '@/types/upload';

export default function DashboardPage() {
  const { findRoutes, isLoading, routeError, setUploadedIncidents } = useTripPlanner();
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>('idle');
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);

  return (
    <div className="flex flex-1 bg-white">
      {/* Control column */}
      <aside className="flex w-[300px] shrink-0 flex-col gap-5 border-r border-slate-200 px-6 py-6">
        <UploadFilePanel
          status={uploadStatus}
          statusMessage={uploadMessage}
          onClear={() => {
            setUploadedIncidents([]);
            setUploadStatus('idle');
            setUploadMessage(null);
          }}
          onFileAccepted={async (dataset) => {
            setUploadStatus('validating');
            setUploadMessage(null);
            try {
              const { incidents, summary, problems } = await parseInputFeaturesCsv(dataset.file);
              setUploadedIncidents(incidents);
              setUploadStatus(incidents.length > 0 ? 'ready' : 'error');
              setUploadMessage(
                incidents.length > 0
                  ? (summary ?? `${incidents.length} on the map.`) +
                      (problems.length > 0 ? ` ${problems.length} could not be placed.` : '')
                  : (problems[0] ?? 'No input features could be read from this file.')
              );
              // The rows that could not be placed are named, not swallowed.
              if (problems.length > 0) console.warn('[ANTRoute] input features CSV:', problems);
            } catch (error: any) {
              setUploadedIncidents([]);
              setUploadStatus('error');
              setUploadMessage(error?.message || 'Could not read this file.');
            }
          }}
        />

        <TripDetailsPanel />
        <DeparturePicker />

        <div>
          <p className="mb-2 flex items-center gap-1.5 text-[11px] text-slate-400">
            <CarFront size={13} aria-hidden />
            Routes optimized for private 4-wheel vehicles.
          </p>
          <Button block onClick={findRoutes} disabled={isLoading}>
            {isLoading ? <Spinner /> : <Search size={15} aria-hidden />}
            {isLoading ? 'Finding routes…' : 'Find Optimal Route'}
          </Button>
          <ErrorText>{routeError}</ErrorText>
        </div>
      </aside>

      {/* Map, then the route options beneath it */}
      <div className="flex min-w-0 flex-1 flex-col gap-5 px-6 py-6">
        <div className="h-[min(520px,55vh)] min-h-[380px] shrink-0">
          <MapCard />
        </div>
        <RouteResultsList />
      </div>
    </div>
  );
}
