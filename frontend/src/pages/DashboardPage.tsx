import { CarFront, Search } from 'lucide-react';

import { Button, ErrorText, Spinner } from '@/components/ui';
import MapCard from '@/features/map/MapCard';
import DeparturePicker from '@/features/trip/DeparturePicker';
import RouteResultsList from '@/features/trip/RouteResultsList';
import TripDetailsPanel from '@/features/trip/TripDetailsPanel';
import { useTripPlanner } from '@/features/trip/TripPlannerContext';
import UploadFilePanel from '@/features/upload/UploadFilePanel';

export default function DashboardPage() {
  const { findRoutes, isLoading, routeError } = useTripPlanner();

  return (
    <div className="flex flex-1 bg-white">
      {/* Control column */}
      <aside className="flex w-[300px] shrink-0 flex-col gap-5 border-r border-slate-200 px-6 py-6">
        <UploadFilePanel
          onFileAccepted={(dataset) => {
            // TODO(csv-branch): parse the file and feed the routing inputs once the input
            // features are finalized. Nothing has read its contents yet.
            console.info('[ANTRoute] CSV selected, not yet parsed:', dataset.name);
          }}
          statusMessage="Parsing is not wired up yet."
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
