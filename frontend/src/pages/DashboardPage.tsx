import { CarFront, Search } from 'lucide-react';

import PageHeader from '@/app/PageHeader';
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
    <>
      <PageHeader title="Dashboard" />

      <div className="flex flex-1 gap-6 p-6">
        {/* Control column */}
        <div className="flex w-[300px] shrink-0 flex-col gap-5 overflow-y-auto">
          <UploadFilePanel
            onFileAccepted={(dataset) => {
              // TODO(csv-branch): parse the file and feed the routing inputs once the
              // input features are finalized. Nothing has read its contents yet.
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

          <RouteResultsList />
        </div>

        {/* Map column */}
        <div className="min-w-0 flex-1">
          <MapCard />
        </div>
      </div>
    </>
  );
}
