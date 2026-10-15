import { useState } from 'react';
import { CarFront, Download, Search } from 'lucide-react';

import { getRecordedSessions } from '@/api/routeService';
import PageHeader from '@/app/PageHeader';
import { Button, ErrorText, Spinner } from '@/components/ui';
import MapCard from '@/features/map/MapCard';
import DeparturePicker from '@/features/trip/DeparturePicker';
import RecordedInputsPanel from '@/features/trip/RecordedInputsPanel';
import RouteResultsList from '@/features/trip/RouteResultsList';
import SimulationPicker from '@/features/trip/SimulationPicker';
import TripDetailsPanel from '@/features/trip/TripDetailsPanel';
import { useTripPlanner } from '@/features/trip/TripPlannerContext';
import UploadFilePanel from '@/features/upload/UploadFilePanel';
import { parseInputFeatures, UploadError } from '@/lib/inputFeatures';
import type { UploadStatus } from '@/types/upload';

export default function DashboardPage() {
  const { findRoutes, isLoading, routeError, setUploadedInputs, setSimulationWindow } = useTripPlanner();
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>('idle');
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);

  /** Reads the input-features CSV in the browser, checks it, and plans on its first recorded window. */
  const acceptUpload = async (file: File) => {
    setUploadStatus('validating');
    setUploadMessage(null);
    try {
      const upload = parseInputFeatures(file.name, await file.text());
      // The routes come from the server's stored scores, so only windows it recorded can be planned.
      const recorded = new Set((await getRecordedSessions()).flatMap((s) => s.windows.map((w) => w.start)));
      const usable = upload.windowStarts.filter((w) => recorded.has(w));
      if (usable.length === 0) {
        throw new UploadError(
          'None of the file’s windows match a recorded window on the server, so there is nothing to plan on.'
        );
      }
      setUploadedInputs(upload);
      setSimulationWindow(usable[0]);
      setUploadStatus('ready');
      setUploadMessage(
        `${upload.rowCount} rows, ${upload.windowStarts.length} windows (${usable.length} usable). ` +
          'Shown in Inputs used; routes use the stored model scores for the same window.'
      );
    } catch (error) {
      setUploadedInputs(null);
      setUploadStatus('error');
      setUploadMessage(error instanceof Error ? error.message : 'Could not read that file.');
    }
  };

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
          <SimulationPicker />

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
          <RecordedInputsPanel />
        </div>

        {/* Map column */}
        <div className="min-w-0 flex-1">
          <MapCard />
        </div>
      </div>
    </>
  );
}
