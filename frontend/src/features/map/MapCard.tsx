import { useState } from 'react';
import { Map, useMap } from '@vis.gl/react-google-maps';
import { Crosshair, Minus, Plus } from 'lucide-react';

import { reverseGeocode } from '@/api/placesService';
import { cn } from '@/lib/cn';
import { MANILA_CENTER, MANILA_DEFAULT_ZOOM, PLACE_ZOOM } from '@/lib/constants';
import { GOOGLE_MAPS_MAP_ID, hasMapsKey } from '@/lib/maps-env';
import { useGeolocation } from '@/features/navigation/useGeolocation';
import { useTripPlanner } from '@/features/trip/TripPlannerContext';
import { useIncidents } from '@/data/useIncidents';

import NavigationBanner from '@/features/navigation/NavigationBanner';
import { useTurnByTurn } from '@/features/navigation/useTurnByTurn';

import MapLegend from './MapLegend';
import {
  CameraController,
  IncidentMarkers,
  RoutePolylines,
  StopMarkers,
} from './MapOverlays';
import TripControls from './TripControls';
import { useTrafficLayer } from './useTrafficLayer';

type MapType = 'roadmap' | 'satellite';

function MapTypeToggle({
  value,
  onChange,
}: {
  value: MapType;
  onChange: (next: MapType) => void;
}) {
  return (
    <div className="pointer-events-auto flex overflow-hidden rounded-lg border border-slate-200 bg-white shadow-panel">
      {(['roadmap', 'satellite'] as const).map((type) => (
        <button
          key={type}
          type="button"
          onClick={() => onChange(type)}
          className={cn(
            'px-4 py-1.5 text-xs font-semibold transition-colors',
            value === type ? 'bg-brand text-white' : 'text-slate-600 hover:bg-slate-50'
          )}
        >
          {type === 'roadmap' ? 'Map' : 'Satellite'}
        </button>
      ))}
    </div>
  );
}

function ZoomRecenterControls() {
  const map = useMap();
  const { setGpsCoords, setOriginFromCoords, origin } = useTripPlanner();
  const { locate, isLocating } = useGeolocation();

  const recenter = async () => {
    try {
      const coords = await locate();
      setGpsCoords(coords);
      map?.panTo({ lat: coords.latitude, lng: coords.longitude });
      map?.setZoom(PLACE_ZOOM);
      // An empty origin box is the common case for "where am I" -- fill it in too.
      if (!origin.trim()) {
        const label = await reverseGeocode(coords.latitude, coords.longitude);
        setOriginFromCoords(coords, label || 'Current location');
      }
    } catch {
      // The origin field surfaces location errors; the map button stays quiet.
    }
  };

  const nudgeZoom = (delta: number) => {
    if (!map) return;
    const current = map.getZoom() ?? MANILA_DEFAULT_ZOOM;
    map.setZoom(current + delta);
  };

  return (
    <div className="pointer-events-auto flex flex-col gap-2">
      <div className="flex flex-col overflow-hidden rounded-lg border border-slate-200 bg-white shadow-panel">
        <button
          type="button"
          onClick={() => nudgeZoom(1)}
          aria-label="Zoom in"
          className="p-2 text-slate-600 hover:bg-slate-50"
        >
          <Plus size={16} />
        </button>
        <span className="h-px bg-slate-200" aria-hidden />
        <button
          type="button"
          onClick={() => nudgeZoom(-1)}
          aria-label="Zoom out"
          className="p-2 text-slate-600 hover:bg-slate-50"
        >
          <Minus size={16} />
        </button>
      </div>
      <button
        type="button"
        onClick={recenter}
        disabled={isLocating}
        aria-label="Centre on my location"
        className="rounded-lg border border-slate-200 bg-white p-2 text-brand shadow-panel hover:bg-slate-50 disabled:opacity-60"
      >
        <Crosshair size={16} />
      </button>
    </div>
  );
}

/** Shown instead of the map when no Maps JavaScript API key is configured. */
function MissingKeyNotice() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 bg-slate-100 px-8 text-center">
      <p className="text-sm font-semibold text-slate-700">Map unavailable</p>
      <p className="max-w-md text-sm text-slate-500">
        Set <code className="rounded bg-slate-200 px-1">VITE_GOOGLE_MAPS_API_KEY</code> in{' '}
        <code className="rounded bg-slate-200 px-1">.env.local</code> and restart the dev
        server. The browser needs a key with the Maps JavaScript API enabled &mdash; the old
        native Android/iOS key will not work here.
      </p>
    </div>
  );
}

export default function MapCard() {
  const [mapType, setMapType] = useState<MapType>('roadmap');
  const [trafficEnabled, setTrafficEnabled] = useState(true);
  const { departAt } = useTripPlanner();
  const { incidents } = useIncidents({ at: departAt });

  return (
    <div className="relative h-full min-h-[560px] overflow-hidden rounded-xl border border-slate-200 bg-white shadow-card">
      {hasMapsKey ? (
        <>
          <Map
            mapId={GOOGLE_MAPS_MAP_ID || undefined}
            defaultCenter={MANILA_CENTER}
            defaultZoom={MANILA_DEFAULT_ZOOM}
            mapTypeId={mapType}
            disableDefaultUI
            clickableIcons={false}
            gestureHandling="greedy"
            className="h-full w-full"
          >
            <IncidentMarkers incidents={incidents} />
            <TrafficLayerBinding enabled={trafficEnabled} />

            {/* Everything that needs the map instance lives inside <Map>. */}
            <MapContent
              mapType={mapType}
              onMapTypeChange={setMapType}
              trafficEnabled={trafficEnabled}
              onTrafficChange={setTrafficEnabled}
            />
          </Map>
        </>
      ) : (
        <MissingKeyNotice />
      )}
    </div>
  );
}

/** useTrafficLayer needs to run inside <Map>, which owns the map instance. */
function TrafficLayerBinding({ enabled }: { enabled: boolean }) {
  useTrafficLayer(enabled);
  return null;
}

/**
 * The overlays and floating chrome. This sits inside <Map> because the navigation hook and
 * the camera controller both call useMap(), which only resolves under the map's provider.
 */
function MapContent({
  mapType,
  onMapTypeChange,
  trafficEnabled,
  onTrafficChange,
}: {
  mapType: MapType;
  onMapTypeChange: (next: MapType) => void;
  trafficEnabled: boolean;
  onTrafficChange: (next: boolean) => void;
}) {
  const nav = useTurnByTurn();

  return (
    <>
      <RoutePolylines navProgress={nav.isNavigating ? nav.progress : null} />
      <StopMarkers />
      <CameraController followingDriver={nav.isNavigating} />

      {/* Chrome floats over the map; the wrapper must not swallow map gestures. */}
      <div className="pointer-events-none absolute inset-0 p-3">
        <div className="flex h-full flex-col justify-between gap-3">
          <div className="flex items-start justify-between gap-3">
            {nav.isNavigating ? (
              <NavigationBanner nav={nav} />
            ) : (
              <div className="pointer-events-auto flex items-center gap-2">
                <MapTypeToggle value={mapType} onChange={onMapTypeChange} />
                <select
                  value={trafficEnabled ? 'on' : 'off'}
                  onChange={(e) => onTrafficChange(e.target.value === 'on')}
                  aria-label="Traffic layer"
                  className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 shadow-panel focus:outline-none"
                >
                  <option value="on">Show traffic layer</option>
                  <option value="off">Hide traffic layer</option>
                </select>
              </div>
            )}
            <MapLegend />
          </div>

          <div className="flex items-end justify-between gap-3">
            <TripControls nav={nav} />
            <ZoomRecenterControls />
          </div>
        </div>
      </div>
    </>
  );
}
