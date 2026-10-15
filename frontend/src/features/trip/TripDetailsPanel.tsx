
import { ArrowUpDown, LocateFixed, MapPin, Plus, X } from 'lucide-react';

import { reverseGeocode } from '@/api/placesService';
import { Spinner } from '@/components/ui';
import { cn } from '@/lib/cn';
import { useGeolocation } from '@/features/navigation/useGeolocation';

import PlacesAutocomplete from './PlacesAutocomplete';
import { useTripPlanner } from './TripPlannerContext';

function OriginField() {
  const {
    origin,
    originCoords,
    setOriginText,
    setOriginFromCoords,
    locationError,
    setLocationError,
    closeSuggestions,
  } = useTripPlanner();
  const { locate, isLocating } = useGeolocation();

  const useCurrentLocation = async () => {
    setLocationError('');
    try {
      const coords = await locate();
      const label = await reverseGeocode(coords.latitude, coords.longitude);
      setOriginFromCoords(coords, label || 'Current location');
    } catch (error: any) {
      setLocationError(error?.message || 'Could not get your location.');
    }
  };

  return (
    <div className="relative">
      <div
        className={cn(
          'flex items-center gap-2 rounded-lg border bg-white px-3 py-2',
          originCoords ? 'border-brand' : 'border-slate-300'
        )}
      >
        <LocateFixed size={16} className="shrink-0 text-brand" aria-hidden />
        <input
          value={origin}
          onChange={(e) => setOriginText(e.target.value)}
          onBlur={() => window.setTimeout(() => closeSuggestions('origin'), 120)}
          placeholder="Current location"
          aria-label="Starting point"
          className="min-w-0 flex-1 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none"
        />
        {isLocating ? (
          <Spinner className="text-brand" />
        ) : origin ? (
          <button
            type="button"
            onClick={() => setOriginText('')}
            aria-label="Clear starting point"
            className="shrink-0 text-slate-400 hover:text-slate-600"
          >
            <X size={15} />
          </button>
        ) : (
          <button
            type="button"
            onClick={useCurrentLocation}
            className="shrink-0 whitespace-nowrap text-xs font-semibold text-brand hover:underline"
          >
            Use GPS
          </button>
        )}
      </div>
      <PlacesAutocomplete field="origin" />
      {locationError && <p className="mt-1 text-xs font-medium text-red-600">{locationError}</p>}
    </div>
  );
}

function DestinationRow({ index, total }: { index: number; total: number }) {
  const { destinations, updateDestination, removeDestination, reorderDestination, closeSuggestions } =
    useTripPlanner();
  const entry = destinations[index];
  const isLast = index === total - 1;

  return (
    <div className="relative">
      <div
        className={cn(
          'flex items-center gap-2 rounded-lg border px-3 py-2',
          entry.coords ? 'border-brand bg-brand-light' : 'border-slate-300 bg-white'
        )}
      >
        <span
          className={cn(
            'flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-bold',
            isLast ? 'bg-brand text-white' : 'bg-accent text-slate-900'
          )}
          aria-hidden
        >
          {total > 1 ? index + 1 : <MapPin size={11} />}
        </span>
        <input
          value={entry.name}
          onChange={(e) => updateDestination(index, e.target.value)}
          onBlur={() => window.setTimeout(() => closeSuggestions(index), 120)}
          placeholder={total > 1 ? `Stop ${index + 1}` : 'Choose destination'}
          aria-label={total > 1 ? `Stop ${index + 1}` : 'Destination'}
          className="min-w-0 flex-1 bg-transparent text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none"
        />
        {total > 1 && (
          <button
            type="button"
            onClick={() => reorderDestination(index)}
            aria-label={`Move stop ${index + 1} down`}
            className="shrink-0 text-slate-400 hover:text-slate-600"
          >
            <ArrowUpDown size={14} />
          </button>
        )}
        <button
          type="button"
          onClick={() => removeDestination(index)}
          aria-label={`Remove stop ${index + 1}`}
          className="shrink-0 text-slate-400 hover:text-red-500"
        >
          <X size={15} />
        </button>
      </div>
      <PlacesAutocomplete field={index} />
    </div>
  );
}

export default function TripDetailsPanel() {
  const { destinations, addDestination } = useTripPlanner();

  return (
    <section>
      <h2 className="mb-2 text-sm font-semibold text-slate-700">Trip Details</h2>
      <div className="flex flex-col gap-2">
        <OriginField />
        {destinations.map((_, index) => (
          <DestinationRow key={index} index={index} total={destinations.length} />
        ))}
        <button
          type="button"
          onClick={addDestination}
          className="flex items-center justify-center gap-1.5 rounded-lg border border-slate-300 bg-white py-2 text-xs font-semibold text-brand hover:bg-brand-light"
        >
          <Plus size={14} aria-hidden />
          Add Destination
        </button>
      </div>
    </section>
  );
}
