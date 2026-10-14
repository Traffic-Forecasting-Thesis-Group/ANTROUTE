import type { PlaceSuggestion } from '@/api/placesService';
import { Spinner } from '@/components/ui';

import { useTripPlanner, type ActiveField } from './TripPlannerContext';

/**
 * The suggestion list under whichever field is active. Mounted by every field; it renders
 * nothing unless `field` is the one currently being typed in.
 */
export default function PlacesAutocomplete({ field }: { field: ActiveField }) {
  const { activeField, suggestions, isSearchingPlaces, selectSuggestion } = useTripPlanner();

  if (activeField !== field) return null;

  if (isSearchingPlaces) {
    return (
      <div className="absolute left-0 right-0 top-full z-20 mt-1 rounded-lg border border-slate-200 bg-white p-3 text-center shadow-panel">
        <Spinner className="text-brand" />
      </div>
    );
  }

  if (suggestions.length === 0) return null;

  return (
    <ul className="absolute left-0 right-0 top-full z-20 mt-1 max-h-64 overflow-y-auto rounded-lg border border-slate-200 bg-white py-1 shadow-panel">
      {suggestions.map((item: PlaceSuggestion) => (
        <li key={item.id}>
          <button
            type="button"
            // The field's blur would otherwise close this list before the click lands.
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => selectSuggestion(item)}
            className="block w-full px-3 py-2 text-left hover:bg-brand-light"
          >
            <span className="block truncate text-sm font-medium text-slate-800">{item.name}</span>
            {(item.address || item.distanceKm != null) && (
              <span className="block truncate text-xs text-slate-500">
                {[item.address, item.distanceKm != null ? `${item.distanceKm} km` : null]
                  .filter(Boolean)
                  .join(' · ')}
              </span>
            )}
          </button>
        </li>
      ))}
    </ul>
  );
}
