import { useEffect, useId, useState } from 'react';
import { searchPlaces } from '../../api/routeService';
import type { StudyNode } from '../../api/routeService';

export default function PlaceInput({ label, value, onSelect }: {
  label: string; value: StudyNode | null; onSelect: (place: StudyNode | null) => void;
}) {
  const id = useId();
  const [query, setQuery] = useState(value?.label ?? '');
  const [results, setResults] = useState<StudyNode[]>([]);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => { if (value) setQuery(value.label); }, [value]);
  useEffect(() => {
    if (query.length < 2 || (value && query === value.label)) { setResults([]); return; }
    const controller = new AbortController();
    const timer = setTimeout(() => {
      searchPlaces(query, controller.signal).then(data => { setResults(data.results); setError(''); })
        .catch((reason: Error) => { if (reason.name !== 'AbortError') setError('Search unavailable. You can select a point on the map.'); });
    }, 250);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [query, value]);
  return <div className="place-input" onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false); }}>
    <label htmlFor={id}>{label}</label>
    <input id={id} required value={query} autoComplete="off" placeholder="Search a road or city" aria-controls={`${id}-results`}
      aria-expanded={open && results.length > 0} role="combobox"
      onFocus={() => setOpen(true)} onChange={event => { setQuery(event.target.value); onSelect(null); setOpen(true); }} />
    {open && results.length > 0 && <ul id={`${id}-results`} className="place-results" role="listbox" aria-label={`${label} suggestions`}>
      {results.map(place => <li key={place.id}><button role="option" aria-selected={place.id === value?.id} type="button" onClick={() => { onSelect(place); setQuery(place.label); setOpen(false); }}>{place.label}</button></li>)}
    </ul>}
    {value && <small className="muted">Selected in {value.city}</small>}
    {error && <small role="status">{error}</small>}
  </div>;
}
