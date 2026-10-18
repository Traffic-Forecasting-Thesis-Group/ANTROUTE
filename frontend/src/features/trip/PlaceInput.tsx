import { useEffect, useId, useState } from 'react';
import type { ReactNode } from 'react';
import { searchPlaces } from '../../api/routeService';
import type { StudyNode } from '../../api/routeService';

const originIcon = <svg viewBox="0 0 24 24" aria-hidden="true" className="origin-icon"><circle cx="12" cy="12" r="7" fill="none" stroke="currentColor" strokeWidth="2" /><circle cx="12" cy="12" r="2.5" fill="currentColor" /><path d="M12 2v3M12 19v3M2 12h3M19 12h3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>;

/** One trip row: the origin (target icon) or a numbered stop, with its own row actions. */
export default function PlaceInput({ label, value, onSelect, placeholder, marker, actions }: {
  label: string; value: StudyNode | null; onSelect: (place: StudyNode | null) => void;
  placeholder: string; marker: 'origin' | number; actions?: ReactNode;
}) {
  const id = useId();
  const [query, setQuery] = useState(value?.label ?? '');
  const [results, setResults] = useState<StudyNode[]>([]);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState('');
  // Typing clears the selection, so only a new selection rewrites the text; parents remount to clear.
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
  return <div className={`place-input ${marker === 'origin' ? 'origin' : 'stop'}`} onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false); }}>
    <label htmlFor={id} className="visually-hidden">{label}</label>
    <div className="place-field">
      {marker === 'origin' ? originIcon : <span className="stop-number" aria-hidden="true">{marker}</span>}
      <input id={id} required value={query} autoComplete="off" placeholder={placeholder} aria-controls={`${id}-results`}
        aria-expanded={open && results.length > 0} role="combobox"
        onFocus={() => setOpen(true)} onChange={event => { setQuery(event.target.value); onSelect(null); setOpen(true); }} />
      {actions && <div className="place-actions">{actions}</div>}
    </div>
    {open && results.length > 0 && <ul id={`${id}-results`} className="place-results" role="listbox" aria-label={`${label} suggestions`}>
      {results.map(place => <li key={place.id}><button role="option" aria-selected={place.id === value?.id} type="button" onClick={() => { onSelect(place); setQuery(place.label); setOpen(false); }}>{place.label}</button></li>)}
    </ul>}
    {error && <small role="status">{error}</small>}
  </div>;
}
