import { useEffect, useState } from 'react';

import type { IncidentQuery, MapIncident } from '@/types/markers';

import { incidentSource } from './incidentSource';

/**
 * Incidents for the map layer. Returns an empty list today, because incidentSource is the
 * empty stub -- pointing that one export at a real feed is all this needs.
 */
export function useIncidents(query: IncidentQuery) {
  const [incidents, setIncidents] = useState<MapIncident[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const at = query.at ? query.at.getTime() : null;
  const kinds = query.kinds?.join(',') ?? '';

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    incidentSource
      .list(query)
      .then((results) => {
        if (cancelled) return;
        setIncidents(results);
        setError(null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setIncidents([]);
        setError(err instanceof Error ? err.message : 'Could not load map incidents.');
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // The query object is rebuilt every render; its scalar parts are the real dependencies.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [at, kinds]);

  return { incidents, isLoading, error };
}
