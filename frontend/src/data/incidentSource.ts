import type { IncidentQuery, IncidentSource, MapIncident } from '@/types/markers';

/** No incidents. The backend has no incident, weather, flood or CCTV feed yet. */
export const emptyIncidentSource: IncidentSource = {
  async list(_query: IncidentQuery): Promise<MapIncident[]> {
    return [];
  },
};

/**
 * TODO(data-branch): point this at the real feed once it lands, e.g.
 *
 *   export const incidentSource: IncidentSource = {
 *     list: ({ bounds, at }) =>
 *       apiClient.get('/incidents', { params: { ...bounds, at: at?.toISOString() } })
 *         .then((r) => r.data.map(toMapIncident)),
 *   };
 *
 * Changing this one line lights up the legend and the marker layer together.
 */
export const incidentSource: IncidentSource = emptyIncidentSource;
