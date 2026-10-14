/**
 * Maps credentials. The old native GOOGLE_MAPS_API_KEY does not carry over: the browser
 * needs a key with the Maps JavaScript API enabled, billing active, and an HTTP-referrer
 * restriction covering the dev server. A key restricted to the Android/iOS SDKs loads a
 * grey box and logs RefererNotAllowedMapError.
 */
export const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY ?? '';

/** Vector Map ID. AdvancedMarker needs one, as does camera heading/tilt. */
export const GOOGLE_MAPS_MAP_ID = import.meta.env.VITE_GOOGLE_MAPS_MAP_ID ?? '';

export const hasMapsKey = GOOGLE_MAPS_API_KEY.trim().length > 0;
export const hasMapId = GOOGLE_MAPS_MAP_ID.trim().length > 0;
