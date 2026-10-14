/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** FastAPI base URL. Falls back to http://127.0.0.1:8100 (docker-compose.override.yml
   *  remaps the backend's 8000 to 8100 on the host). */
  readonly VITE_API_URL?: string;
  /** Maps JavaScript API key. Must have Maps JavaScript API enabled and an HTTP-referrer
   *  rule covering the dev server, or the map renders as a grey box. */
  readonly VITE_GOOGLE_MAPS_API_KEY?: string;
  /** Vector Map ID; required for AdvancedMarker and for camera heading/tilt. */
  readonly VITE_GOOGLE_MAPS_MAP_ID?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
