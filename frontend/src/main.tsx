import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { APIProvider } from '@vis.gl/react-google-maps';

import App from './App';
import { GOOGLE_MAPS_API_KEY, hasMapsKey } from './lib/maps-env';
import './index.css';

if (!import.meta.env.VITE_API_URL) {
  console.warn(
    '[ANTRoute] VITE_API_URL is not set; falling back to http://127.0.0.1:8100. ' +
      'Copy .env.example to .env.local and set it.'
  );
}
if (!hasMapsKey) {
  console.warn(
    '[ANTRoute] VITE_GOOGLE_MAPS_API_KEY is not set, so the map will not load. ' +
      'See .env.example -- the web needs the Maps JavaScript API enabled, not the ' +
      'native SDK restriction the old app used.'
  );
}

const root = createRoot(document.getElementById('root')!);

// APIProvider injects the Maps loader. Mounting it with an empty key would request a
// script that is guaranteed to fail, so without a key the app renders and the map card
// explains itself instead (see MapCard).
const tree = hasMapsKey ? (
  <APIProvider apiKey={GOOGLE_MAPS_API_KEY}>
    <App />
  </APIProvider>
) : (
  <App />
);

root.render(
  <React.StrictMode>
    <BrowserRouter>{tree}</BrowserRouter>
  </React.StrictMode>
);
