import axios from 'axios';

// Set VITE_API_URL in .env.local. The fallback is the local docker-compose stack:
// docker-compose.override.yml remaps the backend's container port 8000 to 8100 on the host.
const baseURL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8100';

const apiClient = axios.create({
  baseURL,
  // Route planning runs two ant colonies on the whole city; give it longer than the
  // platform default rather than reporting a slow answer as an unreachable server.
  timeout: 90000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export default apiClient;
