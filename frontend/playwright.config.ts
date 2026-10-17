import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  timeout: 60000,
  expect: { timeout: 15000 },
  use: { baseURL: 'http://127.0.0.1:5174', channel: 'msedge', headless: true, viewport: { width: 1440, height: 1100 } },
  webServer: [{
    command: '..\\traffic_system\\venv\\Scripts\\python.exe -m uvicorn app.main:app --app-dir ../traffic_system --host 127.0.0.1 --port 8001 --no-access-log',
    url: 'http://127.0.0.1:8001/health',
    reuseExistingServer: false,
    env: { CORS_ORIGINS: 'http://127.0.0.1:5174', STUDY_NETWORK_PATH: '../traffic_system/data/processed/study_network.json' },
  }, {
    command: 'npm run dev -- --host 127.0.0.1 --port 5174',
    url: 'http://127.0.0.1:5174',
    reuseExistingServer: false,
    env: { VITE_GOOGLE_MAPS_API_KEY: '', VITE_API_URL: 'http://127.0.0.1:8001' },
  }],
});
