import { Navigate, Route, Routes } from 'react-router-dom';

import AppShell from './app/AppShell';
import { TripPlannerProvider } from './features/trip/TripPlannerContext';
import DashboardPage from './pages/DashboardPage';
import EvaluationPage from './pages/EvaluationPage';
import ComparisonPage from './pages/ComparisonPage';
import SettingsPage from './pages/SettingsPage';

export default function App() {
  return (
    // The planner lives above <Routes> so a trip planned on the Dashboard is still there
    // when you open Comparison -- mounting it per route would wipe it on every navigation.
    <TripPlannerProvider>
      <AppShell>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
          <Route path="/comparison" element={<ComparisonPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppShell>
    </TripPlannerProvider>
  );
}
