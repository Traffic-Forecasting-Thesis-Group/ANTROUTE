import { CarFront, FlaskConical, GraduationCap, Server, Users } from 'lucide-react';

import PageHeader from '@/app/PageHeader';
import { Card } from '@/components/ui';
import { hasMapId, hasMapsKey } from '@/lib/maps-env';

const AUTHORS = [
  'Guanzon, Kyla Mae C.',
  'Pagallaman, Joshua P.',
  'Palazo, Angel Rose M.',
  'Pepito, Merille Janine O.',
];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mb-6">
      <h2 className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-500">{title}</h2>
      <Card className="p-5">{children}</Card>
    </section>
  );
}

function Row({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-slate-100 py-2.5 last:border-0 last:pb-0 first:pt-0">
      <span className="flex items-center gap-2 text-sm text-slate-700">
        <span className="text-brand">{icon}</span>
        {label}
      </span>
      <span className="text-sm text-slate-500">{value}</span>
    </div>
  );
}

export default function SettingsPage() {
  const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8100 (default)';

  return (
    <>
      <PageHeader title="Settings" />

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl">
          <div className="mb-6 flex items-start gap-2.5 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3">
            <FlaskConical size={16} className="mt-0.5 shrink-0 text-amber-700" aria-hidden />
            <p className="text-sm text-amber-900">
              Simulation prototype for thesis demonstration. Not connected to live CCTV,
              social media, or traffic sensor feeds.
            </p>
          </div>

          <Section title="Connection">
            <Row icon={<Server size={16} />} label="API base URL" value={<code className="text-xs">{apiUrl}</code>} />
            <Row
              icon={<Server size={16} />}
              label="Google Maps key"
              value={hasMapsKey ? 'Configured' : 'Not set'}
            />
            <Row
              icon={<Server size={16} />}
              label="Google Maps Map ID"
              value={hasMapId ? 'Configured' : 'Not set'}
            />
            <p className="mt-3 text-xs text-slate-400">
              Set these in <code>.env.local</code> and restart the dev server. See{' '}
              <code>.env.example</code>.
            </p>
          </Section>

          <Section title="Routing scope">
            <Row
              icon={<CarFront size={16} />}
              label="Vehicle type"
              value="Private 4-wheel vehicles"
            />
            <Row icon={<CarFront size={16} />} label="Coverage" value="Metro Manila" />
          </Section>

          <Section title="What is ANTRoute">
            <p className="text-sm leading-relaxed text-slate-600">
              A CNN-LSTM and RADR-STGNN framework with Ant Colony Optimization that fuses
              CCTV, social media, and traffic data into a congestion-risk score, used to
              route around real, event-driven traffic in Metro Manila.
            </p>
            <p className="mt-3 text-sm leading-relaxed text-slate-600">
              ANTRoute is benchmarked against a baseline Ant Colony model using only path
              length, travel time, and traffic flow. The Comparison page shows Route
              Optimality and ETA accuracy (MAE, RMSE, MSE, R&sup2;) for both.
            </p>
          </Section>

          <Section title="Developed by">
            <div className="flex gap-3">
              <Users size={18} className="mt-0.5 shrink-0 text-brand" aria-hidden />
              <ul className="text-sm text-slate-700">
                {AUTHORS.map((name) => (
                  <li key={name}>{name}</li>
                ))}
              </ul>
            </div>
            <div className="mt-4 flex gap-3">
              <GraduationCap size={18} className="mt-0.5 shrink-0 text-brand" aria-hidden />
              <p className="text-sm leading-relaxed text-slate-600">
                Bachelor of Science in Computer Science
                <br />
                College of Computer and Information Sciences
                <br />
                Polytechnic University of the Philippines, Sta. Mesa, Manila
              </p>
            </div>
          </Section>
        </div>
      </div>
    </>
  );
}
