import { NavLink } from 'react-router-dom';
import { LayoutDashboard, BarChart3, Columns2, FlaskConical, Settings } from 'lucide-react';

import { cn } from '@/lib/cn';

const NAV = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/evaluation', label: 'Evaluation', icon: BarChart3, end: false },
  { to: '/comparison', label: 'Comparison', icon: Columns2, end: false },
  { to: '/what-if', label: 'What-if', icon: FlaskConical, end: false },
  { to: '/settings', label: 'Settings', icon: Settings, end: false },
];

export default function Sidebar() {
  return (
    <nav className="flex w-60 shrink-0 flex-col bg-sidebar px-4 py-6 text-white">
      <div className="mb-10 flex items-center gap-3 px-2">
        <img
          src="/logo-no-bg.png"
          alt=""
          className="h-11 w-11 rounded-full bg-white object-contain p-1"
        />
        <span className="text-xl font-bold tracking-tight">
          ANT<span className="text-accent">Route</span>
        </span>
      </div>

      <ul className="flex flex-col gap-1.5">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <li key={to}>
            <NavLink
              to={to}
              end={end}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-accent text-slate-900 shadow-sm'
                    : 'text-blue-100 hover:bg-sidebar-hover hover:text-white'
                )
              }
            >
              <Icon size={18} aria-hidden />
              {label}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}
