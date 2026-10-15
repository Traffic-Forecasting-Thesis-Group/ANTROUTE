import { NavLink } from 'react-router-dom';

import { cn } from '@/lib/cn';

const NAV = [
  { to: '/', label: 'Home', end: true },
  { to: '/evaluation', label: 'Evaluation', end: false },
  { to: '/comparison', label: 'Comparison', end: false },
];

export default function TopNav() {
  return (
    <header className="flex shrink-0 items-center gap-10 border-b border-slate-200 bg-white px-8 py-4">
      <div className="flex items-center gap-3">
        <img
          src="/logo-no-bg.png"
          alt=""
          className="h-11 w-11 rounded-full object-contain"
        />
        <span className="text-2xl font-bold tracking-tight text-brand">
          ANT<span className="text-accent-dark">Route</span>
        </span>
      </div>

      <nav>
        <ul className="flex items-center gap-7">
          {NAV.map(({ to, label, end }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={end}
                className={({ isActive }) =>
                  cn(
                    'text-sm font-medium transition-colors',
                    isActive
                      ? 'text-brand underline decoration-2 underline-offset-8'
                      : 'text-slate-600 hover:text-brand'
                  )
                }
              >
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
}
