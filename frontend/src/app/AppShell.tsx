import type { ReactNode } from 'react';

import TopNav from './TopNav';

/**
 * The page: a fixed navigation bar across the top, content below. The dashboard grid needs
 * roughly 1100px, so the content keeps a min-width and the window scrolls horizontally below
 * that rather than collapsing into an unreadable stack.
 */
export default function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col bg-slate-100">
      <TopNav />
      <div className="flex-1 overflow-x-auto">
        <main className="flex min-h-full min-w-[var(--app-min-width)] flex-col">{children}</main>
      </div>
    </div>
  );
}
