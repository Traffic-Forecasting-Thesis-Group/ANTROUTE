import type { ReactNode } from 'react';

import Sidebar from './Sidebar';

/**
 * Fixed navigation rail beside the page. The dashboard grid needs roughly 1100px, so the
 * content area keeps a min-width and the window scrolls horizontally below that rather
 * than collapsing the two columns into an unreadable stack.
 */
export default function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full min-h-screen overflow-x-auto bg-white">
      <Sidebar />
      <main className="flex min-w-[var(--app-min-width)] flex-1 flex-col">{children}</main>
    </div>
  );
}
