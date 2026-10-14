import {
  ArrowUp,
  ArrowUpLeft,
  ArrowUpRight,
  CornerUpLeft,
  CornerUpRight,
  Flag,
  RotateCcw,
} from 'lucide-react';

import { Spinner } from '@/components/ui';
import { formatDistance, maneuverText, type ManeuverKind } from '@/lib/guidance';

import type { TurnByTurnState } from './useTurnByTurn';

function ManeuverIcon({ kind }: { kind: ManeuverKind }) {
  const props = { size: 28, color: '#fff' };
  switch (kind) {
    case 'left':
      return <CornerUpLeft {...props} />;
    case 'right':
      return <CornerUpRight {...props} />;
    case 'slight-left':
      return <ArrowUpLeft {...props} />;
    case 'slight-right':
      return <ArrowUpRight {...props} />;
    case 'u-turn':
      return <RotateCcw {...props} />;
    case 'arrive':
      return <Flag {...props} />;
    default:
      return <ArrowUp {...props} />;
  }
}

/** The next-turn banner, shown over the map while navigating. */
export default function NavigationBanner({ nav }: { nav: TurnByTurnState }) {
  if (!nav.isNavigating) return null;

  const { arrived, isRerouting, maneuver, targetName } = nav;

  return (
    <div className="pointer-events-auto flex items-center gap-3 rounded-xl bg-brand px-4 py-3 text-white shadow-panel">
      <span className="flex h-11 w-11 shrink-0 items-center justify-center">
        {arrived ? (
          <Flag size={28} color="#fff" aria-hidden />
        ) : isRerouting ? (
          <Spinner className="h-6 w-6" />
        ) : (
          <ManeuverIcon kind={maneuver?.kind ?? 'straight'} />
        )}
      </span>

      <span className="min-w-0">
        {arrived ? (
          <>
            <span className="block text-lg font-bold leading-tight">You have arrived</span>
            <span className="block truncate text-sm text-blue-100">{targetName}</span>
          </>
        ) : isRerouting ? (
          <>
            <span className="block text-lg font-bold leading-tight">Rerouting{'…'}</span>
            <span className="block text-sm text-blue-100">
              You left the route. Finding a new one.
            </span>
          </>
        ) : maneuver ? (
          <>
            <span className="block text-lg font-bold leading-tight">
              {formatDistance(maneuver.distanceM)}
            </span>
            <span className="block text-sm text-blue-100">
              {maneuver.kind === 'arrive' && targetName
                ? `Arrive at ${targetName}`
                : maneuverText(maneuver.kind)}
            </span>
          </>
        ) : (
          <>
            <span className="block text-lg font-bold leading-tight">Starting navigation</span>
            <span className="block text-sm text-blue-100">Waiting for your location{'…'}</span>
          </>
        )}
      </span>
    </div>
  );
}
