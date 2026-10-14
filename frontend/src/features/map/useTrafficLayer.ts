import { useEffect } from 'react';
import { useMap, useMapsLibrary } from '@vis.gl/react-google-maps';

/** Google's live traffic overlay, toggled by the map card's "Show traffic layer" control. */
export function useTrafficLayer(enabled: boolean) {
  const map = useMap();
  const mapsLib = useMapsLibrary('maps');

  useEffect(() => {
    if (!map || !mapsLib || !enabled) return;
    const layer = new mapsLib.TrafficLayer({ map });
    return () => layer.setMap(null);
  }, [map, mapsLib, enabled]);
}
