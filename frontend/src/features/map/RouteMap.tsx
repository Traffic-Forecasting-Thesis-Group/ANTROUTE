import { useEffect } from 'react';
import { APIProvider, AdvancedMarker, InfoWindow, Map, Pin, useMap, useMapsLibrary } from '@vis.gl/react-google-maps';
import type { BoundaryGeometry, RoadSelection, StudyNode, DatasetMap, RouteExplanation } from '../../api/routeService';
import { StretchCard, stretchColor, stretchLetter, stretchReason } from './RouteExplanation';

interface MapProps {
  geometry: [number, number][];
  baselineGeometry: [number, number][];
  boundaries: Record<string, BoundaryGeometry>;
  origin: StudyNode | null;
  destinations: StudyNode[];
  closures: RoadSelection[];
  onPick: (latitude: number, longitude: number) => void;
  disabled: boolean;
  datasetMap?: DatasetMap | null;
  explanation?: RouteExplanation | null;
  focusedStretch?: number | null;
  onFocusStretch?: (index: number | null) => void;
}

export const ROUTE_COLOR = '#4475f2';
// Purple stays distinct from the city colors of the coverage layer.
export const BASELINE_COLOR = '#a855f7';

function rings(geometry: BoundaryGeometry): number[][][] {
  return geometry.type === 'Polygon' ? [geometry.coordinates[0] as number[][]]
    : (geometry.coordinates as number[][][][]).map(polygon => polygon[0]);
}

function Overlays({ geometry, baselineGeometry, boundaries, closures, datasetMap, explanation }: Pick<MapProps, 'geometry' | 'baselineGeometry' | 'boundaries' | 'closures' | 'datasetMap' | 'explanation'>) {
  const map = useMap();
  const maps = useMapsLibrary('maps');
  useEffect(() => {
    if (!map || !maps || !explanation) return;
    // Dashed alternative stretches sit above the routes so the road ANTROUTE left out stays visible.
    const lines = explanation.stretches.map(stretch => new maps.Polyline({ map, strokeOpacity: 0, zIndex: 25,
      path: stretch.alternative.geometry.map(([lng, lat]) => ({ lat, lng })),
      icons: [{ icon: { path: 'M 0,-1 0,1', strokeOpacity: 1, strokeColor: stretchColor(stretch), strokeWeight: 4, scale: 3 }, offset: '0', repeat: '14px' }] }));
    return () => lines.forEach(line => line.setMap(null));
  }, [explanation, map, maps]);
  useEffect(() => {
    if (!map || !maps || !datasetMap) return;
    const layer = new maps.Data({ map });
    layer.addGeoJson({ type: 'FeatureCollection', features: datasetMap.bands.filter(band => band.lines.length).map(band => ({
      type: 'Feature', properties: { color: band.color }, geometry: { type: 'MultiLineString', coordinates: band.lines },
    })) });
    layer.setStyle(feature => ({ strokeColor: feature.getProperty('color') as string, strokeWeight: 2, strokeOpacity: .7, clickable: false, zIndex: 1 }));
    return () => layer.setMap(null);
  }, [datasetMap, map, maps]);
  useEffect(() => {
    if (!map || !maps) return;
    const polygons = Object.values(boundaries).flatMap(boundary => {
      const parts = boundary.type === 'Polygon' ? [boundary.coordinates as number[][][]] : boundary.coordinates as number[][][][];
      return parts.map(polygon => new maps.Polygon({ map,
        paths: polygon.map(ring => ring.map(([lng, lat]) => ({ lat, lng }))),
        strokeColor: '#334eb5', strokeOpacity: 0.6, strokeWeight: 1.5,
        fillColor: '#4475f2', fillOpacity: 0.035, clickable: false,
      }));
    });
    return () => polygons.forEach(polygon => polygon.setMap(null));
  }, [boundaries, map, maps]);
  useEffect(() => {
    if (!map || !maps) return;
    const lines = closures.map(road => new maps.Polyline({ map,
      path: road.geometry.map(([lng, lat]) => ({ lat, lng })), strokeColor: '#dc2626', strokeWeight: 7, zIndex: 10,
    }));
    return () => lines.forEach(line => line.setMap(null));
  }, [closures, map, maps]);
  useEffect(() => {
    if (!map || !maps) return;
    // The wider baseline line sits under the ACO route so shared roads show both colors.
    const lines = ([[baselineGeometry, BASELINE_COLOR, 8, 15], [geometry, ROUTE_COLOR, 5, 20]] as const)
      .filter(([coordinates]) => coordinates.length >= 2)
      .map(([coordinates, strokeColor, strokeWeight, zIndex]) => new maps.Polyline({ map, strokeColor, strokeWeight, zIndex,
        path: coordinates.map(([lng, lat]) => ({ lat, lng })) }));
    if (!lines.length) return;
    const bounds = new google.maps.LatLngBounds();
    lines.forEach(line => line.getPath().forEach(point => { bounds.extend(point); }));
    map.fitBounds(bounds, 48);
    return () => lines.forEach(line => line.setMap(null));
  }, [geometry, baselineGeometry, map, maps]);
  return null;
}

function NetworkPreview({ geometry, baselineGeometry, boundaries, origin, destinations, closures, datasetMap, explanation, onPick, disabled }: MapProps) {
  const coordinates = Object.values(boundaries).flatMap(boundary => rings(boundary).flat());
  const west = coordinates.length ? Math.min(...coordinates.map(point => point[0])) - .005 : 120.94;
  const east = coordinates.length ? Math.max(...coordinates.map(point => point[0])) + .005 : 121.14;
  const south = coordinates.length ? Math.min(...coordinates.map(point => point[1])) - .005 : 14.515;
  const north = coordinates.length ? Math.max(...coordinates.map(point => point[1])) + .005 : 14.665;
  const project = ([lng, lat]: number[]) => [(lng - west) / (east - west) * 800, (north - lat) / (north - south) * 620];
  const points = (coordinates: number[][]) => coordinates.map(point => project(point).join(',')).join(' ');
  return <svg viewBox="0 0 800 620" className="network-preview" aria-label="Study boundary map. Click to select a road location."
    data-west={west} data-east={east} data-south={south} data-north={north}
    onClick={event => {
      if (disabled) return;
      const rect = event.currentTarget.getBoundingClientRect();
      // Preserve the SVG's letterboxed aspect ratio when converting click positions.
      const scale = Math.min(rect.width / 800, rect.height / 620);
      const x = (event.clientX - rect.left - (rect.width - scale * 800) / 2) / scale;
      const y = (event.clientY - rect.top - (rect.height - scale * 620) / 2) / scale;
      onPick(north - y / 620 * (north - south), west + x / 800 * (east - west));
    }}>
    <rect width="800" height="620" fill="#eef3f8" />
    {Object.entries(boundaries).flatMap(([city, boundary]) => rings(boundary).map((ring, index) => <polygon key={`${city}-${index}`} points={points(ring)} fill="#dfe8fa" stroke="#8ba0cb" strokeWidth="1.5"><title>{city}</title></polygon>))}
    {datasetMap?.bands.map(band => <path key={band.label} d={band.lines.map(line => line.map((point, index) => `${index ? 'L' : 'M'}${project(point).map(value => value.toFixed(2)).join(',')}`).join('')).join(' ')} fill="none" stroke={band.color} strokeWidth=".8" opacity=".8" />)}
    {closures.map(road => <polyline key={road.id} points={points(road.geometry)} fill="none" stroke="#dc2626" strokeWidth="5" />)}
    <polyline points={points(baselineGeometry)} fill="none" stroke={BASELINE_COLOR} strokeWidth="7" />
    <polyline points={points(geometry)} fill="none" stroke={ROUTE_COLOR} strokeWidth="4" />
    {explanation?.stretches.map((stretch, index) => <g key={index}>
      <polyline points={points(stretch.alternative.geometry)} fill="none" stroke={stretchColor(stretch)} strokeWidth="4" strokeDasharray="6 5" />
      <g transform={`translate(${project(stretch.marker).join(',')})`}><circle r="9" fill={stretchColor(stretch)} stroke="white" strokeWidth="2" />
        <text y="4" textAnchor="middle" fontSize="10" fontWeight="700" fill="white">{stretchLetter(index)}</text>
        <title>{(({ title, detail }) => `${title}. ${detail}`)(stretchReason(stretch, explanation.ant_routes))}</title></g>
    </g>)}
    {[origin, ...destinations].map((node, index) => node && <g key={index} transform={`translate(${project([node.longitude, node.latitude]).join(',')})`}>
      <circle r="9" fill={index === 0 ? '#15803d' : '#be123c'} stroke="white" strokeWidth="2" />
      {index > 0 && <text y="4" textAnchor="middle" fontSize="10" fontWeight="700" fill="white">{index}</text>}
      <text y="-15" textAnchor="middle" fontSize="13" fontWeight="700">{index === 0 ? 'Start' : `Destination ${index}`}</text>
    </g>)}
    <text x="16" y="600" fontSize="12" fill="#475569">Study boundaries and route · © OpenStreetMap contributors · map preview</text>
  </svg>;
}

/** Lettered markers on the alternative stretches; clicking one shows why ANTROUTE did not take it. */
function StretchMarkers({ explanation, open, setOpen }: { explanation: RouteExplanation; open: number | null; setOpen: (index: number | null) => void }) {
  const map = useMap();
  const selected = open === null ? null : explanation.stretches[open] ?? null;
  useEffect(() => {
    // Short stretches can sit under the start pin; zoom in so the alternative road is visible.
    if (!map || !selected) return;
    map.panTo({ lat: selected.marker[1], lng: selected.marker[0] });
    if ((map.getZoom() ?? 0) < 16) map.setZoom(16);
  }, [map, selected]);
  return <>
    {explanation.stretches.map((stretch, index) => <AdvancedMarker key={index} position={{ lat: stretch.marker[1], lng: stretch.marker[0] }}
      title={stretchReason(stretch, explanation.ant_routes).title} onClick={() => setOpen(index)} zIndex={30}>
      <Pin background={stretchColor(stretch)} borderColor="white" glyphColor="white" glyph={stretchLetter(index)} scale={0.9} />
    </AdvancedMarker>)}
    {selected && <InfoWindow position={{ lat: selected.marker[1], lng: selected.marker[0] }} onCloseClick={() => setOpen(null)} maxWidth={380}>
      <div className="stretch-info"><StretchCard stretch={selected} letter={stretchLetter(open!)} antRoutes={explanation.ant_routes} /></div>
    </InfoWindow>}
  </>;
}

export default function RouteMap(props: MapProps) {
  if (!import.meta.env.VITE_GOOGLE_MAPS_API_KEY?.trim()) return <NetworkPreview {...props} />;
  return <APIProvider apiKey={import.meta.env.VITE_GOOGLE_MAPS_API_KEY}><Map mapId={import.meta.env.VITE_GOOGLE_MAPS_MAP_ID || 'DEMO_MAP_ID'} defaultCenter={{ lat: 14.59, lng: 121.025 }} defaultZoom={12}
    gestureHandling="cooperative" onClick={event => {
      if (!props.disabled && event.detail.latLng) props.onPick(event.detail.latLng.lat, event.detail.latLng.lng);
    }}>
    <Overlays {...props} />
    {props.explanation && <StretchMarkers explanation={props.explanation} open={props.focusedStretch ?? null} setOpen={props.onFocusStretch ?? (() => undefined)} />}
    {props.origin && <AdvancedMarker position={{ lat: props.origin.latitude, lng: props.origin.longitude }} title="Start location"><Pin background="#15803d" borderColor="#166534" glyphColor="white" /></AdvancedMarker>}
    {props.destinations.map((stop, index) => <AdvancedMarker key={`${index}-${stop.id}`} position={{ lat: stop.latitude, lng: stop.longitude }} title={`Destination ${index + 1}`}><Pin background="#be123c" borderColor="#9f1239" glyphColor="white" glyph={String(index + 1)} /></AdvancedMarker>)}
  </Map></APIProvider>;
}
