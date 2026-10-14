/** Metro Manila, where the routing network lives. */
export const MANILA_CENTER = { lat: 14.5995, lng: 120.9842 };
export const MANILA_DEFAULT_ZOOM = 12;

/**
 * Navigation thresholds. Within ARRIVAL_RADIUS_M of a stop counts as reaching it; being
 * OFF_ROUTE_M from the route for OFF_ROUTE_FIXES fixes in a row (each accurate to within
 * OFF_ROUTE_M) triggers a reroute.
 */
export const ARRIVAL_RADIUS_M = 35;
export const OFF_ROUTE_M = 50;
export const OFF_ROUTE_FIXES = 3;

/** Zoom the camera settles at when following the driver, and when centring on one place. */
export const NAV_FOLLOW_ZOOM = 17;
export const PLACE_ZOOM = 15;

/** Padding for fitBounds, in pixels inside the map card. Top and right leave room for the
 *  map-type toggle and the legend panel, which float over the map. */
export const FIT_PADDING = { top: 72, right: 220, bottom: 56, left: 56 };
