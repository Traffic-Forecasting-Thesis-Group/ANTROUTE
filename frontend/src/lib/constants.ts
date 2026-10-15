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

/**
 * The simulation day. Every trip is planned on recorded data, and 2026-05-25 is the only
 * day held out of the model's training and validation (configs/visual_split_v8.json marks
 * its AM and PM sessions "test"), so it is the one day a demonstrated route was not fitted
 * on. The departure date is fixed to it; only the time of day is chosen.
 *
 * The backend matches a departure to a recorded window by time of day and prefers a window
 * on the same date (src/routing/departure_window.py match_recorded_window), so sending this
 * date lands on May 25's own recorded traffic.
 */
export const SIMULATION_DATE = { year: 2026, month: 5, day: 25 } as const;
export const SIMULATION_DATE_LABEL = 'May 25, 2026';

/** Where the time field starts: inside the recorded AM session. */
export const SIMULATION_DEFAULT_HOUR = 7;
export const SIMULATION_DEFAULT_MINUTE = 0;
