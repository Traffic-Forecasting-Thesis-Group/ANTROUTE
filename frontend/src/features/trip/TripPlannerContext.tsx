import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';

import {
  getRecordedInputs,
  getTripEvaluation,
  planRoute,
  type ComparisonMetricRow,
  type Coordinates,
  type DestinationInput,
  type RecordedInputs,
  type RouteOption,
  type TripEvaluation,
} from '@/api/routeService';
import { searchPlaces, type PlaceSuggestion } from '@/api/placesService';
import { clampDeparture, currentMinute } from '@/lib/time';

export interface DestinationEntry {
  name: string;
  coords: Coordinates | null;
}

/** Which text field is driving the suggestion list: the origin, or a destination index. */
export type ActiveField = 'origin' | number | null;

export type ModelTab = 'antroute' | 'baseline';

/** A camera request the map picks up; the planner never touches the map directly. */
export interface CameraTarget {
  coords: Coordinates;
  /** Bumped on every request so repeating the same coordinates still moves the camera. */
  nonce: number;
}

interface TripPlannerValue {
  // Trip inputs
  origin: string;
  originCoords: Coordinates | null;
  destinations: DestinationEntry[];
  departAt: Date | null;
  setOriginText: (text: string) => void;
  setOriginFromCoords: (coords: Coordinates, label: string) => void;
  updateDestination: (index: number, value: string) => void;
  addDestination: () => void;
  removeDestination: (index: number) => void;
  reorderDestination: (index: number) => void;
  changeDeparture: (next: Date | null) => void;
  setDepartureTime: (time: Date) => void;
  setDepartureDay: (dayOffset: number) => void;

  // Place search
  activeField: ActiveField;
  suggestions: PlaceSuggestion[];
  isSearchingPlaces: boolean;
  runPlacesSearch: (field: ActiveField, text: string) => void;
  selectSuggestion: (suggestion: PlaceSuggestion) => void;
  closeSuggestions: () => void;

  // Results
  isLoading: boolean;
  routeError: string;
  antRouteOptions: RouteOption[];
  baselineModelOptions: RouteOption[];
  baselineNotice: string | null;
  trafficNote: string | null;
  /** The recorded inputs of the window the routes were planned on ("Inputs used" panel). */
  recordedInputs: RecordedInputs | null;
  tripEvaluation: TripEvaluation | null;
  tripEvaluationError: string;
  tripMetricRows: ComparisonMetricRow[];
  hasPlannedTrip: boolean;

  // Simulation mode: plan on a chosen recorded window instead of the departure time's own.
  simulationWindow: string | null;
  setSimulationWindow: (window: string | null) => void;

  // Selection
  selectedModelTab: ModelTab;
  setSelectedModelTab: (tab: ModelTab) => void;
  selectedIndex: number;
  setSelectedIndex: (index: number) => void;
  selectedRoute: RouteOption | undefined;
  antTop: RouteOption | undefined;
  baselineTop: RouteOption | undefined;

  findRoutes: () => void;
  /** Replans from where the driver is now to the stops still ahead (navigation reroute). */
  replanFrom: (from: Coordinates, remaining: DestinationEntry[]) => Promise<void>;

  // Map camera
  cameraTarget: CameraTarget | null;
  requestCamera: (coords: Coordinates) => void;

  // GPS
  gpsCoords: Coordinates | null;
  setGpsCoords: (coords: Coordinates) => void;
  locationError: string;
  setLocationError: (message: string) => void;
}

const TripPlannerContext = createContext<TripPlannerValue | null>(null);

export function useTripPlanner(): TripPlannerValue {
  const value = useContext(TripPlannerContext);
  if (!value) throw new Error('useTripPlanner must be used inside <TripPlannerProvider>');
  return value;
}

export function TripPlannerProvider({ children }: { children: ReactNode }) {
  const [origin, setOrigin] = useState('');
  const [originCoords, setOriginCoords] = useState<Coordinates | null>(null);
  const [destinations, setDestinations] = useState<DestinationEntry[]>([
    { name: '', coords: null },
  ]);
  const [departAt, setDepartAt] = useState<Date | null>(null); // null = leave now

  const [isLoading, setIsLoading] = useState(false);
  const [routeError, setRouteError] = useState('');
  const [antRouteOptions, setAntRouteOptions] = useState<RouteOption[]>([]);
  const [baselineModelOptions, setBaselineModelOptions] = useState<RouteOption[]>([]);
  // Why the baseline has no route for this trip, when it has none (a result, not an error).
  const [baselineNotice, setBaselineNotice] = useState<string | null>(null);
  const [trafficNote, setTrafficNote] = useState<string | null>(null);
  const [recordedInputs, setRecordedInputs] = useState<RecordedInputs | null>(null);
  const [simulationWindow, setSimulationWindowState] = useState<string | null>(null);
  const [tripEvaluation, setTripEvaluation] = useState<TripEvaluation | null>(null);
  const [tripEvaluationError, setTripEvaluationError] = useState('');

  const [selectedIndex, setSelectedIndex] = useState(0);
  const [selectedModelTab, setSelectedModelTabState] = useState<ModelTab>('antroute');

  const [activeField, setActiveField] = useState<ActiveField>(null);
  const [suggestions, setSuggestions] = useState<PlaceSuggestion[]>([]);
  const [isSearchingPlaces, setIsSearchingPlaces] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const searchRequestRef = useRef(0);

  const [cameraTarget, setCameraTarget] = useState<CameraTarget | null>(null);
  const cameraNonce = useRef(0);
  const [gpsCoords, setGpsCoordsState] = useState<Coordinates | null>(null);
  const gpsRef = useRef<Coordinates | null>(null);
  const [locationError, setLocationError] = useState('');

  const requestCamera = useCallback((coords: Coordinates) => {
    cameraNonce.current += 1;
    setCameraTarget({ coords, nonce: cameraNonce.current });
  }, []);

  const setGpsCoords = useCallback((coords: Coordinates) => {
    gpsRef.current = coords;
    setGpsCoordsState(coords);
  }, []);

  const clearRouteResults = useCallback(() => {
    setAntRouteOptions([]);
    setBaselineModelOptions([]);
    setBaselineNotice(null);
    setTrafficNote(null);
    setRecordedInputs(null);
  }, []);

  // Simulation shows the chosen day's overlays (cameras, incidents, weather, flood) as soon as
  // the window is picked, before any route is planned; a planned trip then replaces them with
  // the same window's inputs from the plan itself.
  useEffect(() => {
    if (simulationWindow === null) return;
    let cancelled = false;
    getRecordedInputs(simulationWindow)
      .then((inputs) => !cancelled && setRecordedInputs(inputs))
      .catch(() => !cancelled && setRecordedInputs(null));
    return () => {
      cancelled = true;
    };
  }, [simulationWindow]);

  // Routes were planned on one recorded window; a different one needs new routes.
  const setSimulationWindow = useCallback(
    (window: string | null) => {
      setSimulationWindowState(window);
      clearRouteResults();
    },
    [clearRouteResults]
  );

  const runPlacesSearch = useCallback(
    (field: ActiveField, text: string) => {
      setActiveField(field);

      if (debounceRef.current) clearTimeout(debounceRef.current);
      const requestId = ++searchRequestRef.current;

      if (text.trim().length < 2) {
        setSuggestions([]);
        setIsSearchingPlaces(false);
        return;
      }

      debounceRef.current = setTimeout(async () => {
        setIsSearchingPlaces(true);
        const near = field === 'origin' ? gpsRef.current : (originCoords ?? gpsRef.current);
        const results = await searchPlaces(text, near);
        // A slower earlier request must not overwrite a newer one's results.
        if (requestId !== searchRequestRef.current) return;
        setSuggestions(results);
        setIsSearchingPlaces(false);
      }, 300);
    },
    [originCoords]
  );

  const closeSuggestions = useCallback(() => {
    setSuggestions([]);
    setActiveField(null);
  }, []);

  const setOriginText = useCallback(
    (text: string) => {
      setOrigin(text);
      setOriginCoords(null);
      clearRouteResults();
      runPlacesSearch('origin', text);
    },
    [clearRouteResults, runPlacesSearch]
  );

  const setOriginFromCoords = useCallback(
    (coords: Coordinates, label: string) => {
      setGpsCoords(coords);
      setOriginCoords(coords);
      setOrigin(label);
      clearRouteResults();
      requestCamera(coords);
    },
    [clearRouteResults, requestCamera, setGpsCoords]
  );

  const updateDestination = useCallback(
    (index: number, value: string) => {
      setDestinations((prev) =>
        prev.map((d, i) => (i === index ? { name: value, coords: null } : d))
      );
      clearRouteResults();
      runPlacesSearch(index, value);
    },
    [clearRouteResults, runPlacesSearch]
  );

  const addDestination = useCallback(() => {
    setDestinations((prev) => [...prev, { name: '', coords: null }]);
  }, []);

  const removeDestination = useCallback((index: number) => {
    setDestinations((prev) => {
      const next = prev.filter((_, i) => i !== index);
      // Never leave the panel with no row at all.
      return next.length > 0 ? next : [{ name: '', coords: null }];
    });
  }, []);

  /** Swaps a stop with the one after it, wrapping at the end. */
  const reorderDestination = useCallback((index: number) => {
    setDestinations((prev) => {
      if (prev.length < 2) return prev;
      const next = [...prev];
      const targetIndex = (index + 1) % next.length;
      [next[index], next[targetIndex]] = [next[targetIndex], next[index]];
      return next;
    });
  }, []);

  const selectSuggestion = useCallback(
    (suggestion: PlaceSuggestion) => {
      const coords: Coordinates = { latitude: suggestion.lat, longitude: suggestion.lng };
      searchRequestRef.current++;

      if (activeField === 'origin') {
        setOrigin(suggestion.formattedAddress);
        setOriginCoords(coords);
        requestCamera(coords);
      } else if (typeof activeField === 'number') {
        const index = activeField;
        setDestinations((prev) =>
          prev.map((d, i) => (i === index ? { name: suggestion.formattedAddress, coords } : d))
        );
        requestCamera(coords);
      }
      setSuggestions([]);
      setActiveField(null);
    },
    [activeField, requestCamera]
  );

  // Routes were planned for one departure time; a new one needs new routes.
  const changeDeparture = useCallback(
    (next: Date | null) => {
      setDepartAt(next ? clampDeparture(next) : null);
      clearRouteResults();
    },
    [clearRouteResults]
  );

  /** A time on the departure's current day. A time already past today means tomorrow. */
  const setDepartureTime = useCallback(
    (time: Date) => {
      const next = new Date(departAt ?? currentMinute());
      next.setHours(time.getHours(), time.getMinutes(), 0, 0);
      if (next < currentMinute()) next.setDate(next.getDate() + 1);
      changeDeparture(next);
    },
    [departAt, changeDeparture]
  );

  const setDepartureDay = useCallback(
    (dayOffset: number) => {
      const next = new Date(departAt ?? currentMinute());
      const today = new Date();
      next.setFullYear(today.getFullYear(), today.getMonth(), today.getDate() + dayOffset);
      changeDeparture(next);
    },
    [departAt, changeDeparture]
  );

  const setSelectedModelTab = useCallback((tab: ModelTab) => {
    setSelectedModelTabState(tab);
    setSelectedIndex(0);
  }, []);

  /**
   * Plans a trip from explicit arguments rather than from state, so a caller that has just
   * computed a new origin and stop list (the navigation reroute) does not have to wait for
   * a state update to land first.
   */
  const planTrip = useCallback(
    async (
      originToUse: string,
      originCoordsToUse: Coordinates | null,
      destinationsToUse: DestinationEntry[],
      departAtToUse: Date | null,
      recordedWindowToUse: string | null = null
    ) => {
      const cleaned: DestinationInput[] = destinationsToUse
        .map((d) => ({
          name: d.name.trim(),
          lat: d.coords?.latitude,
          lng: d.coords?.longitude,
        }))
        .filter((d) => d.name.length > 0);

      if (!originToUse.trim()) {
        setRouteError('Add a starting point.');
        return;
      }
      if (cleaned.length === 0) {
        setRouteError('Add at least one destination.');
        return;
      }

      setRouteError('');
      setIsLoading(true);
      setTripEvaluation(null);
      setTripEvaluationError('');
      try {
        const [antResult, baselineResult, evaluationResult] = await Promise.allSettled([
          planRoute(originToUse.trim(), cleaned, originCoordsToUse, 'antroute', departAtToUse, recordedWindowToUse),
          planRoute(originToUse.trim(), cleaned, originCoordsToUse, 'baseline', departAtToUse, recordedWindowToUse),
          // Simulated trips have no Google Maps travel times to be evaluated against.
          recordedWindowToUse
            ? Promise.resolve<TripEvaluation>({
                status: 'not_evaluated',
                message:
                  'Simulation mode: this trip is planned on a recorded window and is not part of the Google Maps evaluation.',
                tripId: null,
                baselineName: 'Improved ACO (Cheng 2023)',
                routeOptimality: null,
                metrics: [],
              })
            : getTripEvaluation(originToUse.trim(), cleaned, originCoordsToUse, departAtToUse),
        ]);

        if (evaluationResult.status === 'fulfilled') {
          setTripEvaluation(evaluationResult.value);
        } else {
          setTripEvaluationError(
            evaluationResult.reason?.message || 'Could not load this trip’s evaluation.'
          );
        }

        if (antResult.status === 'rejected') throw antResult.reason;
        const antPlan = antResult.value;
        // A failed baseline must not hide ANTRoute's routes; its reason goes in the notice.
        const baselinePlan =
          baselineResult.status === 'fulfilled'
            ? baselineResult.value
            : {
                routes: [],
                notice:
                  baselineResult.reason?.message || 'The baseline route could not be computed.',
              };

        setAntRouteOptions(antPlan.routes);
        setBaselineModelOptions(baselinePlan.routes);
        setBaselineNotice(baselinePlan.notice);
        setTrafficNote(antPlan.trafficNote);
        setRecordedInputs(antPlan.recordedInputs);
        setSelectedIndex(0);
      } catch (error: any) {
        setRouteError(error?.message || 'Something went wrong. Please try again.');
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  /** Replans from where the driver is now to the stops still ahead. */
  const replanFrom = useCallback(
    (from: Coordinates, remaining: DestinationEntry[]) => {
      setOrigin('Current location');
      setOriginCoords(from);
      setDestinations(remaining);
      // Rerouting means leaving now, whatever the trip was originally planned for.
      setDepartAt(null);
      return planTrip('Current location', from, remaining, null, simulationWindow);
    },
    [planTrip, simulationWindow]
  );

  const findRoutes = useCallback(() => {
    const nonEmpty = destinations.filter((d) => d.name.trim().length > 0);
    if (nonEmpty.length !== destinations.length) {
      setDestinations(nonEmpty.length > 0 ? nonEmpty : [{ name: '', coords: null }]);
    }
    return planTrip(origin, originCoords, nonEmpty, departAt, simulationWindow);
  }, [destinations, origin, originCoords, departAt, simulationWindow, planTrip]);

  const currentModelOptions =
    selectedModelTab === 'baseline' ? baselineModelOptions : antRouteOptions;
  const selectedRoute = currentModelOptions[selectedIndex];
  const antTop = antRouteOptions[0];
  const baselineTop = baselineModelOptions[0];

  /**
   * This trip's evaluation metrics, exactly the rows the server computed. The backend omits
   * a metric it cannot score for this trip rather than sending a placeholder, so there is no
   * fixed row list to pad against -- an unevaluated trip simply has none, and the panel shows
   * tripEvaluation.message instead.
   */
  const tripMetricRows: ComparisonMetricRow[] = useMemo(
    () => tripEvaluation?.metrics ?? [],
    [tripEvaluation]
  );

  const value: TripPlannerValue = {
    origin,
    originCoords,
    destinations,
    departAt,
    setOriginText,
    setOriginFromCoords,
    updateDestination,
    addDestination,
    removeDestination,
    reorderDestination,
    changeDeparture,
    setDepartureTime,
    setDepartureDay,

    activeField,
    suggestions,
    isSearchingPlaces,
    runPlacesSearch,
    selectSuggestion,
    closeSuggestions,

    isLoading,
    routeError,
    antRouteOptions,
    baselineModelOptions,
    baselineNotice,
    trafficNote,
    recordedInputs,
    tripEvaluation,
    tripEvaluationError,
    tripMetricRows,
    hasPlannedTrip: antRouteOptions.length > 0 || baselineModelOptions.length > 0,

    simulationWindow,
    setSimulationWindow,

    selectedModelTab,
    setSelectedModelTab,
    selectedIndex,
    setSelectedIndex,
    selectedRoute,
    antTop,
    baselineTop,

    findRoutes,
    replanFrom,

    cameraTarget,
    requestCamera,

    gpsCoords,
    setGpsCoords,
    locationError,
    setLocationError,
  };

  return <TripPlannerContext.Provider value={value}>{children}</TripPlannerContext.Provider>;
}
