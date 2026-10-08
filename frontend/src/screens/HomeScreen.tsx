import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  Text,
  KeyboardAvoidingView,
  Platform,
  Keyboard,
  StatusBar,
  ScrollView,
  ActivityIndicator,
  Dimensions,
  Alert,
  Animated,
  PanResponder,
} from 'react-native';

import MapView, { PROVIDER_GOOGLE, Marker, Polyline } from 'react-native-maps';
import Constants, { ExecutionEnvironment } from 'expo-constants';
import { useIsFocused } from '@react-navigation/native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import {
  X,
  ArrowUpDown,
  Info,
  CarFront,
  Search,
  LocateFixed,
  Navigation,
  Plus,
  MapPin,
  Clock,
} from 'lucide-react-native';

import * as Location from 'expo-location';
import DateTimePicker, { DateTimePickerAndroid } from '@react-native-community/datetimepicker';

import {
  planRoute,
  getTripEvaluation,
  ComparisonMetricRow,
  TripEvaluation,
  RouteOption,
  CongestionLevel,
  Coordinates,
  DestinationInput,
} from '../api/routeService';
import { searchPlaces, reverseGeocode, PlaceSuggestion } from '../api/placesService';

const SCREEN_HEIGHT = Dimensions.get('window').height;

// Google Maps everywhere it can render: always on Android, and on iOS in a build made
// with GOOGLE_MAPS_API_KEY (see app.config.js). Expo Go on iPhone has no Google Maps
// SDK, so there it stays on Apple Maps rather than failing to load the map.
const USE_GOOGLE_MAPS =
  Platform.OS === 'android' ||
  (Constants.executionEnvironment !== ExecutionEnvironment.StoreClient &&
    Constants.expoConfig?.extra?.iosGoogleMaps === true);

// Route panel heights the drag handle settles at (low, default, full), and the height
// below which letting go closes the panel back to the search bar. Fractions of the space
// above the keyboard, not of the screen: when the keyboard opens that space shrinks, and
// the panel has to shrink with it or it fills everything up to the status bar.
const SHEET_SNAPS = [0.4, 0.7, 0.92];
const SHEET_DEFAULT = SHEET_SNAPS[1];
const SHEET_FULL = SHEET_SNAPS[SHEET_SNAPS.length - 1];
const SHEET_CLOSE_BELOW = 0.25;

const CONGESTION_COLORS: Record<CongestionLevel, string> = {
  clear: '#10b981',
  moderate: '#f59e0b',
  heavy: '#ef4444',
  unknown: '#d1d5db',
};

const CONGESTION_WIDTH: Record<CongestionLevel, `${number}%`> = {
  clear: '30%',
  moderate: '60%',
  heavy: '85%',
  unknown: '0%',
};

/** Names the algorithm that actually produced a baseline route, which is not always IACO. */
function baselineMethodLabel(route?: RouteOption): string {
  switch (route?.algorithm) {
    case 'shortest_distance':
      return 'Shortest distance (IACO found no route)';
    case 'mixed':
      return 'IACO + shortest distance';
    default:
      return 'Improved ACO (Cheng 2023)';
  }
}

type ActiveField = 'origin' | number | null;

interface DestinationEntry {
  name: string;
  coords: Coordinates | null;
}

type ModelTab = 'antroute' | 'baseline' | 'comparison';

function formatClock(date: Date): string {
  let hours = date.getHours();
  const minutes = date.getMinutes().toString().padStart(2, '0');
  const ampm = hours >= 12 ? 'PM' : 'AM';
  hours = hours % 12 || 12;
  return `${hours}:${minutes} ${ampm}`;
}

// departAt null means leaving now.
function getArrivalTime(durationMin: number, departAt: Date | null): string {
  const arrival = departAt ? new Date(departAt) : new Date();
  arrival.setMinutes(arrival.getMinutes() + durationMin);
  return formatClock(arrival);
}

// Departure picker: any minute, today or tomorrow only.
function startOfDay(date: Date): Date {
  const d = new Date(date);
  d.setHours(0, 0, 0, 0);
  return d;
}

function daysFromToday(date: Date): number {
  return Math.round((startOfDay(date).getTime() - startOfDay(new Date()).getTime()) / 86400000);
}

// The current minute -- the earliest a planned departure can be.
function currentMinute(): Date {
  const d = new Date();
  d.setSeconds(0, 0);
  return d;
}

// Keep a planned departure between now and the last minute of tomorrow.
function clampDeparture(date: Date): Date {
  const earliest = currentMinute();
  const latest = startOfDay(new Date());
  latest.setDate(latest.getDate() + 2);
  latest.setMinutes(-1);
  if (date < earliest) return earliest;
  if (date > latest) return latest;
  return date;
}

function formatDeparture(date: Date): string {
  return `${daysFromToday(date) === 1 ? 'Tomorrow ' : ''}${formatClock(date)}`;
}

// Colour behind the status bar (the map). Change this if you use a dark map style.
const STATUS_BAR_BACKGROUND = '#f5f5f5';

// Locate button: distance below the safe area top, and its square size.
const MAP_BUTTON_TOP = 8;
const MAP_BUTTON_SIZE = 44;

// Light background -> dark icons, dark background -> light icons
function getStatusBarStyle(bgHex: string): 'dark-content' | 'light-content' {
  const hex = bgHex.replace('#', '');
  const full = hex.length === 3 ? hex.split('').map((c) => c + c).join('') : hex;
  const r = parseInt(full.slice(0, 2), 16);
  const g = parseInt(full.slice(2, 4), 16);
  const b = parseInt(full.slice(4, 6), 16);
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance > 0.5 ? 'dark-content' : 'light-content';
}

export default function HomeScreen({ navigation }: any) {
  const mapRef = useRef<MapView | null>(null);
  const insets = useSafeAreaInsets();

  const [isExpanded, setIsExpanded] = useState(false);
  const [origin, setOrigin] = useState('');
  const [originCoords, setOriginCoords] = useState<Coordinates | null>(null);
  const [isLocating, setIsLocating] = useState(false);
  const [locationError, setLocationError] = useState('');
  const [destinations, setDestinations] = useState<DestinationEntry[]>([{ name: '', coords: null }]);
  const [departAt, setDepartAt] = useState<Date | null>(null); // null = leave now

  const [isLoading, setIsLoading] = useState(false);
  const [routeError, setRouteError] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [activeTab, setActiveTab] = useState<ModelTab>('antroute');
  const [selectedModelTab, setSelectedModelTab] = useState<'antroute' | 'baseline'>('antroute');

  const handleTabChange = (tab: ModelTab) => {
    setActiveTab(tab);
    if (tab !== 'comparison') {
      setSelectedModelTab(tab);
      setSelectedIndex(0);
    }
  };

  const [normalRoute, setNormalRoute] = useState<RouteOption | null>(null);
  const [antRouteOptions, setAntRouteOptions] = useState<RouteOption[]>([]);
  const [baselineModelOptions, setBaselineModelOptions] = useState<RouteOption[]>([]);
  // Why the baseline has no route for this trip, when it has none (a result, not an error).
  const [baselineNotice, setBaselineNotice] = useState<string | null>(null);
  const [trafficNote, setTrafficNote] = useState<string | null>(null);

  // The evaluation of the trip just planned: its own metrics when it is a test trip.
  const [tripEvaluation, setTripEvaluation] = useState<TripEvaluation | null>(null);
  const [tripEvaluationError, setTripEvaluationError] = useState('');

  const [isNavigating, setIsNavigating] = useState(false);
  const [isAddingStop, setIsAddingStop] = useState(false);
  const [stopsRevealedDuringNav, setStopsRevealedDuringNav] = useState(false);

  const [activeField, setActiveField] = useState<ActiveField>(null);
  const [suggestions, setSuggestions] = useState<PlaceSuggestion[]>([]);
  const [isSearchingPlaces, setIsSearchingPlaces] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const searchRequestRef = useRef(0);

  const isFocused = useIsFocused();

  const gpsRef = useRef<Coordinates | null>(null);

  const getAccuratePosition = async (): Promise<Location.LocationObject> => {
    const TARGET_ACCURACY_M = 30;
    const accuracyOf = (p: Location.LocationObject) => p.coords.accuracy ?? Infinity;

    let best = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.High });
    if (accuracyOf(best) <= TARGET_ACCURACY_M) return best;

    return new Promise((resolve) => {
      let sub: Location.LocationSubscription | null = null;
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        sub?.remove();
        resolve(best);
      };
      const timer = setTimeout(finish, 8000);

      Location.watchPositionAsync(
        { accuracy: Location.Accuracy.BestForNavigation, timeInterval: 1000, distanceInterval: 0 },
        (pos) => {
          if (accuracyOf(pos) < accuracyOf(best)) best = pos;
          if (accuracyOf(best) <= TARGET_ACCURACY_M) finish();
        }
      )
        .then((subscription) => {
          sub = subscription;
          if (done) subscription.remove();
        })
        .catch(finish);
    });
  };

  const [hasLocationPermission, setHasLocationPermission] = useState(false);
  const [isCentering, setIsCentering] = useState(false);

  useEffect(() => {
    Location.getForegroundPermissionsAsync()
      .then(({ status }) => setHasLocationPermission(status === 'granted'))
      .catch(() => {});
  }, []);

  const centerOnMyLocation = async () => {
    setIsCentering(true);
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Location permission needed', 'Allow location access to see where you are on the map.');
        return;
      }
      setHasLocationPermission(true);

      const position = await getAccuratePosition();
      gpsRef.current = { latitude: position.coords.latitude, longitude: position.coords.longitude };
      mapRef.current?.animateToRegion(
        {
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          latitudeDelta: 0.01,
          longitudeDelta: 0.01,
        },
        800
      );
    } catch {
      Alert.alert('Could not get your location', 'Please check that location services are on and try again.');
    } finally {
      setIsCentering(false);
    }
  };

  const fetchCurrentLocation = async () => {
    setIsLocating(true);
    setLocationError('');
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setLocationError('Location permission denied.');
        setOrigin('');
        setOriginCoords(null);
        return;
      }

      setHasLocationPermission(true);

      const position = await getAccuratePosition();
      const coords: Coordinates = {
        latitude: position.coords.latitude,
        longitude: position.coords.longitude,
      };
      gpsRef.current = coords;
      setOriginCoords(coords);

      mapRef.current?.animateToRegion(
        {
          latitude: coords.latitude,
          longitude: coords.longitude,
          latitudeDelta: 0.01,
          longitudeDelta: 0.01,
        },
        800
      );

      const label = await reverseGeocode(coords.latitude, coords.longitude);
      setOrigin(label || 'Current location');
    } catch {
      setLocationError('Could not get your location.');
      setOrigin('');
      setOriginCoords(null);
    } finally {
      setIsLocating(false);
    }
  };

  const sheetHeight = useRef(new Animated.Value(SHEET_DEFAULT)).current; // fraction of the space
  const sheetHeightPct = sheetHeight.interpolate({ inputRange: [0, 1], outputRange: ['0%', '100%'] });
  const sheetRestRef = useRef(SHEET_DEFAULT); // height the panel last settled at
  const dragStartRef = useRef(SHEET_DEFAULT); // height when the current drag began
  const sheetSpaceRef = useRef(SCREEN_HEIGHT); // px the panel can use, to turn drags into fractions
  const dragSpaceRef = useRef(SCREEN_HEIGHT); // that space when the current drag began

  const snapSheet = (to: number) => {
    sheetRestRef.current = to;
    Animated.spring(sheetHeight, { toValue: to, useNativeDriver: false, bounciness: 0 }).start();
  };

  const handleExpand = () => {
    sheetHeight.setValue(SHEET_DEFAULT);
    sheetRestRef.current = SHEET_DEFAULT;
    setIsExpanded(true);
  };
  const handleCollapse = () => setIsExpanded(false);

  // Created once, so it only touches refs and state setters (both stable).
  const sheetPanResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: (_, g) => Math.abs(g.dy) > 4,
      // Not dismissing the keyboard here: the panel's space would grow mid-drag and the
      // panel would jump up under the finger that is trying to pull it down.
      onPanResponderGrant: () => {
        dragSpaceRef.current = sheetSpaceRef.current;
        sheetHeight.stopAnimation((value) => {
          dragStartRef.current = value;
        });
      },
      onPanResponderMove: (_, g) => {
        const next = dragStartRef.current - g.dy / dragSpaceRef.current;
        sheetHeight.setValue(Math.min(Math.max(next, 0), SHEET_FULL));
      },
      onPanResponderRelease: (_, g) => {
        // A tap or a tiny wiggle leaves the panel where it was.
        if (Math.abs(g.dy) < 8 && Math.abs(g.vy) < 0.3) {
          snapSheet(sheetRestRef.current);
          return;
        }
        // Where the panel would end up with the flick's momentum (vy > 0 = downward).
        const projected = dragStartRef.current - (g.dy + g.vy * 200) / dragSpaceRef.current;
        if (projected < SHEET_CLOSE_BELOW) {
          Keyboard.dismiss();
          setIsExpanded(false);
          return;
        }
        const nearest = SHEET_SNAPS.reduce((best, h) =>
          Math.abs(h - projected) < Math.abs(best - projected) ? h : best
        );
        snapSheet(nearest);
      },
      onPanResponderTerminate: () => snapSheet(sheetRestRef.current),
    })
  ).current;

  const runPlacesSearch = (field: ActiveField, text: string) => {
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
      const near = field === 'origin' ? gpsRef.current : originCoords ?? gpsRef.current;
      const results = await searchPlaces(text, near);
      if (requestId !== searchRequestRef.current) return;
      setSuggestions(results);
      setIsSearchingPlaces(false);
    }, 300);
  };

  const clearRouteResults = () => {
    setNormalRoute(null);
    setAntRouteOptions([]);
    setBaselineModelOptions([]);
    setBaselineNotice(null);
    setTrafficNote(null);
  };

  // Routes were planned for one departure time; a new one needs new routes.
  const changeDeparture = (next: Date | null) => {
    setDepartAt(next ? clampDeparture(next) : null);
    clearRouteResults();
  };

  // A time picked from the clock, on the departure's current day. A time that has
  // already passed today means tomorrow (the Tomorrow chip lights up to show it).
  const setDepartureTime = (time: Date) => {
    const next = new Date(departAt ?? currentMinute());
    next.setHours(time.getHours(), time.getMinutes(), 0, 0);
    if (next < currentMinute()) next.setDate(next.getDate() + 1);
    changeDeparture(next);
  };

  const openTimePicker = () => {
    if (!departAt) return;
    DateTimePickerAndroid.open({
      value: departAt,
      mode: 'time',
      is24Hour: false,
      onValueChange: (_, time) => setDepartureTime(time),
    });
  };

  const setDepartureDay = (dayOffset: number) => {
    const next = new Date(departAt ?? currentMinute());
    const today = new Date();
    next.setFullYear(today.getFullYear(), today.getMonth(), today.getDate() + dayOffset);
    changeDeparture(next);
  };

  const handleOriginChange = (text: string) => {
    setOrigin(text);
    setOriginCoords(null);
    clearRouteResults();
    runPlacesSearch('origin', text);
  };

  const updateDestination = (index: number, value: string) => {
    setDestinations((prev) => prev.map((d, i) => (i === index ? { name: value, coords: null } : d)));

    if (!isNavigating) {
      clearRouteResults();
    }
    runPlacesSearch(index, value);
  };

  const isRowEditable = (index: number) => {
    if (!isNavigating) return true;
    return isAddingStop && index === destinations.length - 1;
  };

  const selectSuggestion = (suggestion: PlaceSuggestion) => {
    const coords: Coordinates = { latitude: suggestion.lat, longitude: suggestion.lng };
    searchRequestRef.current++;

    if (activeField === 'origin') {
      setOrigin(suggestion.formattedAddress);
      setOriginCoords(coords);
      mapRef.current?.animateToRegion(
        { latitude: coords.latitude, longitude: coords.longitude, latitudeDelta: 0.01, longitudeDelta: 0.01 },
        600
      );
    } else if (typeof activeField === 'number') {
      const updatedDestinations = destinations.map((d, i) =>
        i === activeField ? { name: suggestion.formattedAddress, coords } : d
      );
      setDestinations(updatedDestinations);

      if (isNavigating && isAddingStop && activeField === updatedDestinations.length - 1) {
        setIsAddingStop(false);
        handleFindRoutes(updatedDestinations, origin, originCoords);
      }
    }
    setSuggestions([]);
    setActiveField(null);
    Keyboard.dismiss();
  };

  const removeDestination = (index: number) => {
    setDestinations((prev) => prev.filter((_, i) => i !== index));
  };

  const addDestination = () => {
    setDestinations((prev) => [...prev, { name: '', coords: null }]);
  };

  const reorderDestination = (index: number) => {
    setDestinations((prev) => {
      if (prev.length < 2) return prev;
      const next = [...prev];
      const targetIndex = (index + 1) % next.length;
      [next[index], next[targetIndex]] = [next[targetIndex], next[index]];
      return next;
    });
  };

  const handleFindRoutes = async (
    destinationsToUse: DestinationEntry[],
    originToUse: string,
    originCoordsToUse: Coordinates | null
  ) => {
    const cleanedDestinations: DestinationInput[] = destinationsToUse
      .map((d) => ({ name: d.name.trim(), lat: d.coords?.latitude, lng: d.coords?.longitude }))
      .filter((d) => d.name.length > 0);

    if (!originToUse.trim() || cleanedDestinations.length === 0) {
      setRouteError('Add at least one destination.');
      return;
    }

    setRouteError('');
    setIsLoading(true);
    try {
      setTripEvaluation(null);
      setTripEvaluationError('');
      const [antResult, baselineResult, evaluationResult] = await Promise.allSettled([
        planRoute(originToUse.trim(), cleanedDestinations, originCoordsToUse, 'antroute', departAt),
        planRoute(originToUse.trim(), cleanedDestinations, originCoordsToUse, 'baseline', departAt),
        getTripEvaluation(originToUse.trim(), cleanedDestinations, originCoordsToUse, departAt),
      ]);
      if (evaluationResult.status === 'fulfilled') {
        setTripEvaluation(evaluationResult.value);
      } else {
        setTripEvaluationError(evaluationResult.reason?.message || 'Could not load this trip’s evaluation.');
      }
      if (antResult.status === 'rejected') throw antResult.reason;
      const antPlan = antResult.value;
      // A failed baseline should not hide ANTRoute's routes; its reason goes in the notice.
      const baselinePlan =
        baselineResult.status === 'fulfilled'
          ? baselineResult.value
          : { routes: [], notice: baselineResult.reason?.message || 'The baseline route could not be computed.' };

      // The baseline (Improved ACO, Cheng 2023) has one optimal path; it is drawn as the
      // comparison route next to ANTRoute's options.
      setNormalRoute(baselinePlan.routes[0] ?? null);
      setAntRouteOptions(antPlan.routes);
      setBaselineModelOptions(baselinePlan.routes);
      setBaselineNotice(baselinePlan.notice);
      setTrafficNote(antPlan.trafficNote);
      setSelectedIndex(0);
    } catch (error: any) {
      setRouteError(error?.message || 'Something went wrong. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFindRoutesPress = () => {
    const nonEmptyDestinations = destinations.filter((d) => d.name.trim().length > 0);
    if (nonEmptyDestinations.length !== destinations.length) {
      setDestinations(nonEmptyDestinations.length > 0 ? nonEmptyDestinations : [{ name: '', coords: null }]);
    }

    handleFindRoutes(nonEmptyDestinations, origin, originCoords);
  };

  const handleStartNavigation = () => {
    // Starting means leaving now: arrival times and any re-plan (adding a stop) should
    // count from now, not from the departure the routes were planned for.
    setDepartAt(null);
    setIsNavigating(true);
    setStopsRevealedDuringNav(false);
    setIsAddingStop(false);
    Keyboard.dismiss();
    setSuggestions([]);
  };

  const handleEndNavigation = () => {
    setIsNavigating(false);
    setIsAddingStop(false);
    setStopsRevealedDuringNav(false);

    setOrigin('');
    setOriginCoords(null);
    setLocationError('');
    setDestinations([{ name: '', coords: null }]);
    setDepartAt(null);
    clearRouteResults();
    setSelectedIndex(0);
    setActiveTab('antroute');
    setSelectedModelTab('antroute');
    setRouteError('');
    setActiveField(null);
    setSuggestions([]);
  };

  const handleAddStopWhileNavigating = () => {
    const newIndex = destinations.length;
    setDestinations((prev) => [...prev, { name: '', coords: null }]);
    setIsAddingStop(true);
    setStopsRevealedDuringNav(true);
    setActiveField(newIndex);
  };

  const handleExpandNavStops = () => {
    setStopsRevealedDuringNav(true);
  };

  const handleCollapseNavStops = () => {
    if (isAddingStop) {
      setDestinations((prev) => prev.slice(0, -1));
      setIsAddingStop(false);
    }
    setStopsRevealedDuringNav(false);
    setActiveField(null);
    setSuggestions([]);
  };

  const renderSuggestions = (field: ActiveField) => {
    if (activeField !== field) return null;
    if (isSearchingPlaces) {
      return (
        <View style={styles.suggestionsBox}>
          <ActivityIndicator size="small" color="#4475F2" style={{ padding: 12 }} />
        </View>
      );
    }
    if (suggestions.length === 0) return null;

    return (
      <View style={styles.suggestionsBox}>
        {suggestions.map((item) => (
          <TouchableOpacity key={item.id} style={styles.suggestionRow} onPress={() => selectSuggestion(item)}>
            <Text style={styles.suggestionName} numberOfLines={1}>{item.name}</Text>
            {item.address || item.distanceKm != null ? (
              <Text style={styles.suggestionAddress} numberOfLines={1}>
                {[item.address, item.distanceKm != null ? `${item.distanceKm} km` : null]
                  .filter(Boolean)
                  .join(' · ')}
              </Text>
            ) : null}
          </TouchableOpacity>
        ))}
      </View>
    );
  };

  const currentModelOptions = selectedModelTab === 'baseline' ? baselineModelOptions : antRouteOptions;
  const selectedRoute = currentModelOptions[selectedIndex];
  const antTop = antRouteOptions[0];
  const baselineTop = baselineModelOptions[0];
  // Route Optimality of this trip's own evaluated routes; null unless it is a test trip.
  const tripOptimality = tripEvaluation?.status === 'evaluated' ? tripEvaluation.routeOptimality : null;

  // This trip's evaluation metrics, always in the same six rows; a metric the trip has no
  // value for (not timed in Apple Maps, or R² on a single-leg trip) shows a dash.
  const tripMetricRows: ComparisonMetricRow[] = ['Route Optimality', 'MAE', 'RMSE', 'MSE', 'MAPE', 'R²'].map(
    (metric) =>
      tripEvaluation?.metrics.find((m) => m.metric === metric) ?? {
        metric,
        antroute: '–',
        baseline: '–',
        improvementPct: null,
        higherIsBetter: metric === 'Route Optimality' || metric === 'R²',
        pValue: null,
        significant: null,
      }
  );

  const renderMetricsTable = (rows: ComparisonMetricRow[]) => (
    <View style={styles.metricsTable}>
      <View style={styles.metricsTableHeaderRow}>
        <Text style={[styles.metricsTableCell, styles.metricsTableHeaderText, { flex: 1.5 }]}>Metric</Text>
        <Text style={[styles.metricsTableCell, styles.metricsTableHeaderText]}>ANTRoute</Text>
        <Text style={[styles.metricsTableCell, styles.metricsTableHeaderText]}>Baseline</Text>
        <Text style={[styles.metricsTableCell, styles.metricsTableHeaderText]}>Improvement</Text>
      </View>
      {rows.map((row) => {
        const isGood = row.improvementPct != null && row.improvementPct > 0;
        return (
          <View key={row.metric} style={styles.metricsTableRow}>
            <Text style={[styles.metricsTableCell, { flex: 1.5 }]}>
              {row.metric}
              {row.significant ? ' *' : ''}
            </Text>
            <Text style={styles.metricsTableCell}>{row.antroute}</Text>
            <Text style={styles.metricsTableCell}>{row.baseline}</Text>
            <Text style={[styles.metricsTableCell, styles.metricsTableImprovement, isGood && styles.improvementGood]}>
              {row.improvementPct == null ? '–' : `${row.improvementPct > 0 ? '+' : ''}${row.improvementPct}%`}
            </Text>
          </View>
        );
      })}
    </View>
  );
  const hasAnyResults = normalRoute !== null || antRouteOptions.length > 0;
  const showStopsList = !isNavigating || stopsRevealedDuringNav;

  const lastPlacedIndex = destinations.reduce((last, d, i) => (d.coords ? i : last), -1);

  const ANT_COLOR = '#3B6BE6';
  const BASELINE_COLOR = '#F7C441';

  useEffect(() => {
    const path =
      activeTab === 'comparison' && !isNavigating
        ? [...(antTop?.path ?? []), ...(baselineTop?.path ?? [])]
        : selectedRoute?.path ?? [];
    if (path.length > 1) {
      mapRef.current?.fitToCoordinates(path, {
        edgePadding: { top: 80, right: 40, bottom: SCREEN_HEIGHT * 0.55, left: 40 },
        animated: true,
      });
    }
  }, [antRouteOptions, baselineModelOptions, selectedIndex, activeTab, selectedModelTab]);

  return (
    <View style={styles.container}>
      {isFocused && (
        <StatusBar
          barStyle={getStatusBarStyle(STATUS_BAR_BACKGROUND)}
          backgroundColor="transparent"
          translucent={true}
        />
      )}

      <MapView
        ref={mapRef}
        provider={USE_GOOGLE_MAPS ? PROVIDER_GOOGLE : undefined}
        style={styles.map}
        showsUserLocation={hasLocationPermission}
        showsMyLocationButton={false}
        // Apple Maps places its compass inside the map's layout margins; a taller top
        // margin drops the compass just below the locate button. (On Google Maps, padding
        // would also shift the map centre, and its compass sits top-left anyway.)
        mapPadding={
          !USE_GOOGLE_MAPS
            ? { top: MAP_BUTTON_TOP + MAP_BUTTON_SIZE + 10, right: 8, bottom: 8, left: 8 }
            : undefined
        }
        initialRegion={{
          latitude: 14.5995,
          longitude: 120.9842,
          latitudeDelta: 0.05,
          longitudeDelta: 0.05,
        }}
        onPress={() => {
          Keyboard.dismiss();
          setSuggestions([]);
          if (isExpanded && !isNavigating) handleCollapse();
          if (isNavigating && stopsRevealedDuringNav) handleCollapseNavStops();
        }}
      >
        {originCoords && (
          <Marker coordinate={originCoords} anchor={{ x: 0.5, y: 0.5 }}>
            <View style={styles.currentLocationDotOuter}>
              <View style={styles.currentLocationDotInner} />
            </View>
          </Marker>
        )}
        {destinations.map((dest, index) => {
          if (!dest.coords) return null;

          const isLast = index === lastPlacedIndex;
          const showNumber = destinations.length > 1;
          return (
            <Marker
              key={`stop-${index}-${isLast}-${showNumber}`}
              coordinate={dest.coords}
              title={showNumber ? `Stop ${index + 1}: ${dest.name}` : dest.name}
              anchor={{ x: 0.5, y: 0.5 }}
            >
              <View style={[styles.stopMarker, isLast && styles.stopMarkerFinal]}>
                {showNumber ? (
                  <Text style={styles.stopMarkerText}>{index + 1}</Text>
                ) : (
                  <MapPin size={14} color="#fff" />
                )}
              </View>
            </Marker>
          );
        })}

        {activeTab === 'comparison' && !isNavigating ? (
          <>
            {baselineTop?.path?.length ? (
              <Polyline
                coordinates={baselineTop.path}
                strokeColor={BASELINE_COLOR}
                strokeWidth={6}
                lineJoin="round"
                lineCap="round"
              />
            ) : null}
            {antTop?.path?.length ? (
              <Polyline
                coordinates={antTop.path}
                strokeColor={ANT_COLOR}
                strokeWidth={6}
                lineJoin="round"
                lineCap="round"
              />
            ) : null}
          </>
        ) : selectedRoute?.path?.length ? (
          <Polyline
            coordinates={selectedRoute.path}
            strokeColor={selectedModelTab === 'baseline' ? BASELINE_COLOR : ANT_COLOR}
            strokeWidth={6}
            lineJoin="round"
            lineCap="round"
          />
        ) : null}
      </MapView>

      {/* Legends / locate button: sit above the map but BEHIND the scrollable panel */}
      <View style={styles.legendCard}>
        <View style={styles.legendItem}>
          <View style={[styles.dot, { backgroundColor: '#ef4444' }]} />
          <Text style={styles.legendText}>Heavy</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.dot, { backgroundColor: '#f59e0b' }]} />
          <Text style={styles.legendText}>Moderate</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.dot, { backgroundColor: '#10b981' }]} />
          <Text style={styles.legendText}>Clear</Text>
        </View>
      </View>

      <TouchableOpacity
        style={[styles.myLocationButton, { top: insets.top + MAP_BUTTON_TOP }]}
        onPress={centerOnMyLocation}
        disabled={isCentering}
        activeOpacity={0.8}
      >
        {isCentering ? (
          <ActivityIndicator size="small" color="#374151" />
        ) : (
          <Navigation size={20} color="#374151" />
        )}
      </TouchableOpacity>

      {!isNavigating && activeTab === 'comparison' && hasAnyResults && (
        <View style={styles.routeLegendCard}>
          <View style={styles.legendItem}>
            <View style={[styles.pathSwatch, { backgroundColor: ANT_COLOR }]} />
            <Text style={styles.legendText}>ANTRoute</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.pathSwatch, { backgroundColor: BASELINE_COLOR }]} />
            <Text style={styles.legendText}>Baseline</Text>
          </View>
        </View>
      )}

      <KeyboardAvoidingView
        style={styles.flexContainer}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : -40}
        pointerEvents="box-none"
      >
        <View
          style={styles.innerContainer}
          pointerEvents="box-none"
          onLayout={(e) => {
            sheetSpaceRef.current = e.nativeEvent.layout.height;
          }}
        >
          <Animated.View
            style={[
              styles.overlayWrapper,
              isExpanded && !isNavigating && [styles.expandedWrapper, { height: sheetHeightPct }],
              isExpanded && isNavigating && styles.navigatingWrapper,
            ]}
          >
            {!isExpanded ? (
              <TouchableOpacity
                style={styles.searchBar}
                onPress={handleExpand}
              >
                <Search size={20} color="#6b7280" style={styles.searchIcon} />
                <Text style={styles.placeholderText}>Plan Your Route!</Text>
              </TouchableOpacity>
            ) : (
              <>
              {/* Drag handle: outside the ScrollView so it stays put and owns the gesture */}
              {!isNavigating && (
                <View style={styles.dragHandleZone} {...sheetPanResponder.panHandlers}>
                  <View style={styles.dragHandle} />
                </View>
              )}
              <ScrollView
                showsVerticalScrollIndicator={false}
                style={isNavigating ? styles.routeBox : styles.routeBoxFill}
                contentContainerStyle={styles.routeBoxContent}
                keyboardShouldPersistTaps="handled"
                nestedScrollEnabled
                bounces={false}
              >

                {!isNavigating && (
                  <View style={styles.tabBarRow}>
                    {(['antroute', 'baseline', 'comparison'] as ModelTab[]).map((tab) => (
                      <TouchableOpacity
                        key={tab}
                        onPress={() => handleTabChange(tab)}
                        style={[styles.tabPill, activeTab === tab && styles.tabPillActive]}
                      >
                        <Text style={[styles.tabPillText, activeTab === tab && styles.tabPillTextActive]}>
                          {tab === 'antroute' ? 'ANTRoute' : tab === 'baseline' ? 'Baseline' : 'Comparison'}
                        </Text>
                      </TouchableOpacity>
                    ))}
                  </View>
                )}

                {showStopsList && (
                  <>
                    {/* Origin */}
                    <View style={styles.inputRow}>
                      <View style={[styles.pillInputContainer, styles.currentLocationInput]}>
                        {isLocating ? (
                          <ActivityIndicator size="small" color="#3b82f6" style={styles.originIcon} />
                        ) : (
                          <LocateFixed size={16} color="#3b82f6" style={styles.originIcon} />
                        )}
                        {!isNavigating ? (
                          <TextInput
                            value={origin}
                            onChangeText={handleOriginChange}
                            onFocus={() => runPlacesSearch('origin', origin)}
                            style={[styles.actualInput, styles.originInputText]}
                            placeholder="Add start location"
                            placeholderTextColor="#9ca3af"
                            editable={!isLocating}
                            autoFocus={isExpanded && origin.length === 0}
                          />
                        ) : (
                          <Text style={[styles.actualInput, styles.originInputText]} numberOfLines={1}>
                            {origin || 'Current location'}
                          </Text>
                        )}
                        {!isNavigating && (
                          origin.length > 0 ? (
                            <TouchableOpacity
                              onPress={() => {
                                setOrigin('');
                                setOriginCoords(null);
                                setSuggestions([]);
                              }}
                            >
                              <X size={18} color="#9ca3af" />
                            </TouchableOpacity>
                          ) : (
                            <TouchableOpacity onPress={fetchCurrentLocation} disabled={isLocating}>
                              <Text style={styles.useGpsLink}>Use GPS</Text>
                            </TouchableOpacity>
                          )
                        )}
                      </View>
                    </View>
                    {locationError ? <Text style={styles.errorText}>{locationError}</Text> : null}
                    {!isNavigating && renderSuggestions('origin')}

                    {/* Destinations */}
                    {destinations.map((dest, index) => {
                      const editable = isRowEditable(index);
                      return (
                        <React.Fragment key={index}>
                          <View style={styles.inputRow}>
                            <View style={[styles.pillInputContainer, styles.activePillInput]}>
                              <View style={styles.stopBadge}>
                                {destinations.length > 1 && (
                                  <Text style={styles.stopBadgeText}>{index + 1}</Text>
                                )}
                              </View>
                              {editable ? (
                                <TextInput
                                  value={dest.name}
                                  onChangeText={(text) => updateDestination(index, text)}
                                  onFocus={() => runPlacesSearch(index, dest.name)}
                                  style={styles.actualInput}
                                  placeholder="Add a destination"
                                  autoFocus={
                                    (isAddingStop && index === destinations.length - 1) ||
                                    (!isNavigating && index === destinations.length - 1 && dest.name === '' && origin.trim().length > 0)
                                  }
                                />
                              ) : (
                                <Text style={styles.actualInput} numberOfLines={1}>{dest.name}</Text>
                              )}
                              {!isNavigating && destinations.length > 1 && (
                                <TouchableOpacity
                                  onPress={() => reorderDestination(index)}
                                  style={styles.reorderHandle}
                                >
                                  <ArrowUpDown size={16} color="#9ca3af" />
                                </TouchableOpacity>
                              )}
                              {editable && (dest.name.length > 0 || destinations.length > 1) && (
                                <TouchableOpacity
                                  onPress={() => {
                                    if (dest.name.length > 0) {
                                      updateDestination(index, '');
                                    } else {
                                      removeDestination(index);
                                      if (isNavigating) setIsAddingStop(false);
                                    }
                                  }}
                                >
                                  <X size={18} color="#9ca3af" />
                                </TouchableOpacity>
                              )}
                            </View>
                          </View>
                          {editable && renderSuggestions(index)}
                        </React.Fragment>
                      );
                    })}

                    {(!isNavigating || !isAddingStop) && (
                      <TouchableOpacity
                        style={styles.addDestinationButton}
                        onPress={isNavigating ? handleAddStopWhileNavigating : addDestination}
                      >
                        <Plus size={16} color="#4475F2" />
                        <Text style={styles.addDestinationText}>Add Destination</Text>
                      </TouchableOpacity>
                    )}

                    {/* Departure time */}
                    {!isNavigating && (
                      <View style={styles.departCard}>
                        <View style={styles.departHeaderRow}>
                          <Clock size={16} color="#4475F2" />
                          <Text style={styles.departLabel}>Departure</Text>
                          {[
                            { label: 'Leave now', active: !departAt, onPress: () => departAt && changeDeparture(null) },
                            { label: 'Depart at', active: !!departAt, onPress: () => !departAt && changeDeparture(currentMinute()) },
                          ].map((chip) => (
                            <TouchableOpacity
                              key={chip.label}
                              style={[styles.departChip, chip.active && styles.departChipActive]}
                              onPress={chip.onPress}
                            >
                              <Text style={[styles.departChipText, chip.active && styles.departChipTextActive]}>
                                {chip.label}
                              </Text>
                            </TouchableOpacity>
                          ))}
                        </View>

                        {departAt && (
                          <>
                            <View style={styles.departDayRow}>
                              {['Today', 'Tomorrow'].map((label, offset) => {
                                const active = daysFromToday(departAt) === offset;
                                return (
                                  <TouchableOpacity
                                    key={label}
                                    style={[styles.departChip, active && styles.departChipActive]}
                                    onPress={() => setDepartureDay(offset)}
                                  >
                                    <Text style={[styles.departChipText, active && styles.departChipTextActive]}>
                                      {label}
                                    </Text>
                                  </TouchableOpacity>
                                );
                              })}
                            </View>
                            <View style={styles.departTimeRow}>
                              {Platform.OS === 'ios' ? (
                                <DateTimePicker
                                  style={styles.departTimePickerIOS}
                                  value={departAt}
                                  mode="time"
                                  display="compact"
                                  onValueChange={(_, time) => setDepartureTime(time)}
                                />
                              ) : (
                                <TouchableOpacity style={styles.departTimeButton} onPress={openTimePicker}>
                                  <Text style={styles.departTimeText}>{formatClock(departAt)}</Text>
                                </TouchableOpacity>
                              )}
                            </View>
                          </>
                        )}
                      </View>
                    )}

                  </>
                )}

                {routeError ? <Text style={styles.errorText}>{routeError}</Text> : null}

                {!isNavigating && !hasAnyResults && (
                  <>
                    <View style={styles.vehicleNoteRow}>
                      <CarFront size={14} color="#9ca3af" />
                      <Text style={styles.vehicleNoteText}>Routes for private 4-wheel vehicles only.</Text>
                    </View>
                    <TouchableOpacity
                      style={[styles.findButton, isLoading && styles.findButtonDisabled]}
                      onPress={handleFindRoutesPress}
                      disabled={isLoading}
                    >
                      {isLoading ? (
                        <ActivityIndicator color="#fff" />
                      ) : (
                        <Text style={styles.findButtonText}>Find Optimal Route</Text>
                      )}
                    </TouchableOpacity>
                  </>
                )}

                {isLoading && hasAnyResults && (
                  <View style={styles.loadingRow}>
                    <ActivityIndicator size="small" color="#4475F2" />
                    <Text style={styles.loadingText}>Updating route…</Text>
                  </View>
                )}

                {!isNavigating && activeTab === 'comparison' ? null : selectedRoute && (
                  <TouchableOpacity
                    style={styles.summaryBar}
                    activeOpacity={isNavigating && !stopsRevealedDuringNav ? 0.7 : 1}
                    onPress={() => {
                      if (isNavigating && !stopsRevealedDuringNav) handleExpandNavStops();
                    }}
                  >
                    {isNavigating && destinations[0]?.name ? (
                      <View style={styles.summaryDestinationRow}>
                        <MapPin size={14} color="#4475F2" />
                        <Text style={styles.summaryDestinationText} numberOfLines={1}>
                          {destinations[0].name}
                        </Text>
                      </View>
                    ) : null}
                    <View style={styles.summaryMainRow}>
                      <CarFront size={20} color="#111827" />
                      <View style={styles.summaryTextWrap}>
                        <Text style={styles.summaryDuration}>
                          <Text style={styles.summaryDurationNumber}>{selectedRoute.duration_min}</Text> min
                        </Text>
                        <Text style={styles.summarySubtext}>
                          {departAt ? `Leave ${formatDeparture(departAt)} · ` : ''}
                          Arrive By {getArrivalTime(selectedRoute.duration_min, departAt)} · {selectedRoute.distance_km} km
                        </Text>
                      </View>
                      {isNavigating ? (
                        <TouchableOpacity
                          style={[styles.summaryButton, styles.summaryButtonEnd]}
                          onPress={handleEndNavigation}
                        >
                          <Text style={styles.summaryButtonText}>End</Text>
                        </TouchableOpacity>
                      ) : (
                        <TouchableOpacity style={styles.summaryButton} onPress={handleStartNavigation}>
                          <Text style={styles.summaryButtonText}>Start</Text>
                        </TouchableOpacity>
                      )}
                    </View>
                  </TouchableOpacity>
                )}

                {!isNavigating && hasAnyResults && activeTab === 'antroute' && normalRoute && (
                  <View style={styles.normalRouteCard}>
                    <View style={styles.normalRouteTopRow}>
                      <Text style={styles.normalRouteTime}>{normalRoute.duration_min} min</Text>
                      <Text style={styles.normalRouteDistance}>{normalRoute.distance_km} km</Text>
                    </View>
                    <Text style={styles.normalRouteVia}>Via {normalRoute.via}</Text>
                    <Text style={styles.normalRouteLabel}>Baseline route · {baselineMethodLabel(normalRoute)}</Text>
                  </View>
                )}

                {!isNavigating && activeTab !== 'comparison' && currentModelOptions.length > 0 && (
                  <>
                    <Text style={styles.sectionHeaderBlue}>
                      {selectedModelTab === 'baseline'
                        ? 'Route by Baseline Model (Improved ACO)'
                        : 'Alternative Routes by ANTRoute Model'}
                    </Text>
                    {trafficNote ? (
                      <View style={styles.trafficNoteRow}>
                        <Clock size={12} color="#9ca3af" />
                        <Text style={styles.trafficNoteText}>{trafficNote}</Text>
                      </View>
                    ) : null}
                    {currentModelOptions.map((route, index) => {
                      const isSelected = index === selectedIndex;
                      return (
                        <TouchableOpacity
                          key={index}
                          style={[styles.routeListItem, isSelected && styles.routeListItemSelected]}
                          onPress={() => setSelectedIndex(index)}
                          activeOpacity={0.85}
                        >
                          <View style={styles.routeListTopRow}>
                            <View style={styles.routeListTimeRow}>
                              <Text style={styles.routeListTime}>{route.duration_min} min</Text>
                              {index === 0 && (
                                <View style={styles.recommendedBadge}>
                                  <Text style={styles.recommendedBadgeText}>Recommended</Text>
                                </View>
                              )}
                            </View>
                          </View>
                          <Text style={styles.routeListDistance}>{route.distance_km} km</Text>
                          <Text style={styles.routeListVia}>Via {route.via}</Text>

                          <View style={styles.metricsRow}>
                            <Text style={styles.metricLabel}>Congestion</Text>
                            <View style={styles.progressBar}>
                              <View
                                style={[
                                  styles.progress,
                                  {
                                    width: CONGESTION_WIDTH[route.congestion_level],
                                    backgroundColor: CONGESTION_COLORS[route.congestion_level],
                                  },
                                ]}
                              />
                            </View>
                            <Text style={styles.metricValue}>
                              {route.congestion_level.charAt(0).toUpperCase() + route.congestion_level.slice(1)}
                            </Text>
                          </View>

                          {route.event_note ? (
                            <View style={styles.infoBox}>
                              <Info size={14} color="#3b82f6" />
                              <Text style={styles.infoText}>{route.event_note}</Text>
                            </View>
                          ) : null}

                          {route.fallback_reason ? (
                            <View style={styles.infoBox}>
                              <Info size={14} color="#3b82f6" />
                              <Text style={styles.infoText}>{route.fallback_reason}</Text>
                            </View>
                          ) : null}
                        </TouchableOpacity>
                      );
                    })}
                  </>
                )}

                {!isNavigating &&
                  activeTab === 'baseline' &&
                  hasAnyResults &&
                  baselineModelOptions.length === 0 && (
                    <View style={styles.comparisonEmptyState}>
                      <Info size={18} color="#9ca3af" />
                      <Text style={styles.comparisonEmptyText}>
                        {baselineNotice ?? 'The baseline has no route for this trip.'}
                      </Text>
                    </View>
                  )}

                {!isNavigating && activeTab === 'comparison' && !hasAnyResults && (
                  <View style={styles.comparisonEmptyState}>
                    <Info size={18} color="#9ca3af" />
                    <Text style={styles.comparisonEmptyText}>
                      Add an origin and destination and find a route to see the ANTRoute vs. Baseline comparison.
                    </Text>
                  </View>
                )}

                {/* Comparison for the trip just planned: each model's distance and ETA, and this
                    trip's own evaluation against Apple Maps. Only trips timed in Apple Maps have
                    one; any other trip shows dashes rather than another trip's numbers. */}
                {!isNavigating && activeTab === 'comparison' && hasAnyResults && (
                  <View>
                    <Text style={styles.comparisonResultLabel}>Comparison Result</Text>

                    <View style={styles.comparisonCardsRow}>
                      {[
                        { name: 'ANTRoute', route: antTop, notice: null, optimality: tripOptimality?.antroute ?? null },
                        { name: 'Baseline', route: baselineTop, notice: baselineNotice, optimality: tripOptimality?.baseline ?? null },
                      ].map((item) => (
                        <View key={item.name} style={styles.comparisonCard}>
                          <View style={styles.comparisonCardHeader}>
                            <Text style={styles.comparisonCardModel}>{item.name}</Text>
                            {item.route ? (
                              <Text style={styles.comparisonCardDistance}>{item.route.distance_km} km</Text>
                            ) : null}
                          </View>
                          {item.route ? (
                            <>
                              <Text style={styles.comparisonCardLabel}>ETA</Text>
                              <Text style={styles.comparisonCardEta}>{item.route.duration_min} min</Text>
                              <Text style={[styles.comparisonCardLabel, { marginTop: 10 }]}>Route Optimality</Text>
                              <Text style={styles.comparisonCardOpt}>
                                {item.optimality != null ? `${Math.round(item.optimality)}%` : '–'}
                              </Text>
                            </>
                          ) : (
                            <Text style={styles.comparisonCardNoRoute}>{item.notice ?? 'No route for this trip.'}</Text>
                          )}
                        </View>
                      ))}
                    </View>

                    <View style={styles.comparisonDivider} />

                    <Text style={styles.evaluationTitle}>Evaluation Metrics</Text>
                    {tripEvaluationError ? (
                      <View style={styles.comparisonEmptyState}>
                        <Text style={styles.comparisonEmptyText}>{tripEvaluationError}</Text>
                      </View>
                    ) : !tripEvaluation ? (
                      <ActivityIndicator style={{ marginVertical: 30 }} color="#4475F2" />
                    ) : (
                      <>
                        {renderMetricsTable(tripMetricRows)}
                        <Text style={styles.comparisonFootnote}>
                          {tripEvaluation.status === 'evaluated'
                            ? 'Lower is better for MAE, RMSE, MSE, MAPE. Higher is better for Route Optimality and R².'
                            : tripEvaluation.message}
                        </Text>
                      </>
                    )}
                  </View>
                )}

              </ScrollView>
              </>
            )}
          </Animated.View>
        </View>
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  // Panel layer: above legends (zIndex 1) so the scrollable covers them
  flexContainer: {
    flex: 1,
    zIndex: 20,
  },
  innerContainer: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  map: {
    ...StyleSheet.absoluteFill,
  },
  currentLocationDotOuter: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: 'rgba(59, 130, 246, 0.25)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  currentLocationDotInner: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#3b82f6',
    borderWidth: 2,
    borderColor: '#fff',
  },
  legendCard: {
    position: 'absolute',
    top: StatusBar.currentHeight ? StatusBar.currentHeight + 20 : 60,
    left: 20,
    zIndex: 1,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 3,
  },
  dot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 10,
  },
  legendText: {
    fontSize: 12,
    fontWeight: '400',
    color: '#111827',
  },
  stopMarker: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#f59e0b',
    borderWidth: 2,
    borderColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  stopMarkerFinal: {
    backgroundColor: '#FEA106',
  },
  stopMarkerText: {
    fontSize: 12,
    fontWeight: '700',
    color: '#fff',
  },
  myLocationButton: {
    position: 'absolute',
    // Top-right, above the map's compass (see mapPadding); centred on the
    // compass's column, whose centre sits ~35pt from the right edge.
    right: 13,
    zIndex: 30,
    width: MAP_BUTTON_SIZE,
    height: MAP_BUTTON_SIZE,
    borderRadius: 10,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 25,
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowRadius: 5,
    shadowOffset: { width: 0, height: 2 },
  },
  routeLegendCard: {
    position: 'absolute',
    // Just under the traffic legend, with a small gap between the two groups
    top: (StatusBar.currentHeight ? StatusBar.currentHeight + 20 : 60) + 74,
    left: 20,
    zIndex: 1,
  },
  pathSwatch: {
    width: 16,
    height: 4,
    borderRadius: 2,
    marginRight: 10,
  },
  liveMapLabel: {
    position: 'absolute',
    left: 20,
    bottom: '32%',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  liveMapDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#111827',
  },
  liveMapLabelText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#374151',
  },
  overlayWrapper: {
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  // Height comes from the drag handle (sheetHeight)
  expandedWrapper: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 30,
    borderTopRightRadius: 30,
    paddingTop: 10,
    paddingHorizontal: 25,
    elevation: 20,
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowRadius: 10,
  },
  navigatingWrapper: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    paddingTop: 16,
    paddingHorizontal: 20,
    paddingBottom: 4,
    elevation: 20,
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowRadius: 10,
  },
  searchBar: {
    flexDirection: 'row',
    alignItems: 'center',
    height: 60,
    paddingHorizontal: 20,
    backgroundColor: '#fff',
    borderRadius: 30,
    elevation: 10,
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowRadius: 8,
  },
  searchIcon: {
    marginRight: 12,
  },
  placeholderText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#9ca3af',
  },
  // Navigation mode: wrapper has no fixed height, so cap the ScrollView
  routeBox: {
    width: '100%',
    maxHeight: SCREEN_HEIGHT * 0.7,
  },
  // Planning mode: fill the fixed-height panel so it scrolls fully
  routeBoxFill: {
    width: '100%',
    flex: 1,
  },
  routeBoxContent: {
    paddingBottom: 60,
  },
  // Full-width touch target, much taller than the visible bar so it's easy to grab
  dragHandleZone: {
    alignSelf: 'stretch',
    alignItems: 'center',
    marginTop: -10,
    marginHorizontal: -25,
    paddingTop: 12,
    paddingBottom: 15,
  },
  dragHandle: {
    width: 40,
    height: 5,
    backgroundColor: '#d1d5db',
    borderRadius: 3,
  },
  inputRow: {
    width: '100%',
    marginBottom: 10,
  },
  pillInputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    height: 52,
    paddingHorizontal: 15,
    backgroundColor: '#f9fafb',
    borderRadius: 25,
    borderWidth: 1,
    borderColor: '#f3f4f6',
    gap: 8,
  },
  currentLocationInput: {
    backgroundColor: '#fff',
    borderColor: '#3b82f6',
  },
  activePillInput: {
    backgroundColor: '#fffcf0',
    borderColor: '#ffe082',
  },
  actualInput: {
    flex: 1,
    fontSize: 14,
    fontWeight: '500',
    color: '#1f2937',
  },
  originInputText: {
    color: '#3b82f6',
  },
  originIcon: {
    marginRight: 10,
  },
  useGpsLink: {
    fontSize: 11,
    fontWeight: '700',
    color: '#3b82f6',
  },
  reorderHandle: {
    paddingHorizontal: 2,
  },
  stopBadge: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: '#fef3c7',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 0,
  },
  stopBadgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#FEA106',
  },
  suggestionsBox: {
    backgroundColor: '#fff',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#f3f4f6',
    marginTop: -4,
    marginBottom: 10,
    overflow: 'hidden',
    elevation: 3,
    shadowColor: '#000',
    shadowOpacity: 0.08,
    shadowRadius: 4,
  },
  suggestionRow: {
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  suggestionName: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1f2937',
  },
  suggestionAddress: {
    fontSize: 11,
    color: '#9ca3af',
    marginTop: 2,
  },
  addDestinationButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#c7d2fe',
    borderStyle: 'dashed',
    marginBottom: 12,
    gap: 6,
  },
  addDestinationText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#4475F2',
  },
  departCard: {
    backgroundColor: '#f9fafb',
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#f3f4f6',
    paddingVertical: 10,
    paddingHorizontal: 14,
    marginBottom: 12,
    gap: 10,
  },
  departHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  departLabel: {
    flex: 1,
    fontSize: 13,
    fontWeight: '600',
    color: '#1f2937',
  },
  departChip: {
    paddingVertical: 5,
    paddingHorizontal: 10,
    borderRadius: 12,
  },
  departChipActive: {
    backgroundColor: '#eff6ff',
  },
  departChipText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#9ca3af',
  },
  departChipTextActive: {
    color: '#4475F2',
  },
  departDayRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 6,
  },
  departTimeRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  departTimePickerIOS: {
    alignSelf: 'center',
  },
  departTimeButton: {
    paddingVertical: 6,
    paddingHorizontal: 20,
    borderRadius: 16,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#c7d2fe',
  },
  departTimeText: {
    fontSize: 18,
    fontWeight: '700',
    color: '#111827',
  },
  trafficNoteRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: -4,
    marginBottom: 10,
  },
  trafficNoteText: {
    flex: 1,
    fontSize: 11,
    color: '#9ca3af',
  },
  errorText: {
    color: '#ef4444',
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 10,
    textAlign: 'center',
  },
  vehicleNoteRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    marginBottom: 12,
  },
  vehicleNoteText: {
    fontSize: 12,
    color: '#9ca3af',
  },
  findButton: {
    backgroundColor: '#4475F2',
    height: 54,
    borderRadius: 27,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  findButtonDisabled: {
    opacity: 0.7,
  },
  findButtonText: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#fff',
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginBottom: 12,
  },
  loadingText: {
    fontSize: 12,
    color: '#6b7280',
  },
  summaryBar: {
    backgroundColor: '#f9fafb',
    borderRadius: 20,
    paddingVertical: 12,
    paddingHorizontal: 14,
    marginBottom: 16,
  },
  summaryDestinationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 10,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  summaryDestinationText: {
    flex: 1,
    fontSize: 13,
    fontWeight: '500',
    color: '#4475F2',
  },
  summaryMainRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  summaryTextWrap: {
    flex: 1,
    marginLeft: 10,
  },
  summaryDuration: {
    fontSize: 16,
    fontWeight: '700',
    color: '#f59e0b',
  },
  summaryDurationNumber: {
    color: '#f59e0b',
  },
  summarySubtext: {
    fontSize: 11,
    color: '#6b7280',
    marginTop: 1,
  },
  summaryButton: {
    backgroundColor: '#4475F2',
    paddingVertical: 10,
    paddingHorizontal: 22,
    borderRadius: 18,
  },
  summaryButtonEnd: {
    backgroundColor: '#ef4444',
  },
  summaryButtonText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#fff',
  },
  tabBarRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 6,
    marginBottom: 16,
  },
  tabPill: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 14,
    backgroundColor: 'transparent',
  },
  tabPillActive: {
    backgroundColor: '#eff6ff',
  },
  tabPillText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#9ca3af',
  },
  tabPillTextActive: {
    color: '#4475F2',
  },
  normalRouteCard: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#f3f4f6',
    borderRadius: 16,
    padding: 14,
    marginBottom: 16,
  },
  normalRouteTopRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 8,
  },
  normalRouteTime: {
    fontSize: 16,
    fontWeight: '700',
    color: '#111827',
  },
  normalRouteDistance: {
    fontSize: 12,
    color: '#9ca3af',
  },
  normalRouteVia: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 2,
  },
  normalRouteLabel: {
    fontSize: 11,
    color: '#9ca3af',
    marginTop: 6,
  },
  sectionHeaderBlue: {
    fontSize: 13,
    fontWeight: '700',
    color: '#4475F2',
    marginBottom: 10,
  },
  comparisonEmptyState: {
    alignItems: 'center',
    paddingVertical: 40,
    paddingHorizontal: 20,
    gap: 10,
  },
  comparisonEmptyText: {
    fontSize: 13,
    color: '#9ca3af',
    textAlign: 'center',
    lineHeight: 18,
  },
  retryButton: {
    paddingVertical: 8,
    paddingHorizontal: 20,
    borderRadius: 16,
    backgroundColor: '#eff6ff',
  },
  retryButtonText: {
    fontSize: 12,
    fontWeight: '700',
    color: '#4475F2',
  },
  comparisonResultLabel: {
    fontSize: 16,
    fontWeight: '700',
    color: 'black',
    marginBottom: 14,
  },
  comparisonCardsRow: {
    flexDirection: 'row',
    gap: 12,
  },
  comparisonCard: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#a5b4fc',
    borderRadius: 16,
    padding: 14,
    backgroundColor: '#fff',
  },
  comparisonCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  comparisonCardModel: {
    fontSize: 11,
    fontWeight: '700',
    color: '#6b7280',
  },
  comparisonCardDistance: {
    fontSize: 11,
    color: '#9ca3af',
  },
  // Which algorithm made this card's route; the baseline's is not always IACO
  comparisonCardMethod: {
    fontSize: 10,
    color: '#6b7280',
    marginBottom: 8,
  },
  comparisonCardNoRoute: {
    fontSize: 11,
    color: '#6b7280',
    lineHeight: 15,
  },
  comparisonCardLabel: {
    fontSize: 12,
    color: '#374151',
  },
  comparisonCardEta: {
    fontSize: 18,
    fontWeight: '700',
    color: '#4475F2',
  },
  comparisonCardOpt: {
    fontSize: 18,
    fontWeight: '700',
    color: '#65a30d',
  },
  comparisonDivider: {
    height: 1,
    backgroundColor: '#f3f4f6',
    marginVertical: 18,
  },
  evaluationTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: 'black',
    marginBottom: 12,
  },
  evaluationSubtitle: {
    fontSize: 11,
    color: '#6b7280',
    lineHeight: 15,
    marginTop: -6,
    marginBottom: 12,
  },
  metricsTable: {
    borderWidth: 1,
    borderColor: '#a5b4fc',
    borderRadius: 20,
    overflow: 'hidden',
    marginBottom: 10,
  },
  metricsTableHeaderRow: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    paddingVertical: 12,
    paddingHorizontal: 12,
  },
  metricsTableHeaderText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#4475F2',
  },
  metricsTableRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 14,
    paddingHorizontal: 12,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
  },
  metricsTableCell: {
    flex: 1,
    fontSize: 12,
    color: '#1f2937',
  },
  metricsTableImprovement: {
    fontWeight: '600',
    color: '#ef4444',
  },
  improvementGood: {
    color: '#65a30d',
  },
  comparisonFootnote: {
    fontSize: 10,
    color: '#9ca3af',
    marginBottom: 16,
  },
  sectionHeader: {
    fontSize: 13,
    fontWeight: '700',
    color: '#111827',
    marginBottom: 10,
  },
  routeListItem: {
    padding: 15,
    backgroundColor: '#fff',
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#f3f4f6',
    marginBottom: 10,
  },
  routeListItemSelected: {
    borderColor: '#89A6F0',
    borderWidth: 2,
  },
  routeListTopRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  routeListTimeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  routeListTime: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#111827',
  },
  routeListDistance: {
    fontSize: 11,
    color: '#9ca3af',
    marginTop: 1,
  },
  routeListVia: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1f2937',
    marginTop: 8,
    marginBottom: 8,
  },
  recommendedBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    backgroundColor: '#eff6ff',
    borderRadius: 10,
  },
  recommendedBadgeText: {
    fontSize: 9,
    fontWeight: 'bold',
    color: '#3b82f6',
  },
  metricsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  metricLabel: {
    width: 75,
    fontSize: 11,
    color: '#6b7280',
  },
  progressBar: {
    flex: 1,
    height: 6,
    marginHorizontal: 10,
    backgroundColor: '#f3f4f6',
    borderRadius: 3,
  },
  progress: {
    height: '100%',
    borderRadius: 3,
  },
  metricValue: {
    width: 60,
    textAlign: 'right',
    fontSize: 11,
    fontWeight: '600',
    color: '#4b5563',
  },
  infoBox: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 10,
    padding: 12,
    backgroundColor: '#f0f7ff',
    borderRadius: 12,
  },
  infoText: {
    flex: 1,
    marginLeft: 8,
    fontSize: 11,
    color: '#3b82f6',
  },
});