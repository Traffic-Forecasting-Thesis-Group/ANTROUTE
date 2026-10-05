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
} from 'react-native';

import MapView, { PROVIDER_GOOGLE, Marker, Polyline } from 'react-native-maps';
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
} from 'lucide-react-native';

import * as Location from 'expo-location';

import {
  planRoute,
  getComparisonMetrics,
  ComparisonMetrics,
  RouteOption,
  CongestionLevel,
  Coordinates,
  DestinationInput,
} from '../api/routeService';
import { searchPlaces, reverseGeocode, PlaceSuggestion } from '../api/placesService';

const SCREEN_HEIGHT = Dimensions.get('window').height;

const CONGESTION_COLORS: Record<CongestionLevel, string> = {
  clear: '#10b981',
  moderate: '#f59e0b',
  heavy: '#ef4444',
};

const CONGESTION_WIDTH: Record<CongestionLevel, `${number}%`> = {
  clear: '30%',
  moderate: '60%',
  heavy: '85%',
};

type ActiveField = 'origin' | number | null;

interface DestinationEntry {
  name: string;
  coords: Coordinates | null;
}

type ModelTab = 'antroute' | 'baseline' | 'comparison';

function getArrivalTime(durationMin: number): string {
  const now = new Date();
  now.setMinutes(now.getMinutes() + durationMin);
  let hours = now.getHours();
  const minutes = now.getMinutes().toString().padStart(2, '0');
  const ampm = hours >= 12 ? 'PM' : 'AM';
  hours = hours % 12 || 12;
  return `${hours}:${minutes} ${ampm}`;
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

  const [comparisonMetrics, setComparisonMetrics] = useState<ComparisonMetrics | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState('');

  const loadComparisonMetrics = async () => {
    setMetricsLoading(true);
    setMetricsError('');
    try {
      setComparisonMetrics(await getComparisonMetrics());
    } catch (e: any) {
      setMetricsError(e?.message || 'Could not load comparison metrics.');
    } finally {
      setMetricsLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'comparison' && !comparisonMetrics && !metricsLoading) {
      loadComparisonMetrics();
    }
  }, [activeTab]);

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

  const handleExpand = () => setIsExpanded(true);
  const handleCollapse = () => setIsExpanded(false);

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
      const [normalResults, antResults, baselineResults] = await Promise.all([
        planRoute(originToUse.trim(), cleanedDestinations, false, originCoordsToUse),
        planRoute(originToUse.trim(), cleanedDestinations, true, originCoordsToUse),
        planRoute(originToUse.trim(), cleanedDestinations, true, originCoordsToUse, 'baseline'),
      ]);

      setNormalRoute(normalResults[0] ?? null);
      setAntRouteOptions(antResults);
      setBaselineModelOptions(baselineResults);
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
        provider={Platform.OS === 'android' ? PROVIDER_GOOGLE : undefined}
        style={styles.map}
        showsUserLocation={hasLocationPermission}
        showsMyLocationButton={false}
        // iOS places its compass inside the map's layout margins; a taller top margin
        // drops the compass just below the locate button. (On Android, padding would
        // also shift the map centre, and Google's compass sits top-left anyway.)
        mapPadding={
          Platform.OS === 'ios'
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
        <View style={styles.innerContainer} pointerEvents="box-none">
          <View
            style={[
              styles.overlayWrapper,
              isExpanded && !isNavigating && styles.expandedWrapper,
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
              <ScrollView
                showsVerticalScrollIndicator={false}
                style={isNavigating ? styles.routeBox : styles.routeBoxFill}
                contentContainerStyle={styles.routeBoxContent}
                keyboardShouldPersistTaps="handled"
                nestedScrollEnabled
                bounces={false}
              >
                {!isNavigating && <View style={styles.dragHandle} />}

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

                  </>
                )}

                {routeError ? <Text style={styles.errorText}>{routeError}</Text> : null}

                {!isNavigating && !hasAnyResults && (
                  <>
                    <View style={styles.vehicleNoteRow}>
                      <CarFront size={14} color="#9ca3af" />
                      <Text style={styles.vehicleNoteText}>Routes optimized for 4-wheel vehicles.</Text>
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
                          Arrive By {getArrivalTime(selectedRoute.duration_min)} · {selectedRoute.distance_km} km
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

                {!isNavigating && hasAnyResults && activeTab !== 'comparison' && normalRoute && (
                  <View style={styles.normalRouteCard}>
                    <View style={styles.normalRouteTopRow}>
                      <Text style={styles.normalRouteTime}>{normalRoute.duration_min} min</Text>
                      <Text style={styles.normalRouteDistance}>{normalRoute.distance_km} km</Text>
                    </View>
                    <Text style={styles.normalRouteVia}>Via {normalRoute.via}</Text>
                    <Text style={styles.normalRouteLabel}>Normal route (no traffic optimization)</Text>
                  </View>
                )}

                {!isNavigating && activeTab !== 'comparison' && currentModelOptions.length > 0 && (
                  <>
                    <Text style={styles.sectionHeaderBlue}>
                      Alternative Route by {selectedModelTab === 'baseline' ? 'Baseline' : 'ANTRoute'} Model
                    </Text>
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
                        </TouchableOpacity>
                      );
                    })}
                  </>
                )}

                {!isNavigating && activeTab === 'comparison' && !hasAnyResults && (
                  <View style={styles.comparisonEmptyState}>
                    <Info size={18} color="#9ca3af" />
                    <Text style={styles.comparisonEmptyText}>
                      Add an origin and destination and find a route to see the ANTRoute vs. Baseline comparison.
                    </Text>
                  </View>
                )}

                {!isNavigating && activeTab === 'comparison' && hasAnyResults && metricsLoading && (
                  <ActivityIndicator style={{ marginVertical: 30 }} color="#4475F2" />
                )}

                {!isNavigating && activeTab === 'comparison' && hasAnyResults && !metricsLoading && metricsError ? (
                  <View style={styles.comparisonEmptyState}>
                    <Text style={styles.comparisonEmptyText}>{metricsError}</Text>
                    <TouchableOpacity style={styles.retryButton} onPress={loadComparisonMetrics}>
                      <Text style={styles.retryButtonText}>Retry</Text>
                    </TouchableOpacity>
                  </View>
                ) : null}

                {!isNavigating && activeTab === 'comparison' && hasAnyResults && !metricsLoading && comparisonMetrics && (
                  <View>
                    <Text style={styles.comparisonResultLabel}>Comparison Result</Text>

                    <View style={styles.comparisonCardsRow}>
                      {[
                        { name: 'ANTRoute', route: antTop, opt: comparisonMetrics.routeOptimalityPct.antroute },
                        { name: 'Baseline', route: baselineTop, opt: comparisonMetrics.routeOptimalityPct.baseline },
                      ].map((item) => (
                        <View key={item.name} style={styles.comparisonCard}>
                          <View style={styles.comparisonCardHeader}>
                            <Text style={styles.comparisonCardModel}>{item.name}</Text>
                            <Text style={styles.comparisonCardDistance}>{item.route?.distance_km} km</Text>
                          </View>
                          <Text style={styles.comparisonCardLabel}>ETA</Text>
                          <Text style={styles.comparisonCardEta}>{item.route?.duration_min} min</Text>
                          <Text style={[styles.comparisonCardLabel, { marginTop: 10 }]}>Route Optimality</Text>
                          <Text style={styles.comparisonCardOpt}>{item.opt}%</Text>
                        </View>
                      ))}
                    </View>

                    <View style={styles.comparisonDivider} />

                    <Text style={styles.evaluationTitle}>Evaluation Metrics</Text>
                    <View style={styles.metricsTable}>
                      <View style={styles.metricsTableHeaderRow}>
                        <Text style={[styles.metricsTableCell, styles.metricsTableHeaderText, { flex: 1.5 }]}>
                          Metric
                        </Text>
                        <Text style={[styles.metricsTableCell, styles.metricsTableHeaderText]}>ANTRoute</Text>
                        <Text style={[styles.metricsTableCell, styles.metricsTableHeaderText]}>Baseline</Text>
                        <Text style={[styles.metricsTableCell, styles.metricsTableHeaderText]}>Improvement</Text>
                      </View>
                      {comparisonMetrics.metrics.map((row) => {
                        const isGood = row.higherIsBetter ? row.improvementPct > 0 : row.improvementPct < 0;
                        return (
                          <View key={row.metric} style={styles.metricsTableRow}>
                            <Text style={[styles.metricsTableCell, { flex: 1.5 }]}>{row.metric}</Text>
                            <Text style={styles.metricsTableCell}>{row.antroute}</Text>
                            <Text style={styles.metricsTableCell}>{row.baseline}</Text>
                            <Text
                              style={[
                                styles.metricsTableCell,
                                styles.metricsTableImprovement,
                                isGood && styles.improvementGood,
                              ]}
                            >
                              {row.improvementPct > 0 ? '+' : ''}
                              {row.improvementPct}%
                            </Text>
                          </View>
                        );
                      })}
                    </View>
                    <Text style={styles.comparisonFootnote}>
                      Lower is better for MAE, RMSE, MSE, MAPE. Higher is better for R².
                    </Text>
                  </View>
                )}

              </ScrollView>
            )}
          </View>
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
  expandedWrapper: {
    height: '70%',
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
  dragHandle: {
    alignSelf: 'center',
    width: 40,
    height: 4,
    marginBottom: 15,
    backgroundColor: '#e5e7eb',
    borderRadius: 2,
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
    borderColor: '#4475F2',
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