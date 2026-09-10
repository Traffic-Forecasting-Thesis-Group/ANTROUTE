import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  Text,
  KeyboardAvoidingView,
  Platform,
  TouchableWithoutFeedback,
  Keyboard,
  StatusBar,
  ScrollView,
  ActivityIndicator,
} from 'react-native';

import MapView, { PROVIDER_GOOGLE, Marker } from 'react-native-maps';
import { useIsFocused } from '@react-navigation/native';

import {
  X,
  ArrowUpDown,
  Info,
  CarFront,
  Search,
  LocateFixed,
  Plus,
  MapPin,
} from 'lucide-react-native';

import * as Location from 'expo-location';

import { planRoute, RouteOption, CongestionLevel, Coordinates, DestinationInput } from '../api/routeService';
import { searchPlaces, reverseGeocode, PlaceSuggestion } from '../api/placesService';

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

function getArrivalTime(durationMin: number): string {
  const now = new Date();
  now.setMinutes(now.getMinutes() + durationMin);
  let hours = now.getHours();
  const minutes = now.getMinutes().toString().padStart(2, '0');
  const ampm = hours >= 12 ? 'PM' : 'AM';
  hours = hours % 12 || 12;
  return `${hours}:${minutes} ${ampm}`;
}

export default function HomeScreen({ navigation }: any) {
  const mapRef = useRef<MapView | null>(null);

  const [isExpanded, setIsExpanded] = useState(false);
  const [origin, setOrigin] = useState('');
  const [originCoords, setOriginCoords] = useState<Coordinates | null>(null);
  const [isLocating, setIsLocating] = useState(false);
  const [locationError, setLocationError] = useState('');
  const [destinations, setDestinations] = useState<DestinationEntry[]>([{ name: '', coords: null }]);

  const [isLoading, setIsLoading] = useState(false);
  const [routeError, setRouteError] = useState('');
  const [routeOptions, setRouteOptions] = useState<RouteOption[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);

  const [isNavigating, setIsNavigating] = useState(false);
  const [isAddingStop, setIsAddingStop] = useState(false);
  const [stopsRevealedDuringNav, setStopsRevealedDuringNav] = useState(false);

  const [activeField, setActiveField] = useState<ActiveField>(null);
  const [suggestions, setSuggestions] = useState<PlaceSuggestion[]>([]);
  const [isSearchingPlaces, setIsSearchingPlaces] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isFocused = useIsFocused();

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

      const position = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.Balanced,
      });
      const coords: Coordinates = {
        latitude: position.coords.latitude,
        longitude: position.coords.longitude,
      };
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

    if (text.trim().length < 2) {
      setSuggestions([]);
      return;
    }

    debounceRef.current = setTimeout(async () => {
      setIsSearchingPlaces(true);
      const results = await searchPlaces(text);
      setSuggestions(results);
      setIsSearchingPlaces(false);
    }, 300);
  };

  const handleOriginChange = (text: string) => {
    setOrigin(text);
    setOriginCoords(null);
    setRouteOptions([]);
    runPlacesSearch('origin', text);
  };

  const updateDestination = (index: number, value: string) => {
    setDestinations((prev) => prev.map((d, i) => (i === index ? { name: value, coords: null } : d)));
    
    if (!isNavigating) {
      setRouteOptions([]);
    }
    runPlacesSearch(index, value);
  };

  const isRowEditable = (index: number) => {
    if (!isNavigating) return true;
    return isAddingStop && index === destinations.length - 1;
  };

  const selectSuggestion = (suggestion: PlaceSuggestion) => {
    const coords: Coordinates = { latitude: suggestion.lat, longitude: suggestion.lng };

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
      const results = await planRoute(originToUse.trim(), cleanedDestinations, true, originCoordsToUse);
      setRouteOptions(results);
      setSelectedIndex(0);
    } catch (error: any) {
      setRouteError(error?.message || 'Something went wrong. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFindRoutesPress = () => {
    handleFindRoutes(destinations, origin, originCoords);
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
    setRouteOptions([]);
    setSelectedIndex(0);
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
            <Text style={styles.suggestionText} numberOfLines={1}>{item.formattedAddress}</Text>
          </TouchableOpacity>
        ))}
      </View>
    );
  };

  const selectedRoute = routeOptions[selectedIndex];
  const showStopsList = !isNavigating || stopsRevealedDuringNav;

  return (
    <View style={styles.container}>
      <StatusBar
        barStyle="dark-content"
        backgroundColor="transparent"
        translucent={true}
      />

      <MapView
        ref={mapRef}
        provider={Platform.OS === 'android' ? PROVIDER_GOOGLE : undefined}
        style={styles.map}
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
        {destinations.map((dest, index) =>
          dest.coords ? (
            <Marker key={index} coordinate={dest.coords} title={dest.name} pinColor="#f59e0b" />
          ) : null
        )}
      </MapView>

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

      <KeyboardAvoidingView
        style={styles.flexContainer}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : -40}
        pointerEvents="box-none"
      >
        <TouchableWithoutFeedback
          onPress={() => {
            Keyboard.dismiss();
            setSuggestions([]);
            if (isNavigating && stopsRevealedDuringNav) handleCollapseNavStops();
          }}
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
                <ScrollView showsVerticalScrollIndicator={false} style={styles.routeBox} keyboardShouldPersistTaps="handled">
                  {!isNavigating && <View style={styles.dragHandle} />}

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

                  {!isNavigating && routeOptions.length === 0 && (
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
                  )}

                  {isLoading && routeOptions.length > 0 && (
                    <View style={styles.loadingRow}>
                      <ActivityIndicator size="small" color="#4475F2" />
                      <Text style={styles.loadingText}>Updating route…</Text>
                    </View>
                  )}

                  {selectedRoute && (
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

                  {!isNavigating && routeOptions.length > 0 && (
                    <>
                      <Text style={styles.sectionHeader}>
                        {routeOptions.length} Best Route{routeOptions.length !== 1 ? 's' : ''}
                      </Text>
                      {routeOptions.map((route, index) => {
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
                </ScrollView>
              )}
            </View>
          </View>
        </TouchableWithoutFeedback>
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  flexContainer: {
    flex: 1,
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
    right: 20,
    backgroundColor: '#fff',
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 16,
    elevation: 4,
    shadowColor: '#000',
    shadowOpacity: 0.15,
    shadowRadius: 4,
    zIndex: 10,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 3,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  legendText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#374151',
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
  routeBox: {
    width: '100%',
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
  suggestionText: {
    fontSize: 13,
    color: '#1f2937',
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

