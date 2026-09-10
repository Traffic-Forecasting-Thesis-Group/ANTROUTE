import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  StatusBar,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Svg, { Path } from 'react-native-svg';
import { Newspaper, Globe, AlertTriangle } from 'lucide-react-native';

type SourceFilter = 'All' | 'Affects my route' | 'Nearby' | 'X' | 'News API' | 'GDELT';
type FeedStatus = 'affects_route' | 'nearby' | 'monitoring' | 'cleared';

interface FeedEvent {
  id: string;
  source: 'X' | 'News API' | 'GDELT';
  status: FeedStatus;
  timeAgo: string;
  headline: string;
  location: string;
  attribution: string;
}

function XIcon({ size = 13, color = '#1f2937' }: { size?: number; color?: string }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
      <Path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
    </Svg>
  );
}

const STATUS_STYLES: Record<FeedStatus, { label: string; bg: string; color: string; stripe: string }> = {
  affects_route: { label: 'Affects my route', bg: '#F7C9C9', color: '#C83433', stripe: '#C83433' },
  nearby: { label: 'Nearby', bg: '#F9E7B3', color: '#f59e0b', stripe: '#f59e0b' },
  monitoring: { label: 'Monitoring', bg: '#f3f4f6', color: '#6b7280', stripe: '#9ca3af' },
  cleared: { label: 'Cleared', bg: '#f0fdf4', color: '#10b981', stripe: '#10b981' },
};

// TODO: replace with a real fetch from your backend (e.g. GET /events) once available
const MOCK_EVENTS: FeedEvent[] = [
  {
    id: '1',
    source: 'X',
    status: 'affects_route',
    timeAgo: '4m ago',
    headline: 'Road users report heavy backup along EDSA near Ortigas due to a stalled bus.',
    location: 'EDSA, Ortigas',
    attribution: 'MMDA',
  },
  {
    id: '2',
    source: 'News API',
    status: 'nearby',
    timeAgo: '22m ago',
    headline: 'Concert at MOA Arena tonight expected to draw large crowds, city advises alternate routes.',
    location: 'Pasay',
    attribution: 'Manila Bulletin',
  },
  {
    id: '3',
    source: 'GDELT',
    status: 'monitoring',
    timeAgo: '1h ago',
    headline: 'Planned rally near Mendiola flagged as a public-gathering event with possible road closures.',
    location: 'Mendiola',
    attribution: 'GDELT Event Database',
  },
];

const SOURCE_ICON = {
  'X': XIcon,
  'News API': Newspaper,
  'GDELT': Globe,
};

export default function FeedScreen({ navigation }: any) {
  const [activeFilter, setActiveFilter] = useState<SourceFilter>('All');

  const filters: SourceFilter[] = ['All', 'Affects my route', 'Nearby', 'X', 'News API', 'GDELT'];

  const visibleEvents = MOCK_EVENTS.filter((event) => {
    if (activeFilter === 'All') return true;
    if (activeFilter === 'Affects my route') return event.status === 'affects_route';
    if (activeFilter === 'Nearby') return event.status === 'nearby';
    return event.source === activeFilter;
  });

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#F0A93A" />

      <SafeAreaView edges={['top']} style={styles.header}>
        <Text style={styles.headerTitle}>Event Reports</Text>
      </SafeAreaView>

      <ScrollView
        horizontal
        style={styles.filterScroll}
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.filterRow}
      >
        {filters.map((filter) => {
          const isActive = filter === activeFilter;
          return (
            <TouchableOpacity
              key={filter}
              style={[styles.filterChip, isActive && styles.filterChipActive]}
              onPress={() => setActiveFilter(filter)}
            >
              <Text style={[styles.filterChipText, isActive && styles.filterChipTextActive]}>
                {filter}
              </Text>
            </TouchableOpacity>
          );
        })}
      </ScrollView>

      <ScrollView
        style={styles.listContainer}
        contentContainerStyle={styles.list}
        showsVerticalScrollIndicator={false}
      >
        {visibleEvents.length === 0 ? (
          <View style={styles.emptyState}>
            <AlertTriangle size={22} color="#9ca3af" />
            <Text style={styles.emptyText}>No events match this filter right now.</Text>
          </View>
        ) : (
          visibleEvents.map((event) => {
            const statusStyle = STATUS_STYLES[event.status];
            const SourceIcon = SOURCE_ICON[event.source];
            return (
              <View
                key={event.id}
                style={[styles.card, { borderLeftColor: statusStyle.stripe }]}
              >
                <View style={styles.cardHeader}>
                  <SourceIcon size={13} color="#1f2937" />
                  <Text style={styles.sourceLabel}>{event.source} · {event.timeAgo}</Text>
                  <View style={[styles.statusPill, { backgroundColor: statusStyle.bg }]}>
                    <Text style={[styles.statusPillText, { color: statusStyle.color }]}>
                      {statusStyle.label}
                    </Text>
                  </View>
                </View>
                <Text style={styles.headline}>{event.headline}</Text>
                <Text style={styles.meta}>{event.location} · {event.attribution}</Text>
              </View>
            );
          })
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  header: {
    backgroundColor: '#F0A93A',
    paddingBottom: 16,
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 17,
    fontWeight: '700',
    color: '#fff',
    marginTop: 10,
  },
  filterScroll: {
    flexGrow: 0,
    height: 62,
  },
  filterRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    gap: 8,
  },
  filterChip: {
    paddingVertical: 7,
    paddingHorizontal: 14,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    backgroundColor: '#fff',
  },
  filterChipActive: {
    backgroundColor: '#4475F2',
    borderColor: '#4475F2',
  },
  filterChipText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6b7280',
  },
  filterChipTextActive: {
    color: '#fff',
  },
  listContainer: {
    flex: 1,
  },
  list: {
    paddingHorizontal: 20,
    paddingBottom: 30,
    gap: 10,
  },
  card: {
    backgroundColor: '#f9fafb',
    borderRadius: 14,
    padding: 12,
    borderLeftWidth: 3,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 6,
  },
  sourceLabel: {
    fontSize: 11,
    color: '#9ca3af',
    flex: 1,
  },
  statusPill: {
    paddingVertical: 2,
    paddingHorizontal: 8,
    borderRadius: 10,
  },
  statusPillText: {
    fontSize: 10,
    fontWeight: '700',
  },
  headline: {
    fontSize: 13,
    lineHeight: 18,
    color: '#1f2937',
    marginBottom: 4,
  },
  meta: {
    fontSize: 10,
    color: '#9ca3af',
  },
  emptyState: {
    alignItems: 'center',
    marginTop: 60,
    gap: 10,
  },
  emptyText: {
    fontSize: 13,
    color: '#9ca3af',
  },
});