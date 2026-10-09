import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  StatusBar,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useIsFocused } from '@react-navigation/native';

import { getComparisonMetrics, ComparisonMetrics } from '../api/routeService';

/**
 * The thesis routing evaluation over every test trip timed in Apple Maps (Section 3.9,
 * Appendix 3): ANTRoute against the baseline Improved ACO, overall and per scenario.
 * A single trip's own evaluation is on the Home screen's Comparison tab.
 */
export default function EvaluationScreen() {
  const isFocused = useIsFocused();
  const [data, setData] = useState<ComparisonMetrics | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [scenario, setScenario] = useState('overall');

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      setData(await getComparisonMetrics());
    } catch (e: any) {
      setError(e?.message || 'Could not load the evaluation.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const shown = data?.scenarios.find((s) => s.scenario === scenario) ?? data?.scenarios[0];
  const total = data?.scenarios[0]?.nTrials ?? 0;

  return (
    <View style={styles.container}>
      {isFocused && <StatusBar barStyle="light-content" backgroundColor="#4475F2" />}

      <SafeAreaView edges={['top']} style={styles.header}>
        <Text style={styles.headerTitle}>Evaluation</Text>
      </SafeAreaView>

      {data && data.scenarios.length > 0 && (
        <ScrollView
          horizontal
          style={styles.filterScroll}
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.filterRow}
        >
          {data.scenarios.map((s) => {
            const isActive = s.scenario === shown?.scenario;
            return (
              <TouchableOpacity
                key={s.scenario}
                style={[styles.filterChip, isActive && styles.filterChipActive]}
                onPress={() => setScenario(s.scenario)}
              >
                <Text style={[styles.filterChipText, isActive && styles.filterChipTextActive]}>
                  {`${s.label} (${s.nTrials})`}
                </Text>
              </TouchableOpacity>
            );
          })}
        </ScrollView>
      )}

      <ScrollView
        style={styles.body}
        contentContainerStyle={styles.bodyContent}
        refreshControl={<RefreshControl refreshing={loading && !!data} onRefresh={load} />}
      >
        {loading && !data ? (
          <ActivityIndicator style={{ marginVertical: 40 }} color="#4475F2" />
        ) : error ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyText}>{error}</Text>
            <TouchableOpacity style={styles.retryButton} onPress={load}>
              <Text style={styles.retryButtonText}>Retry</Text>
            </TouchableOpacity>
          </View>
        ) : shown ? (
          <>
            <Text style={styles.title}>
              {shown.scenario === 'overall' ? `All ${total} Test Trips` : `${shown.label} Trips`}
            </Text>
            <Text style={styles.subtitle}>
              {`ANTRoute against the baseline (${data?.baselineName}) on ${shown.nTrials} test trips, ` +
                `both routed on the same stops and departure times and timed in Apple Maps typical traffic.` +
                (data?.etaCalibrated
                  ? ` ETAs calibrated: each model's predicted ETA is scaled by one factor fitted on the other test trips.`
                  : '')}
            </Text>

            <View style={styles.metricsTable}>
              <View style={styles.metricsTableHeaderRow}>
                <Text style={[styles.metricsTableCell, styles.metricsTableHeaderText, { flex: 1.5 }]}>Metric</Text>
                <Text style={[styles.metricsTableCell, styles.metricsTableHeaderText]}>ANTRoute</Text>
                <Text style={[styles.metricsTableCell, styles.metricsTableHeaderText]}>Baseline</Text>
                <Text style={[styles.metricsTableCell, styles.metricsTableHeaderText]}>Improvement</Text>
              </View>
              {shown.metrics.map((row) => {
                const isGood = row.improvementPct != null && row.improvementPct > 0;
                return (
                  <View key={row.metric} style={styles.metricsTableRow}>
                    <Text style={[styles.metricsTableCell, { flex: 1.5 }]}>
                      {row.metric}
                      {row.significant ? ' *' : ''}
                    </Text>
                    <Text style={styles.metricsTableCell}>{row.antroute}</Text>
                    <Text style={styles.metricsTableCell}>{row.baseline}</Text>
                    <Text
                      style={[styles.metricsTableCell, styles.metricsTableImprovement, isGood && styles.improvementGood]}
                    >
                      {row.improvementPct == null ? '–' : `${row.improvementPct > 0 ? '+' : ''}${row.improvementPct}%`}
                    </Text>
                  </View>
                );
              })}
            </View>

            <Text style={styles.footnote}>
              {`Lower is better for MAE, RMSE, MSE, MAPE. Higher is better for Route Optimality and R². ` +
                `Route Optimality is the mean over trips; the ETA errors pool every leg. ` +
                `Improvement is the relative difference Δ% (Equations 7–8): positive means ANTRoute is better. ` +
                `* significant, Wilcoxon signed-rank p < 0.05.`}
            </Text>
          </>
        ) : null}
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
    backgroundColor: '#4475F2',
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
  body: {
    flex: 1,
  },
  bodyContent: {
    paddingHorizontal: 20,
    paddingBottom: 30,
  },
  title: {
    fontSize: 16,
    fontWeight: '700',
    color: 'black',
    marginBottom: 12,
  },
  subtitle: {
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
  footnote: {
    fontSize: 10,
    color: '#9ca3af',
    lineHeight: 14,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 40,
    paddingHorizontal: 20,
    gap: 10,
  },
  emptyText: {
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
});
