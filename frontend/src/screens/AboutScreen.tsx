import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  ScrollView,
  Image,
} from 'react-native';

import {
  ChevronLeft,
  Route,
  Brain,
  CarFront,
  Users,
  GraduationCap,
  FlaskConical,
} from 'lucide-react-native';

const APP_VERSION = '1.0.0';

const LOGO_SOURCE = require('../../assets/logo-no-bg.png');

interface InfoRow {
  key: string;
  label: string;
  value: string;
  icon: React.ComponentType<{ size: number; color: string }>;
}

const INFO_ROWS: InfoRow[] = [
  {
    key: 'vehicle-scope',
    label: 'Vehicle Scope',
    value: '4-Wheel only',
    icon: CarFront,
  },
  {
    key: 'model',
    label: 'Routing Model',
    value: 'ANTRoute',
    icon: Brain,
  },
  {
    key: 'version',
    label: 'Prototype Version',
    value: APP_VERSION,
    icon: Route,
  },
];

export default function AboutAntRouteScreen({ navigation }: any) {
  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="transparent" translucent />

      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
        >
          <ChevronLeft size={22} color="#111827" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>About</Text>
        <View style={styles.headerSpacer} />
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.logoSection}>
          <Image source={LOGO_SOURCE} style={styles.logoImage} resizeMode="contain" />
          <Text style={styles.appName}>ANTRoute</Text>
          <Text style={styles.appTagline}>Thesis Prototype</Text>
        </View>

        <View style={styles.simulationNotice}>
          <FlaskConical size={16} color="#b45309" />
          <Text style={styles.simulationNoticeText}>
            Simulation prototype for thesis demonstration. Not connected to live CCTV,
            social media, or traffic sensor feeds.
          </Text>
        </View>

        <View style={styles.sectionWrap}>
          <Text style={styles.sectionHeaderText}>WHAT IS ANTROUTE</Text>
          <View style={styles.card}>
            <Text style={styles.bodyText}>
              A CNN-LSTM and RADR-STGNN framework with Ant Colony Optimization that
              fuses CCTV, social media, and traffic data into a congestion-risk score,
              used to route around real, event-driven traffic in Metro Manila.
            </Text>
          </View>
        </View>

        <View style={styles.sectionWrap}>
          <Text style={styles.sectionHeaderText}>HOW ROUTES ARE COMPARED</Text>
          <View style={styles.card}>
            <Text style={styles.bodyText}>
              ANTRoute is benchmarked against a baseline Ant Colony model using only
              path length, travel time, and traffic flow. The Comparison tab shows Route
              Optimality and ETA accuracy (MAE, RMSE, MSE, R²) for both.
            </Text>
          </View>
        </View>

        <View style={styles.sectionWrap}>
          <Text style={styles.sectionHeaderText}>DETAILS</Text>
          <View style={styles.menuGroup}>
            {INFO_ROWS.map((row, index) => {
              const Icon = row.icon;
              const isLast = index === INFO_ROWS.length - 1;
              return (
                <View
                  key={row.key}
                  style={[styles.menuRow, !isLast && styles.menuRowDivider]}
                >
                  <View style={styles.menuLeft}>
                    <Icon size={18} color="#4475F2" />
                    <Text style={styles.menuLabel}>{row.label}</Text>
                  </View>
                  <Text style={styles.menuValue}>{row.value}</Text>
                </View>
              );
            })}
          </View>
        </View>

        <View style={styles.sectionWrap}>
          <Text style={styles.sectionHeaderText}>DEVELOPED BY</Text>
          <View style={styles.card}>
            <View style={styles.developerRow}>
              <Users size={18} color="#4475F2" />
              <View style={{ flex: 1 }}>
                <Text style={styles.authorName}>Guanzon, Kyla Mae C.</Text>
                <Text style={styles.authorName}>Pagallaman, Joshua P.</Text>
                <Text style={styles.authorName}>Palazo, Angel Rose M.</Text>
                <Text style={styles.authorName}>Pepito, Merille Janine O.</Text>
              </View>
            </View>
            <View style={[styles.developerRow, { marginTop: 12 }]}>
              <GraduationCap size={18} color="#4475F2" />
              <Text style={styles.bodyText}>
                Bachelor of Science in Computer Science{'\n'}
                College of Computer and Information Sciences{'\n'}
                Polytechnic University of the Philippines, Sta. Mesa, Manila
              </Text>
            </View>
          </View>
        </View>

        <Text style={styles.footerText}>ANTRoute v{APP_VERSION} · Simulation prototype for thesis defense</Text>
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
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: StatusBar.currentHeight ? StatusBar.currentHeight + 10 : 50,
    paddingHorizontal: 16,
    paddingBottom: 12,
  },
  backButton: {
    width: 32,
    height: 32,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#111827',
  },
  headerSpacer: {
    width: 32,
  },
  scrollContent: {
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  logoSection: {
    alignItems: 'center',
    marginTop: 10,
    marginBottom: 24,
  },
  simulationNotice: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    backgroundColor: '#fffbeb',
    borderWidth: 1,
    borderColor: '#fde68a',
    borderRadius: 14,
    padding: 14,
    marginBottom: 18,
  },
  simulationNoticeText: {
    flex: 1,
    fontSize: 12,
    lineHeight: 18,
    color: '#92400e',
  },
  logoImage: {
    width: 88,
    height: 88,
    marginBottom: 12,
  },
  appName: {
    fontSize: 20,
    fontWeight: '700',
    color: '#111827',
  },
  appTagline: {
    fontSize: 12,
    color: '#9ca3af',
    marginTop: 2,
  },
  sectionWrap: {
    marginBottom: 18,
  },
  sectionHeaderText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#9ca3af',
    letterSpacing: 0.5,
    paddingHorizontal: 4,
    marginBottom: 8,
  },
  card: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#f3f4f6',
    backgroundColor: '#fff',
    padding: 16,
  },
  bodyText: {
    flex: 1,
    fontSize: 13,
    lineHeight: 20,
    color: '#4b5563',
  },
  authorName: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1f2937',
    lineHeight: 20,
  },
  developerRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
  },
  menuGroup: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#f3f4f6',
    overflow: 'hidden',
  },
  menuRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 14,
    paddingHorizontal: 16,
    backgroundColor: '#fff',
  },
  menuRowDivider: {
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  menuLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  menuLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
  },
  menuValue: {
    fontSize: 13,
    color: '#9ca3af',
  },
  footerText: {
    fontSize: 11,
    color: '#d1d5db',
    textAlign: 'center',
    marginTop: 8,
  },
}); 