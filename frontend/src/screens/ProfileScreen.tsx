import React, { useCallback, useState } from 'react';

import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  ScrollView,
  Alert,
} from 'react-native';

import { useFocusEffect } from '@react-navigation/native';

import {
  User,
  Lock,
  Info,
  CarFront,
  Camera,
  ChevronRight,
} from 'lucide-react-native';

import {
  getCurrentUser,
  getStoredUser,
  signOut,
  AuthUser,
} from '../api/authService';

type Chevron = 'right' | 'none';

interface MenuRow {
  key: string;
  label: string;
  subtitle: string;
  icon: React.ComponentType<{ size: number; color: string }>;
  iconColor: string;
  chevron: Chevron;
  onPress?: (navigation: any) => void;
}

interface MenuSection {
  title: string;
  data: MenuRow[];
}

function getInitials(name?: string | null): string {
  if (!name) return 'G';

  const parts = name.trim().split(/\s+/);
  const first = parts[0]?.[0] ?? '';
  const last = parts.length > 1 ? parts[parts.length - 1][0] : '';

  return (first + last).toUpperCase() || 'G';
}

export default function ProfileScreen({ navigation }: any) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isSigningOut, setIsSigningOut] = useState(false);

  useFocusEffect(
    useCallback(() => {
      let isActive = true;

      const loadUser = async () => {
        const cached = await getStoredUser();

        if (isActive && cached) {
          setUser(cached);
        }

        const fresh = await getCurrentUser();

        if (isActive && fresh) {
          setUser(fresh);
        }
      };

      loadUser();

      return () => {
        isActive = false;
      };
    }, [])
  );

  const handleSignOut = () => {
    Alert.alert(
      'Sign Out',
      'Are you sure you want to sign out?',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Sign Out',
          style: 'destructive',
          onPress: async () => {
            setIsSigningOut(true);

            try {
              await signOut();

              navigation.reset({
                index: 0,
                routes: [{ name: 'Landing' }],
              });
            } finally {
              setIsSigningOut(false);
            }
          },
        },
      ]
    );
  };

  const SECTIONS: MenuSection[] = [
    {
      title: 'Account',
      data: [
        {
          key: 'personal-info',
          label: 'Personal Information',
          subtitle: 'Name, email, phone',
          icon: User,
          iconColor: '#4475F2',
          chevron: 'right',
          onPress: (nav) => nav.navigate('EditProfileScreen'),
        },
        {
          key: 'password',
          label: 'Password & Security',
          subtitle: 'Manage password',
          icon: Lock,
          iconColor: '#4475F2',
          chevron: 'right',
          onPress: (nav) => nav.navigate('PasswordScreen'),
        },
      ],
    },
    {
      title: 'About',
      data: [
        {
          key: 'antroute',
          label: 'ANTRoute',
          subtitle: 'Thesis Prototype',
          icon: Info,
          iconColor: '#4475F2',
          chevron: 'right',
          onPress: (nav) => nav.navigate('AboutScreen'),
        },
        {
          key: 'vehicle-scope',
          label: 'Vehicle Scope',
          subtitle: '4-Wheel only',
          icon: CarFront,
          iconColor: '#4475F2',
          chevron: 'none',
        },
      ],
    },
  ];

  return (
    <View style={styles.container}>
      <StatusBar
        barStyle="dark-content"
        backgroundColor="transparent"
        translucent
      />

      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Profile */}
        <View style={styles.avatarSection}>
          <View style={styles.avatarWrap}>
            <View style={styles.avatarCircle}>
              <Text style={styles.avatarInitials}>
                {getInitials(user?.name)}
              </Text>
            </View>

            <TouchableOpacity
              style={styles.avatarEditBadge}
              onPress={() => navigation.navigate('EditProfileScreen')}
            >
              <Camera size={12} color="#fff" />
            </TouchableOpacity>
          </View>

          <Text style={styles.profileName}>
            {user?.name || 'Guest'}
          </Text>

          <Text style={styles.profileEmail}>
            {user?.email || 'Not signed in'}
          </Text>
        </View>

        {/* Menu Sections */}
        {SECTIONS.map((section) => (
          <View key={section.title} style={styles.sectionWrap}>
            <View style={styles.sectionHeaderRow}>
              <Text style={styles.sectionHeaderText}>
                {section.title.toUpperCase()}
              </Text>
            </View>

            <View style={styles.menuGroup}>
              {section.data.map((item, index) => {
                const Icon = item.icon;
                const isLast = index === section.data.length - 1;
                const isPressable = !!item.onPress;
                const RowWrapper = isPressable
                  ? TouchableOpacity
                  : View;

                return (
                  <RowWrapper
                    key={item.key}
                    style={[
                      styles.menuRow,
                      !isLast && styles.menuRowDivider,
                    ]}
                    {...(isPressable
                      ? {
                          onPress: () =>
                            item.onPress!(navigation),
                          activeOpacity: 0.7,
                        }
                      : {})}
                  >
                    <View style={styles.menuLeft}>
                      <Icon
                        size={18}
                        color={item.iconColor}
                      />

                      <View>
                        <Text style={styles.menuLabel}>
                          {item.label}
                        </Text>

                        <Text style={styles.menuSubtitle}>
                          {item.subtitle}
                        </Text>
                      </View>
                    </View>

                    {item.chevron === 'right' && (
                      <ChevronRight
                        size={18}
                        color="#d1d5db"
                      />
                    )}
                  </RowWrapper>
                );
              })}
            </View>
          </View>
        ))}

        {/* Sign Out */}
        <TouchableOpacity
          style={styles.signOutButton}
          onPress={handleSignOut}
          disabled={isSigningOut}
        >
          <Text style={styles.signOutText}>
            {isSigningOut ? 'Signing out…' : 'Sign Out'}
          </Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    paddingTop: StatusBar.currentHeight
      ? StatusBar.currentHeight + 10
      : 50,
  },

  scrollContent: {
    paddingHorizontal: 20,
    paddingBottom: 30,
  },

  avatarSection: {
    alignItems: 'center',
    marginTop: 10,
    marginBottom: 24,
  },

  avatarWrap: {
    position: 'relative',
    marginBottom: 12,
  },

  avatarCircle: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: '#4475F2',
    justifyContent: 'center',
    alignItems: 'center',
  },

  avatarInitials: {
    fontSize: 24,
    fontWeight: '700',
    color: '#fff',
  },

  avatarEditBadge: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#f59e0b',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#fff',
  },

  profileName: {
    fontSize: 16,
    fontWeight: '700',
    color: '#111827',
  },

  profileEmail: {
    fontSize: 12,
    color: '#9ca3af',
    marginTop: 2,
  },

  sectionWrap: {
    marginBottom: 18,
  },

  sectionHeaderRow: {
    paddingVertical: 8,
    paddingHorizontal: 4,
  },

  sectionHeaderText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#9ca3af',
    letterSpacing: 0.5,
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

  menuSubtitle: {
    fontSize: 11,
    color: '#9ca3af',
    marginTop: 1,
  },

  signOutButton: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#fecaca',
    backgroundColor: '#fff',
    marginTop: 8,
  },

  signOutText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#ef4444',
  },
});