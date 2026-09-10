import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  ScrollView,
} from 'react-native';
import { User, ChevronRight, LogOut, Bell, Lock, HelpCircle } from 'lucide-react-native';
import { getStoredUser, signOut, AuthUser } from '../api/authService';

interface MenuItem {
  key: string;
  label: string;
  icon: React.ComponentType<{ size: number; color: string }>;
  onPress: (navigation: any) => void;
}

const MENU_ITEMS: MenuItem[] = [
  {
    key: 'edit-profile',
    label: 'Edit Profile',
    icon: User,
    onPress: (navigation) => navigation.navigate('EditProfileScreen'),
  },
  {
    key: 'password',
    label: 'Password',
    icon: Lock,
    onPress: (navigation) => navigation.navigate('PasswordScreen'),
  },
  {
    key: 'notifications',
    label: 'Notification Settings',
    icon: Bell,
    onPress: (navigation) => navigation.navigate('NotificationSettingsScreen'),
  },
  {
    key: 'help',
    label: 'Help & Support',
    icon: HelpCircle,
    onPress: () => {
      // TODO: link to a real support screen or external URL once available
    },
  },
];

export default function ProfileScreen({ navigation }: any) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isSigningOut, setIsSigningOut] = useState(false);

  useEffect(() => {
    let isMounted = true;
    getStoredUser().then((storedUser) => {
      if (isMounted) setUser(storedUser);
    });
    return () => {
      isMounted = false;
    };
  }, []);

  const handleSignOut = async () => {
    setIsSigningOut(true);
    try {
      await signOut();
      navigation.reset({ index: 0, routes: [{ name: 'Landing' }] });
    } finally {
      setIsSigningOut(false);
    }
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="transparent" translucent />

      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <Text style={styles.headerTitle}>Profile</Text>

        <View style={styles.profileCard}>
          <View style={styles.avatarCircle}>
            <User size={28} color="#4475F2" />
          </View>
          <View style={styles.profileText}>
            <Text style={styles.profileName}>{user?.name || 'Guest'}</Text>
            <Text style={styles.profileEmail}>{user?.email || 'Not signed in'}</Text>
          </View>
        </View>

        <View style={styles.menuGroup}>
          {MENU_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <TouchableOpacity
                key={item.key}
                style={styles.menuRow}
                onPress={() => item.onPress(navigation)}
              >
                <View style={styles.menuLeft}>
                  <Icon size={18} color="#4b5563" />
                  <Text style={styles.menuLabel}>{item.label}</Text>
                </View>
                <ChevronRight size={18} color="#d1d5db" />
              </TouchableOpacity>
            );
          })}
        </View>

        <TouchableOpacity
          style={styles.signOutButton}
          onPress={handleSignOut}
          disabled={isSigningOut}
        >
          <LogOut size={18} color="#ef4444" />
          <Text style={styles.signOutText}>
            {isSigningOut ? 'Signing out…' : 'Sign out'}
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
    paddingTop: StatusBar.currentHeight ? StatusBar.currentHeight + 10 : 50,
  },
  scrollContent: {
    paddingHorizontal: 20,
    paddingBottom: 30,
  },
  headerTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#111827',
    marginBottom: 16,
  },
  profileCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
    borderRadius: 16,
    padding: 16,
    marginBottom: 20,
    gap: 14,
  },
  avatarCircle: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: '#eff6ff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  profileText: {
    flex: 1,
  },
  profileName: {
    fontSize: 16,
    fontWeight: '700',
    color: '#111827',
  },
  profileEmail: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 2,
  },
  menuGroup: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#f3f4f6',
    overflow: 'hidden',
    marginBottom: 24,
  },
  menuRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 16,
    paddingHorizontal: 16,
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
    fontWeight: '500',
    color: '#1f2937',
  },
  signOutButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 14,
    borderRadius: 30,
    borderWidth: 1,
    borderColor: '#fecaca',
    backgroundColor: '#fff5f5',
  },
  signOutText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#ef4444',
  },
});
