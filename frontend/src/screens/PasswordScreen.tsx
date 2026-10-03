import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  TouchableOpacity,
  ScrollView,
  TextInput,
  ActivityIndicator,
  Alert,
} from 'react-native';

import {
  ChevronLeft,
  Eye,
  EyeOff,
} from 'lucide-react-native';

import { changePassword, deleteAccount, signOut } from '../api/authService';

export default function PasswordSecurityScreen({ navigation }: any) {
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  const [showCurrent, setShowCurrent] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [formError, setFormError] = useState('');

  const handleUpdate = async () => {
    setFormError('');

    if (!currentPassword || !newPassword || !confirmPassword) {
      setFormError('Please fill in all password fields.');
      return;
    }

    if (newPassword !== confirmPassword) {
      setFormError('The new password and confirm password do not match.');
      return;
    }

    if (newPassword.length < 8) {
      setFormError('New password must be at least 8 characters.');
      return;
    }

    setIsSubmitting(true);
    try {
      await changePassword(currentPassword, newPassword);
      Alert.alert('Success', 'Password updated successfully!', [
        { text: 'OK', onPress: () => navigation.goBack() },
      ]);
    } catch (error: any) {
      setFormError(error?.message || 'Could not update your password. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const performDeleteAccount = async () => {
    setIsDeleting(true);
    try {
      await deleteAccount();
      await signOut();
      navigation.reset({ index: 0, routes: [{ name: 'Landing' }] });
    } catch (error: any) {
      Alert.alert('Error', error?.message || 'Could not delete your account. Please try again.');
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDeleteAccount = () => {
    Alert.alert(
      'Delete account?',
      'This will permanently delete your account and sign you out. This cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Delete', style: 'destructive', onPress: performDeleteAccount },
      ]
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      {/* HEADER */}
      <View style={styles.header}>
        <TouchableOpacity
          onPress={() => navigation.goBack()}
          style={styles.backButton}
        >
          <ChevronLeft size={24} color="#0084FF" />
        </TouchableOpacity>

        <Text style={styles.headerTitle}>Password & security</Text>
        <View style={{ width: 32 }} />
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        {/* PASSWORD SECTION */}
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>CHANGE PASSWORD</Text>

          <View style={styles.fieldGroup}>
            <PasswordField
              label="CURRENT PASSWORD"
              placeholder="Enter current password"
              value={currentPassword}
              onChangeText={setCurrentPassword}
              secureTextEntry={!showCurrent}
              onToggle={() => setShowCurrent(!showCurrent)}
            />

            <PasswordField
              label="NEW PASSWORD"
              placeholder="Enter new password"
              value={newPassword}
              onChangeText={setNewPassword}
              secureTextEntry={!showNew}
              onToggle={() => setShowNew(!showNew)}
            />

            <PasswordField
              label="CONFIRM NEW PASSWORD"
              placeholder="Re-enter password"
              value={confirmPassword}
              onChangeText={setConfirmPassword}
              secureTextEntry={!showConfirm}
              onToggle={() => setShowConfirm(!showConfirm)}
              isError={newPassword !== confirmPassword && confirmPassword.length > 0}
              isLast
            />
          </View>

          {formError ? <Text style={styles.formError}>{formError}</Text> : null}
        </View>

        {/* BUTTONS */}
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={[styles.updateButton, isSubmitting && styles.updateButtonDisabled]}
            onPress={handleUpdate}
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <ActivityIndicator color="#000" />
            ) : (
              <Text style={styles.updateButtonText}>Update password</Text>
            )}
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.deleteButton, isDeleting && styles.updateButtonDisabled]}
            onPress={handleDeleteAccount}
            disabled={isDeleting}
          >
            {isDeleting ? (
              <ActivityIndicator color="#EF4444" />
            ) : (
              <Text style={styles.deleteButtonText}>Delete account</Text>
            )}
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const PasswordField = ({
  label,
  value,
  onChangeText,
  secureTextEntry,
  onToggle,
  placeholder,
  isError,
  isLast,
}: any) => (
  <View style={[styles.fieldWrapper, isLast && styles.fieldWrapperLast]}>
    <Text style={[styles.fieldLabel, isError && { color: '#EF4444' }]}>
      {isError ? "PASSWORDS DO NOT MATCH" : label}
    </Text>

    <View style={styles.inputRow}>
      <TextInput
        style={styles.textInput}
        value={value}
        onChangeText={onChangeText}
        secureTextEntry={secureTextEntry}
        placeholder={placeholder}
        placeholderTextColor="#9CA3AF"
        autoCapitalize="none"
      />

      <TouchableOpacity onPress={onToggle}>
        {secureTextEntry ? (
          <EyeOff size={20} color="#D1D5DB" />
        ) : (
          <Eye size={20} color="#D1D5DB" />
        )}
      </TouchableOpacity>
    </View>
  </View>
);

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#F3F4F6' },

  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },

  backButton: {
    padding: 8,
    backgroundColor: '#F3F4F6',
    borderRadius: 12,
  },

  headerTitle: {
    fontSize: 17,
    fontWeight: 'bold',
    color: '#111827',
  },

  scrollContent: {
    paddingBottom: 40,
  },

  section: {
    marginTop: 20,
  },

  sectionLabel: {
    fontSize: 11,
    fontWeight: 'bold',
    color: '#9CA3AF',
    letterSpacing: 0.5,
    paddingHorizontal: 20,
    marginBottom: 8,
  },

  fieldGroup: {
    backgroundColor: '#fff',
  },

  fieldWrapper: {
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },

  fieldWrapperLast: {
    borderBottomWidth: 0,
  },

  fieldLabel: {
    fontSize: 10,
    color: '#9CA3AF',
    marginBottom: 4,
    fontWeight: '600',
  },

  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },

  textInput: {
    flex: 1,
    fontSize: 15,
    color: '#111827',
    paddingVertical: 4,
  },

  formError: {
    color: '#EF4444',
    fontSize: 12,
    fontWeight: '600',
    paddingHorizontal: 20,
    marginTop: 10,
  },

  buttonContainer: {
    paddingHorizontal: 20,
    marginTop: 24,
  },

  updateButton: {
    backgroundColor: '#FFD700',
    padding: 18,
    borderRadius: 14,
    alignItems: 'center',
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#EAB308',
  },

  updateButtonDisabled: {
    opacity: 0.7,
  },

  updateButtonText: {
    color: '#000',
    fontWeight: 'bold',
    fontSize: 15,
  },

  deleteButton: {
    backgroundColor: '#FFF1F2',
    padding: 18,
    borderRadius: 14,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#FFE4E6',
  },

  deleteButtonText: {
    color: '#EF4444',
    fontWeight: 'bold',
    fontSize: 15,
  },
});