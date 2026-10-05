import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  TouchableOpacity,
  ScrollView,
  TextInput,
  Alert,
  Image,
  Modal,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from 'react-native';
import { ChevronLeft, Camera, X } from 'lucide-react-native';
import * as ImagePicker from 'expo-image-picker';
import { getCurrentUser, getStoredUser, updateProfile } from '../api/authService';

type EditableKey = 'firstName' | 'lastName';

function splitName(fullName: string): { firstName: string; lastName: string } {
  const parts = fullName.trim().split(/\s+/);
  if (parts.length === 1) return { firstName: parts[0] || '', lastName: '' };
  return { firstName: parts.slice(0, -1).join(' '), lastName: parts[parts.length - 1] };
}

export default function EditProfileScreen({ navigation }: any) {
  const [image, setImage] = useState<string | null>(null);
  const [isViewVisible, setIsViewVisible] = useState(false);
  const [isLoadingUser, setIsLoadingUser] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [email, setEmail] = useState('');

  const [editingField, setEditingField] = useState<EditableKey | null>(null);

  useEffect(() => {
    let isMounted = true;

    const loadUser = async () => {
      const cached = await getStoredUser();
      if (isMounted && cached) {
        const { firstName: f, lastName: l } = splitName(cached.name);
        setFirstName(f);
        setLastName(l);
        setEmail(cached.email);
      }

      const fresh = await getCurrentUser();
      if (isMounted && fresh) {
        const { firstName: f, lastName: l } = splitName(fresh.name);
        setFirstName(f);
        setLastName(l);
        setEmail(fresh.email);
      }

      if (isMounted) setIsLoadingUser(false);
    };

    loadUser();
    return () => {
      isMounted = false;
    };
  }, []);

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert("Permission denied. Access to the gallery is required.");
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });
    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  const handleSave = async () => {
    setEditingField(null);
    const fullName = `${firstName.trim()} ${lastName.trim()}`.trim();

    if (!fullName) {
      Alert.alert('Error', 'First and last name cannot both be empty.');
      return;
    }

    setIsSaving(true);
    try {
      await updateProfile(fullName, email);
      Alert.alert('Success', 'Profile updated successfully!');
      navigation.goBack();
    } catch (error: any) {
      Alert.alert('Error', error?.message || 'Could not update your profile. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  const initials = ((firstName[0] || '') + (lastName[0] || '')).toUpperCase();

  return (
    <SafeAreaView style={styles.safeArea}>
      <Modal visible={isViewVisible} transparent={true} animationType="fade">
        <View style={styles.modalBackground}>
          <TouchableOpacity 
            style={styles.closeModalButton} 
            onPress={() => setIsViewVisible(false)}
          >
            <X size={30} color="#fff" />
          </TouchableOpacity>
          {image ? (
            <Image source={{ uri: image }} style={styles.fullScreenImage} />
          ) : (
            <View style={[styles.avatarCircle, { width: 250, height: 250, borderRadius: 125 }]}>
              <Text style={[styles.avatarText, { fontSize: 80 }]}>{initials}</Text>
            </View>
          )}
        </View>
      </Modal>

      <KeyboardAvoidingView 
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={{ flex: 1 }}
      >
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
            <ChevronLeft size={24} color="#0084FF" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Personal information</Text>
          <TouchableOpacity onPress={handleSave} disabled={isSaving}>
            <Text style={styles.headerSaveText}>Save</Text>
          </TouchableOpacity>
        </View>

        {isLoadingUser ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator color="#0084FF" />
          </View>
        ) : (
          <ScrollView 
            showsVerticalScrollIndicator={false} 
            contentContainerStyle={styles.scrollContent}
            keyboardShouldPersistTaps="handled"
          >
            <View style={styles.photoSection}>
              <View style={styles.avatarWrapper}>
                <TouchableOpacity 
                  activeOpacity={0.8}
                  onPress={() => setIsViewVisible(true)} 
                  style={styles.avatarCircle}
                >
                  {image ? (
                    <Image source={{ uri: image }} style={styles.profileImage} />
                  ) : (
                    <Text style={styles.avatarText}>{initials}</Text>
                  )}
                </TouchableOpacity>
                
                <TouchableOpacity 
                  activeOpacity={0.9}
                  onPress={pickImage} 
                  style={styles.cameraBadge}
                >
                  <Camera size={14} color="#000" />
                </TouchableOpacity>
              </View>
              
              <TouchableOpacity onPress={pickImage}>
                <Text style={styles.changePhotoText}>Change photo</Text>
              </TouchableOpacity>
            </View>

            <Text style={styles.sectionBar}>BASIC DETAILS</Text>
            <View style={styles.fieldGroup}>
              <DisplayField
                label="FIRST NAME"
                value={firstName}
                onChangeText={setFirstName}
                isEditing={editingField === 'firstName'}
                onEdit={() => setEditingField('firstName')}
                onDone={() => setEditingField(null)}
              />
              <DisplayField
                label="LAST NAME"
                value={lastName}
                onChangeText={setLastName}
                isEditing={editingField === 'lastName'}
                onEdit={() => setEditingField('lastName')}
                onDone={() => setEditingField(null)}
                isLast
              />
            </View>

            <Text style={styles.sectionBar}>CONTACT</Text>
            <View style={styles.fieldGroup}>
              <View style={[styles.fieldRow, styles.fieldRowLast]}>
                <Text style={styles.fieldLabel}>EMAIL ADDRESS</Text>
                <Text style={styles.fieldValue}>{email}</Text>
              </View>
            </View>

            <TouchableOpacity style={styles.mainSaveButton} onPress={handleSave} disabled={isSaving}>
              {isSaving ? (
                <ActivityIndicator color="#000" />
              ) : (
                <Text style={styles.mainSaveText}>Save changes</Text>
              )}
            </TouchableOpacity>
          </ScrollView>
        )}
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const DisplayField = ({
  label,
  value,
  onChangeText,
  isEditing,
  onEdit,
  onDone,
  isLast,
}: any) => (
  <View style={[styles.fieldRow, isLast && styles.fieldRowLast]}>
    <View style={{ flex: 1 }}>
      <Text style={styles.fieldLabel}>{label}</Text>
      {isEditing ? (
        <TextInput
          style={styles.fieldInput}
          value={value}
          onChangeText={onChangeText}
          autoFocus
          onBlur={onDone}
          onSubmitEditing={onDone}
        />
      ) : (
        <Text style={styles.fieldValue}>{value}</Text>
      )}
    </View>
    <TouchableOpacity onPress={isEditing ? onDone : onEdit}>
      <Text style={styles.editLink}>{isEditing ? 'Done' : 'Edit'}</Text>
    </TouchableOpacity>
  </View>
);

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#fff' },
  header: { 
    flexDirection: 'row', 
    alignItems: 'center', 
    justifyContent: 'space-between', 
    paddingHorizontal: 16, 
    paddingVertical: 12, 
    borderBottomWidth: 1, 
    borderBottomColor: '#F3F4F6' 
  },
  backButton: { 
    padding: 8, 
    backgroundColor: '#F3F4F6', 
    borderRadius: 12 
  },
  headerTitle: { 
    fontSize: 16, 
    fontWeight: 'bold', 
    color: '#000' 
  },
  headerSaveText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#0084FF',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scrollContent: { paddingBottom: 40 },
  photoSection: { 
    alignItems: 'center', 
    marginVertical: 25 
  },
  avatarWrapper: {
    position: 'relative',
    marginBottom: 10,
  },
  avatarCircle: { 
    width: 80, 
    height: 80, 
    borderRadius: 40, 
    backgroundColor: '#0084FF', 
    justifyContent: 'center', 
    alignItems: 'center', 
    overflow: 'hidden',
  },
  profileImage: { width: '100%', height: '100%' },
  avatarText: { color: '#fff', fontSize: 28, fontWeight: 'bold' },
  cameraBadge: { 
    position: 'absolute', 
    bottom: 0, 
    right: 0, 
    backgroundColor: '#FFD700', 
    padding: 6, 
    borderRadius: 15, 
    borderWidth: 2, 
    borderColor: '#fff',
    zIndex: 10,
  },
  modalBackground: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.9)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  fullScreenImage: {
    width: '90%',
    height: '60%',
    resizeMode: 'contain',
  },
  closeModalButton: {
    position: 'absolute',
    top: 50,
    right: 20,
    zIndex: 20,
  },
  changePhotoText: { 
    color: '#0084FF', 
    fontSize: 13, 
    fontWeight: '500' 
  },
  sectionBar: {
    fontSize: 11,
    fontWeight: 'bold',
    color: '#9CA3AF',
    letterSpacing: 0.5,
    backgroundColor: '#F9FAFB',
    paddingVertical: 8,
    paddingHorizontal: 20,
  },
  fieldGroup: {
    backgroundColor: '#fff',
  },
  fieldRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  fieldRowLast: {
    borderBottomWidth: 0,
  },
  fieldLabel: { 
    fontSize: 10, 
    color: '#9CA3AF', 
    marginBottom: 2 
  },
  fieldValue: {
    fontSize: 14,
    color: '#111827',
  },
  fieldInput: {
    fontSize: 14,
    color: '#111827',
    paddingVertical: 2,
    borderBottomWidth: 1,
    borderBottomColor: '#0084FF',
  },
  editLink: {
    fontSize: 13,
    fontWeight: '600',
    color: '#0084FF',
    marginLeft: 12,
  },
  mainSaveButton: { 
    backgroundColor: '#FFD700', 
    marginHorizontal: 20, 
    marginTop: 20, 
    padding: 16, 
    borderRadius: 12, 
    alignItems: 'center', 
    borderWidth: 0
  },
  mainSaveText: { 
    color: '#000', 
    fontWeight: 'bold', 
    fontSize: 15 
  },
});