import AsyncStorage from '@react-native-async-storage/async-storage';
import apiClient from './client';

const TOKEN_KEY = 'authToken';
const USER_KEY = 'authUser';

export interface AuthUser {
  id: string;
  name: string;
  email: string;
}

export interface SignInResponse {
  token: string;
  user: AuthUser;
}

export interface SignUpResponse {
  token: string;
  user: AuthUser;
}

function applyAuthHeader(token: string | null) {
  if (token) {
    apiClient.defaults.headers.common.Authorization = `Bearer ${token}`;
  } else {
    delete apiClient.defaults.headers.common.Authorization;
  }
}

async function persistSession(token: string, user: AuthUser | null) {
  await AsyncStorage.setItem(TOKEN_KEY, token);
  await AsyncStorage.setItem(USER_KEY, JSON.stringify(user ?? null));
  applyAuthHeader(token);
}

function toReadableError(error: any, fallbackMessage: string): Error {
  if (error.response) {
    return new Error(error.response.data?.detail || error.response.data?.message || fallbackMessage);
  }
  if (error.request) {
    return new Error('Could not reach the server. Check your connection and try again.');
  }
  return error instanceof Error ? error : new Error(fallbackMessage);
}

export async function restoreSession(): Promise<AuthUser | null> {
  const [token, storedUser] = await Promise.all([
    AsyncStorage.getItem(TOKEN_KEY),
    AsyncStorage.getItem(USER_KEY),
  ]);

  if (!token) return null;

  applyAuthHeader(token);
  return storedUser ? JSON.parse(storedUser) : null;
}

export async function getStoredUser(): Promise<AuthUser | null> {
  const storedUser = await AsyncStorage.getItem(USER_KEY);
  return storedUser ? JSON.parse(storedUser) : null;
}

export async function signIn(email: string, password: string): Promise<SignInResponse> {
  try {
    const { data } = await apiClient.post<SignInResponse>('/auth/signin', {
      email,
      password,
    });

    if (!data?.token) {
      throw new Error('Unexpected response from server.');
    }

    await persistSession(data.token, data.user);
    return data;
  } catch (error: any) {
    throw toReadableError(error, 'Incorrect email or password.');
  }
}

export async function signUp(name: string, email: string, password: string): Promise<SignUpResponse> {
  try {
    const { data } = await apiClient.post<SignUpResponse>('/auth/signup', {
      name,
      email,
      password,
    });

    if (!data?.token) {
      throw new Error('Unexpected response from server.');
    }

    await persistSession(data.token, data.user);
    return data;
  } catch (error: any) {
    throw toReadableError(error, 'Could not create your account. Please try again.');
  }
}

export async function signOut(): Promise<void> {
  await AsyncStorage.multiRemove([TOKEN_KEY, USER_KEY]);
  applyAuthHeader(null);
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  try {
    await apiClient.post('/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
  } catch (error: any) {
    throw toReadableError(error, 'Could not update your password. Please try again.');
  }
}

export async function deleteAccount(): Promise<void> {
  try {
    await apiClient.delete('/auth/account');
  } catch (error: any) {
    throw toReadableError(error, 'Could not delete your account. Please try again.');
  }
}

export async function getCurrentUser(): Promise<AuthUser | null> {
  try {
    const { data } = await apiClient.get<AuthUser>('/auth/me');
    await AsyncStorage.setItem(USER_KEY, JSON.stringify(data));
    return data;
  } catch (error: any) {
    console.warn('getCurrentUser failed:', error?.message || error);
    return null;
  }
}

export async function updateProfile(name: string, email: string): Promise<AuthUser> {
  try {
    const { data } = await apiClient.patch<AuthUser>('/auth/me', { name, email });
    await AsyncStorage.setItem(USER_KEY, JSON.stringify(data));
    return data;
  } catch (error: any) {
    throw toReadableError(error, 'Could not update your profile. Please try again.');
  }
}