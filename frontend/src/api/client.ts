import axios from 'axios';
import Constants from 'expo-constants';

const resolveApiBaseUrl = (): string => {
  const explicitUrl = process.env.EXPO_PUBLIC_API_URL;
  if (explicitUrl) {
    return explicitUrl;
  }

  const constants = Constants as unknown as {
    expoConfig?: { hostUri?: string };
    manifest2?: { extra?: { expoClient?: { hostUri?: string } } };
    manifest?: { debuggerHost?: string };
  };
  
  const hostUri =
    constants.expoConfig?.hostUri ||
    constants.manifest2?.extra?.expoClient?.hostUri ||
    constants.manifest?.debuggerHost;

  const host = hostUri?.split(':')[0];
  if (host) {
    return `http://${host}:8000`;
  }

  return 'http://127.0.0.1:8000';
};

const apiClient = axios.create({
  baseURL: resolveApiBaseUrl(),
  headers: {
    'Content-Type': 'application/json',
  },
});

export default apiClient;