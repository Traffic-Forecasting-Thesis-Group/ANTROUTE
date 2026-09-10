import apiClient from './client';

export interface PlaceSuggestion {
  id: string;
  lat: number;
  lng: number;
  name: string;
  address: string;
  formattedAddress: string;
}

interface SearchResponse {
  results: {
    id: string;
    lat: number;
    lng: number;
    name: string;
    address: string;
    formatted_address: string;
  }[];
}

interface ReverseResponse {
  label: string | null;
}

export async function searchPlaces(query: string): Promise<PlaceSuggestion[]> {
  const trimmed = query.trim();
  if (trimmed.length < 2) return [];

  try {
    const { data } = await apiClient.get<SearchResponse>('/places/search', {
      params: { query: trimmed },
    });
    return data.results.map((item) => ({
      id: item.id,
      lat: item.lat,
      lng: item.lng,
      name: item.name,
      address: item.address,
      formattedAddress: item.formatted_address,
    }));
  } catch (error: any) {
    console.warn('searchPlaces failed:', error?.message || error);
    return [];
  }
}

export async function reverseGeocode(lat: number, lng: number): Promise<string | null> {
  try {
    const { data } = await apiClient.get<ReverseResponse>('/places/reverse', {
      params: { lat, lng },
    });
    return data.label;
  } catch (error: any) {
    console.warn('reverseGeocode failed:', error?.message || error);
    return null;
  }
}