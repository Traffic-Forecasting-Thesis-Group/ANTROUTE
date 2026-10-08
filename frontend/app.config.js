// Extends app.json. Google Maps needs an API key compiled into dev/production builds
// (Expo Go on Android ships its own; Expo Go on iOS has no Google Maps at all). Set
// GOOGLE_MAPS_API_KEY when building and the key is added for both platforms;
// extra.iosGoogleMaps tells HomeScreen the iOS build can render Google Maps.
module.exports = ({ config }) => {
  const key = process.env.GOOGLE_MAPS_API_KEY;
  if (!key) return config;
  return {
    ...config,
    plugins: [
      ...(config.plugins ?? []),
      ['react-native-maps', { iosGoogleMapsApiKey: key, androidGoogleMapsApiKey: key }],
    ],
    extra: { ...(config.extra ?? {}), iosGoogleMaps: true },
  };
};
