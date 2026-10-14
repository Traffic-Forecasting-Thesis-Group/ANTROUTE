/**
 * Design tokens for the desktop web app. The colours below were inline constants in the
 * old React Native HomeScreen; they live here now so the Tailwind classes and the
 * imperative Google Maps code (which needs real colour strings) read the same values --
 * see src/lib/theme.ts, which re-exports them for the polyline and marker layers.
 */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Brand blue: headers, primary buttons, the recommended route.
        brand: {
          DEFAULT: '#4475F2',
          dark: '#3B6BE6',
          darker: '#2E55BD',
          light: '#EEF3FE',
        },
        // The fixed left navigation rail.
        sidebar: {
          DEFAULT: '#1E3C8F',
          hover: '#2A4CA8',
        },
        // Active nav pill, and the baseline route on the map.
        accent: {
          DEFAULT: '#F5C242',
          dark: '#E0A91F',
        },
        // Congestion bands (old CONGESTION_COLORS).
        congestion: {
          clear: '#10b981',
          moderate: '#f59e0b',
          heavy: '#ef4444',
          unknown: '#d1d5db',
        },
        // Route polyline colours (old ANT_COLOR / BASELINE_COLOR).
        route: {
          ant: '#3B6BE6',
          baseline: '#F7C441',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      boxShadow: {
        card: '0 1px 3px rgba(16, 24, 40, 0.08), 0 1px 2px rgba(16, 24, 40, 0.04)',
        panel: '0 4px 12px rgba(16, 24, 40, 0.10)',
      },
    },
  },
  plugins: [],
};
