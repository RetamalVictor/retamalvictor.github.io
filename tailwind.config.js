/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx,html}",
  ],
  safelist: [
    // Classes used in dynamic template literals (TernaryLMDemo)
    'text-accent-cyan',
    'text-accent-purple',
    'bg-dark-bg',
    'bg-dark-surface',
    'border-dark-border',
    'hover:bg-dark-surface',
    // Classes used in DroneRacingDemo
    'bg-dark-surface/80',
    'bg-dark-surface/90',
    'bg-accent-cyan/20',
    'border-accent-cyan',
    'border-accent-cyan/30',
    'hover:border-accent-cyan',
    'hover:text-accent-cyan',
    'text-gray-400',
    'font-mono',
    'text-sm',
    'right-14',
    'bottom-3',
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          bg: '#0a0a0f',
          surface: '#12121a',
          border: '#1e1e2e',
          hover: '#1a1a24',
        },
        accent: {
          cyan: '#00d4ff',
          purple: '#a855f7',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in-up': 'fadeInUp 0.6s ease-out',
        'fade-in-down': 'fadeInDown 0.3s ease-out',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        fadeInUp: {
          '0%': {
            opacity: '0',
            transform: 'translateY(30px)'
          },
          '100%': {
            opacity: '1',
            transform: 'translateY(0)'
          }
        },
        fadeInDown: {
          '0%': {
            opacity: '0',
            transform: 'translateY(-10px)'
          },
          '100%': {
            opacity: '1',
            transform: 'translateY(0)'
          }
        },
        glow: {
          '0%': {
            boxShadow: '0 0 5px rgba(0, 212, 255, 0.2), 0 0 20px rgba(0, 212, 255, 0.1)'
          },
          '100%': {
            boxShadow: '0 0 10px rgba(0, 212, 255, 0.4), 0 0 40px rgba(0, 212, 255, 0.2)'
          }
        }
      }
    },
  },
  plugins: [],
}