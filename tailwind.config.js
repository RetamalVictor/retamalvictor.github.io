/** @type {import('tailwindcss').Config} */

// Every colour goes through a CSS variable so the whole site can swap between
// the light "blueprint" palette and the original dark one without touching markup.
// Variables hold raw RGB channels ("11 87 208") so Tailwind's /opacity modifiers
// keep working.
const token = (name) => `rgb(var(${name}) / <alpha-value>)`;

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
          bg: token('--c-bg'),
          surface: token('--c-surface'),
          border: token('--c-border'),
          hover: token('--c-hover'),
        },
        accent: {
          cyan: token('--c-accent'),
          purple: token('--c-accent-2'),
        },
        // "white" is the strongest text colour in the markup - in the light
        // theme that means deep navy ink, not literal white.
        white: token('--c-ink'),
        gray: {
          50: token('--c-gray-50'),
          100: token('--c-gray-100'),
          200: token('--c-gray-200'),
          300: token('--c-gray-300'),
          400: token('--c-gray-400'),
          500: token('--c-gray-500'),
          600: token('--c-gray-600'),
          700: token('--c-gray-700'),
          800: token('--c-gray-800'),
          900: token('--c-gray-900'),
          950: token('--c-gray-950'),
        },
        // Semantic highlights used inside the demos. Only the shades that need
        // to survive both backgrounds are tokenised.
        yellow: {
          400: token('--c-yellow'),
          500: '#eab308',
          600: '#ca8a04',
          900: '#713f12',
        },
        amber: {
          400: token('--c-amber'),
        },
        green: {
          400: token('--c-green'),
          500: '#22c55e',
          600: '#16a34a',
          800: '#166534',
        },
        red: {
          400: token('--c-red'),
          500: '#ef4444',
        },
        cyan: {
          400: token('--c-accent'),
          500: '#06b6d4',
        },
        blue: {
          400: token('--c-blue'),
        },
        purple: {
          400: token('--c-accent-2'),
          600: '#9333ea',
          800: '#6b21a8',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        retro: ['Verdana', 'Geneva', 'Tahoma', 'sans-serif'],
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
            boxShadow: '0 0 5px rgb(var(--c-accent) / 0.2), 0 0 20px rgb(var(--c-accent) / 0.1)'
          },
          '100%': {
            boxShadow: '0 0 10px rgb(var(--c-accent) / 0.4), 0 0 40px rgb(var(--c-accent) / 0.2)'
          }
        }
      }
    },
  },
  plugins: [],
}
