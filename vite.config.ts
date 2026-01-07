import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  // Use root base for custom domain deployment
  base: '/',
  // Include binary model files as assets
  assetsInclude: ['**/*.tbin'],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@/components': resolve(__dirname, 'src/components'),
      '@/utils': resolve(__dirname, 'src/utils'),
      '@/types': resolve(__dirname, 'src/types'),
      '@/styles': resolve(__dirname, 'src/styles'),
      '@/templates': resolve(__dirname, 'src/templates')
    }
  },
  server: {
    port: 3000,
    open: true,
    host: true,
    // Handle SPA routing - always serve index.html for any route
    middlewareMode: false
  },
  preview: {
    port: 3000,
    open: true,
    host: true
  },
  appType: 'spa',
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['three'],
          utils: ['src/utils/dom.ts']
        }
      }
    }
  },
});