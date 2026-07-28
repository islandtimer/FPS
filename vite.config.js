import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    outDir: 'dist',
    assetsInlineLimit: 0,
    rollupOptions: { output: { manualChunks: { three: ['three'] } } },
  },
  server: { port: 5173, strictPort: true },
});
