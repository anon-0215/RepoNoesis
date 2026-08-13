import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const serverEnv = loadEnv(mode, '..', 'FRONTEND_PORT');
  const parsedPort = Number(serverEnv.FRONTEND_PORT || 5173);
  return {
    plugins: [react()],
    envDir: '..',
    cacheDir: '.vite-cache',
    server: {
      host: '127.0.0.1',
      port: Number.isInteger(parsedPort) && parsedPort > 0 && parsedPort < 65536 ? parsedPort : 5173
    }
  };
});
