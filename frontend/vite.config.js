import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.VITE_API_TARGET || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // All /api/* requests are forwarded to FastAPI during development.
      // Override the target with VITE_API_TARGET (handy when another worktree
      // is already serving the default port).
      "/api": {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
});
