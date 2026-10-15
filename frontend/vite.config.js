import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/health": "http://localhost:8000",
      "/models": "http://localhost:8000",
      "/monitoring": "http://localhost:8000",
      "/predict": "http://localhost:8000",
      "/batch-predict": "http://localhost:8000",
    },
  },
});
