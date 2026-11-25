import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// dev: `npm run dev` proxies /api to the Node gateway on :8787. build: relative base so dist/ can be hosted anywhere.
export default defineConfig({ plugins: [react()], base: "./", server: { proxy: { "/api": "http://localhost:8787" } } });
