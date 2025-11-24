// import path from "path"
// import tailwindcss from "@tailwindcss/vite"
// import react from "@vitejs/plugin-react"
// import { defineConfig } from "vite"

// // https://vite.dev/config/
// export default defineConfig({
//   plugins: [react(), tailwindcss()],
//   resolve: {
//     alias: {
//       "@": path.resolve(__dirname, "./src"),
//     },
//   },
// })

import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  
  // 🔑 DODANA KONFIGURACJA SERWERA DEWELOPERSKIEGO
  server: {
    // 1. Host: Zapewnia, że serwer nasłuchuje na wszystkich interfejsach 
    //    wewnątrz kontenera (konieczne dla Dockera).
    host: '0.0.0.0', 
    
    // 2. Port: Upewnia się, że Vite działa na oczekiwanym porcie. 
    //    Musisz to zmapować w docker-compose.yml (np. 5173:5173).
    port: 5173,
    
    // 3. Opcjonalne: Użyj opcji 'strictPort', aby uniemożliwić Vite 
    //    automatyczne wybieranie innego portu.
    strictPort: true, 
  },
})