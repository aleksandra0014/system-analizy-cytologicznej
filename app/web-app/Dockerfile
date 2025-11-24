# Używamy stabilnego obrazu Node.js
FROM node:20

# Ustawienie katalogu roboczego
WORKDIR /app

# Kopiowanie tylko plików package.json i package-lock.json
# Pomoże to Dockerowi wykorzystać cache, jeśli zależności się nie zmienią
COPY package*.json ./

# Instalacja zależności
# Użyj tej samej komendy, której używasz lokalnie
RUN npm install

# Skopiowanie reszty kodu. W trybie dev, wolumeny (volumes)
# w docker-compose przejmą rolę synchronizacji zmian
COPY . .

# Wystawienie portu, na którym działa serwer deweloperski
# Standardowo dla Vite/React/Vue jest to 5173
EXPOSE 5173

# Uruchomienie komendy 'dev'. 
# UWAGA: W większości frameworków musisz upewnić się, że serwer 
# nasłuchuje na 0.0.0.0, aby był dostępny z zewnątrz kontenera!
# Często wymaga to flagi --host 0.0.0.0 lub ustawienia w vite.config.js/webpack.config.js
CMD ["npm", "run", "dev"]