## Instalacja i konfiguracja aplikacji

### 1. Uruchomienie kontenerów
```bash
docker-compose up --build
```

### 2. Pobranie modelu VLM
Połącz się z kontenerem Ollama:
```bash
docker exec -it ollama /bin/bash
```

Następnie pobierz model:
```bash
ollama pull qwen2.5-7b-chat
```

### 3. Rejestracja użytkownika
Przejdź do interfejsu Swagger pod adresem [http://localhost:8000/docs](http://localhost:8000/docs) i użyj endpointu `auth_register` do utworzenia konta.

### 4. Gotowe!
Aplikacja jest skonfigurowana i gotowa do użycia.