## Instalacja i konfiguracja aplikacji

### 1. Pobranie modeli 

* **Pobierz Modele:**
    * Skorzystaj z linku do Dysku Google, aby pobrać wszystkie pliki modeli: (https://drive.google.com/drive/folders/1N_Wlv6McMAUv3i7iV_oHIK61UxiQh6Gu?usp=drive_link).
* **Utwórz Folder:**
    * W głównym katalogu projektu (tam, gdzie znajduje się plik `docker-compose.yml`) utwórz nowy folder o nazwie: **`models`**.
* **Wklej Pliki:**
    * Wklej **pobrane pliki modeli** do nowo utworzonego folderu **`models`**.

### 2. Uruchomienie kontenerów
```bash
docker-compose up --build
```

### 3. Pobranie modelu VLM
Połącz się z kontenerem Ollama:
```bash
docker exec -it cytology_ollama2 /bin/bash
```

Następnie pobierz model:
```bash
ollama pull qwen2.5vl:7b
```
* Aby model działał odpowiednio w przypadku słabszych sprzętów należy ustawić limit pamięci Docker na minimum 16GB. * 

### 4. Rejestracja użytkownika
Przejdź do interfejsu Swagger pod adresem [http://localhost:8000/docs](http://localhost:8000/docs) i użyj endpointu `auth_register` do utworzenia konta.

### 5. Gotowe!
Aplikacja jest skonfigurowana i gotowa do użycia.
