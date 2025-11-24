# 1. Używamy oficjalnego obrazu Pythona jako bazy
FROM python:3.11-slim

# 2. Ustawienie zmiennej środowiskowej, która zapobiega buforowaniu wyjścia Pythona
ENV PYTHONUNBUFFERED 1

# 3. Ustawienie katalogu roboczego wewnątrz kontenera
WORKDIR /app

# 4. Kopiowanie pliku z zależnościami. Zakładam, że masz już wygenerowany requirements.txt
COPY requirements.txt .

# 5. Instalacja zależności Pythona
# Pamiętaj, aby dodać zależności dla motor, uvicorn, fastapi i wszystkich innych modułów
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install --no-cache-dir -r requirements.txt

# 6. Kopiowanie całego kodu aplikacji do katalogu roboczego
# Zakładając, że struktura Twojego projektu to np. app/backend/main.py
COPY . /app

# 7. Ustawienie, która część aplikacji ma zostać uruchomiona i jak.
# Używamy gunicorn z workerami uvicorn dla lepszej wydajności w produkcji.
# Parametry:
# - app.backend.main:app -> ścieżka do Twojej instancji FastAPI.
# - --host 0.0.0.0 -> pozwala na dostęp z zewnątrz kontenera (dla dockera).
# - --port 8000 -> standardowy port dla API.
# - --workers 4 -> liczba workerów, dostosuj do liczby rdzeni CPU na serwerze (np. 2 * CORE + 1).

# UWAGA: W main.py masz uvicorn.run z reload=True. Zmieniamy to na komendę Gunicorn + Uvicorn
# w Dockerfile dla środowiska produkcyjnego/konteneryzacji.

# ZMIENIONE POLECENIE URUCHAMIAJĄCE W PRODUKCJI:
# CMD ["gunicorn", "app.backend.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# Opcjonalnie, jeśli chcesz używać tylko Uvicorn, jak w Twoim main.py (mniej skalowalne w produkcji):
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]