FROM python:3.11-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# ZMIENIONE POLECENIE URUCHAMIAJĄCE W PRODUKCJI:
# CMD ["gunicorn", "app.backend.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]