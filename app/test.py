import os
import pathlib
import re
from datetime import datetime
import sys
import json
import random

# --- Ścieżki i importy projektu ---
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_testing.test import get_info
from llm_testing.test_gemini import analyze_with_gemini, analyze_with_ollama

# --- Klucz API ---
API_KEY = os.getenv("API_KEY", os.getenv("api_key", ""))
if not API_KEY:
    print("Ostrzeżenie: Klucz API nie został znaleziony. Wywołania API mogą się nie udać.")

# --- Funkcja pomocnicza: czytanie JSON-a z odpowiedzi LLM ---
def _read_json(json_data: str | dict):
    if isinstance(json_data, dict):
        return json_data
    if isinstance(json_data, str):
        s = json_data.strip()
        # Usuwanie ewentualnych fence’ów ```json ... ```
        s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s*```$', '', s)
        return json.loads(s)
    raise TypeError(f"Unsupported type: {type(json_data)}")

# --- Twoja funkcja: przetworzenie pojedynczego obrazu ---
def process_image(image_path: str, analyze) -> str:
    """
    Przetwarza pojedynczy obraz i zwraca przewidzianą klasę.
    """
    print(f"  Przetwarzanie obrazu: {os.path.basename(image_path)}")
    try:
        (features_list, predict_fused, probs, df_preds,
         bbox_image_path, crop_paths) = get_info(
            image_path,
            show_image=True  # ustaw na False, aby uniknąć wyskakujących okienek
        )

        response = analyze(
            bbox_image_path, features_list, predict_fused, probs, model='qwen2.5vl:7b')
            #api_key=API_KEY
        

        response_data = _read_json(response)
        slide_summary = response_data.get("slide_summary", {}) if isinstance(response_data, dict) else {}
        overall_class = slide_summary.get("overall_class", "UNKNOWN")
        return overall_class

    except Exception as e:
        print(f"    Wystąpił błąd podczas przetwarzania obrazu {os.path.basename(image_path)}: {e}")
        return "ERROR"

# --- Funkcja testująca dokładność (z wyborem podzbioru) ---
def test_folder_accuracy(
    directory_path: str,
    expected_class: str,
    max_images: int | None = None,
    random_pick: bool = True,
    seed: int | None = 42
):
    """
    Testuje dokładność klasyfikacji dla obrazów w folderze.

    Parametry:
    - directory_path: ścieżka do folderu z obrazami.
    - expected_class: oczekiwana etykieta dla wszystkich obrazów w folderze.
    - max_images: ile obrazów przetestować (None = wszystkie).
    - random_pick: czy wybierać losowo (True) czy brać pierwsze N po sortowaniu (False).
    - seed: ziarno losowości dla powtarzalności (gdy random_pick=True).
    """
    print(f"\n--- Rozpoczynam test dla folderu: '{directory_path}' ---")
    print(f"Oczekiwana klasa dla wszystkich obrazów: '{expected_class}'")

    if not os.path.isdir(directory_path):
        print(f"BŁĄD: Folder '{directory_path}' nie istnieje.")
        return

    # Dozwolone rozszerzenia plików graficznych
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # Zbierz wszystkie pliki graficzne
    all_files = [f for f in os.listdir(directory_path) if f.lower().endswith(image_extensions)]
    total_available = len(all_files)
    if total_available == 0:
        print("W podanym folderze nie znaleziono żadnych obrazów.")
        return

    # Wybór podzbioru do testu
    if max_images is not None and max_images < total_available:
        if random_pick:
            if seed is not None:
                random.seed(seed)
            image_files = random.sample(all_files, k=max_images)
            picked_info = f"losowo {max_images} z {total_available} (seed={seed})"
        else:
            image_files = sorted(all_files)[:max_images]
            picked_info = f"pierwsze {max_images} z {total_available} (po sortowaniu)"
    else:
        image_files = all_files
        picked_info = f"wszystkie {total_available}"

    print(f"Wybrane pliki: {picked_info}")

    correct_predictions = 0
    total_images = len(image_files)

    # Główna pętla testowa
    for i, filename in enumerate(image_files):
        image_path = os.path.join(directory_path, filename)
        print(f"\n[{i+1}/{total_images}] Testowanie pliku: {filename}")

        # Wywołujemy Twoją funkcję, przekazując analyze_with_gemini jako analizator
        predicted_class = process_image(image_path, analyze_with_ollama)

        print(f"    Otrzymana klasa: '{predicted_class}'")
        if predicted_class == expected_class:
            correct_predictions += 1
            print("    Wynik: POPRAWNY ✅")
        else:
            print(f"    Wynik: BŁĘDNY ❌ (Oczekiwano: '{expected_class}')")

    # --- Podsumowanie ---
    print("\n--- PODSUMOWANIE TESTU ---")
    print(f"Dostępnych obrazów w folderze: {total_available}")
    print(f"Przetestowanych obrazów: {total_images}")
    print(f"Poprawne predykcje: {correct_predictions}")
    accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0.0
    print(f"Dokładność (Accuracy): {accuracy:.2f}%")

# --- Uruchomienie testu ---
if __name__ == "__main__":
    # Konfiguracja testu
    TARGET_FOLDER = r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\LBC_slides\LSIL\pow 40'
    EXPECTED_CLASS = "LSIL"

    # Przykład 1: przetestuj dokładnie 10 losowych obrazów (powtarzalnie dzięki seed=42)
    test_folder_accuracy(
        TARGET_FOLDER,
        EXPECTED_CLASS,
        max_images=10,
        random_pick=True,
        seed=42
    )

    # Przykład 2: (odkomentuj, jeśli wolisz „pierwsze 10” po sortowaniu)
    # test_folder_accuracy(
    #     TARGET_FOLDER,
    #     EXPECTED_CLASS,
    #     max_images=10,
    #     random_pick=False
    # )


### gemini dla 10: 
# --- PODSUMOWANIE TESTU --- HSIL
# Dostępnych obrazów w folderze: 62
# Przetestowanych obrazów: 10
# Poprawne predykcje: 8
# Dokładność (Accuracy): 80.00%

# --- PODSUMOWANIE TESTU --- NSIL 
# Dostępnych obrazów w folderze: 76
# Przetestowanych obrazów: 10
# Poprawne predykcje: 5
# Dokładność (Accuracy): 50.00%

# --- PODSUMOWANIE TESTU --- lsil
# Dostępnych obrazów w folderze: 30
# Przetestowanych obrazów: 10
# Poprawne predykcje: 0
# Dokładność (Accuracy): 0.00%