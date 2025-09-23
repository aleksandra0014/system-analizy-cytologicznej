import os
import pathlib
import re
from datetime import datetime
import sys
import json 
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Zaimportuj swoje funkcje z odpowiednich plików
# Poniżej są przykładowe ścieżki - dostosuj je do struktury swojego projektu
from llm_testing.test import get_info
from llm_testing.test_gemini import analyze_with_gemini

# --- Konfiguracja klucza API ---
# Upewnij się, że klucz API jest dostępny jako zmienna środowiskowa
API_KEY = os.getenv("API_KEY", os.getenv("api_key", ""))
if not API_KEY:
    print("Ostrzeżenie: Klucz API nie został znaleziony. Wywołania API mogą się nie udać.")

# --- Twoja funkcja (bez zmian) ---
def process_image(image_path: str, analyze) -> str:
    """
    Przetwarza pojedynczy obraz i zwraca przewidzianą klasę.
    """
    print(f"  Przetwarzanie obrazu: {os.path.basename(image_path)}")
    try:
        (features_list, predict_fused, probs, df_preds,
         bbox_image_path, crop_paths) = get_info(image_path, show_image=False) # show_image=False, aby uniknąć wyskakujących okienek

        response = analyze(
            bbox_image_path, features_list, predict_fused, probs, api_key=API_KEY
        )

        def read_json(json_data: str | dict):
            if isinstance(json_data, dict): return json_data
            if isinstance(json_data, str):
                s = json_data.strip()
                s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.IGNORECASE)
                s = re.sub(r'\s*```$', '', s)
                return json.loads(s)
            raise TypeError(f"Unsupported type: {type(json_data)}")
        
        response_data = read_json(response)
        slide_summary = response_data.get("slide_summary", {}) if isinstance(response_data, dict) else {}
        overall_class = slide_summary.get("overall_class", "UNKNOWN")
        return overall_class
    
    except Exception as e:
        print(f"    Wystąpił błąd podczas przetwarzania obrazu {os.path.basename(image_path)}: {e}")
        return "ERROR"

# --- Funkcja testująca dokładność ---
def test_folder_accuracy(directory_path: str, expected_class: str):
    """
    Testuje dokładność klasyfikacji dla wszystkich obrazów w danym folderze.
    """
    print(f"\n--- Rozpoczynam test dla folderu: '{directory_path}' ---")
    print(f"Oczekiwana klasa dla wszystkich obrazów: '{expected_class}'")
    
    if not os.path.isdir(directory_path):
        print(f"BŁĄD: Folder '{directory_path}' nie istnieje.")
        return

    # Filtrujemy tylko pliki z popularnymi rozszerzeniami obrazów
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(image_extensions)]
    
    total_images = len(image_files)
    if total_images == 0:
        print("W podanym folderze nie znaleziono żadnych obrazów.")
        return

    correct_predictions = 0
    
    for i, filename in enumerate(image_files):
        image_path = os.path.join(directory_path, filename)
        print(f"\n[{i+1}/{total_images}] Testowanie pliku: {filename}")
        
        # Wywołujemy Twoją funkcję, przekazując `analyze_with_gemini` jako funkcję analizującą
        predicted_class = process_image(image_path, analyze_with_gemini)
        
        print(f"    Otrzymana klasa: '{predicted_class}'")
        
        if predicted_class == expected_class:
            correct_predictions += 1
            print("    Wynik: POPRAWNY ✅")
        else:
            print(f"    Wynik: BŁĘDNY ❌ (Oczekiwano: '{expected_class}')")

    # --- Podsumowanie ---
    print("\n--- PODSUMOWANIE TESTU ---")
    print(f"Przetworzono łącznie obrazów: {total_images}")
    print(f"Poprawne predykcje: {correct_predictions}")
    
    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        print(f"Dokładność (Accuracy): {accuracy:.2f}%")
    else:
        print("Dokładność (Accuracy): N/A (brak obrazów)")

# --- Uruchomienie testu ---
if __name__ == "__main__":
    # Konfiguracja testu
    TARGET_FOLDER = r'C:\Users\aleks\OneDrive\Documents\inzynierka\data\LBC_slides\HSIL\pow 10'
    EXPECTED_CLASS = "HSIL"
    
    # Wywołanie funkcji testującej
    test_folder_accuracy(TARGET_FOLDER, EXPECTED_CLASS)