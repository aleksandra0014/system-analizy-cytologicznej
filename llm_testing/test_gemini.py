import os
import json
import mimetypes
from string import Template
import numpy as np
from google import genai


def to_builtin(obj):
    """Recursively convert numpy types to Python builtins so they can be JSON-serialized."""
    if isinstance(obj, (np.integer, )):
        return int(obj)
    if isinstance(obj, (np.floating, )):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    return obj

def guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def render_prompt(template_text: str, *, features, predictions, probs) -> str:
    """Render the prompt template with JSON-serialized data using string.Template ($PLACEHOLDER)."""
    features_clean = to_builtin(features)
    preds_clean    = to_builtin(predictions)
    probs_clean    = to_builtin(probs)

    tpl = Template(template_text)
    return tpl.substitute(
        FEATURES_JSON=json.dumps(features_clean, ensure_ascii=False, indent=2),
        PREDICTIONS_JSON=json.dumps(preds_clean,    ensure_ascii=False, indent=2),
        PROBS_JSON=json.dumps(probs_clean,          ensure_ascii=False, indent=2),
    )

def analyze_with_gemini(
    image_path: str,
    features,
    predictions,
    probs,
    api_key: str,
    system_path: str = "llm_testing\system.txt",
    prompt_template_path: str = "llm_testing\prompt.txt",
    model: str = "gemini-2.5-flash",
):
    """
    Build an instruction + data prompt for Gemini (multimodal) using external text files.
    - system_path: path to system description text file
    - prompt_template_path: path to main prompt template (with $FEATURES_JSON etc.)
    Returns: model text (expected JSON).
    """
    if not os.path.exists(system_path):
        raise FileNotFoundError(f"System file not found: {system_path}")
    if not os.path.exists(prompt_template_path):
        raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    system_text   = load_text(system_path)
    template_text = load_text(prompt_template_path)
    prompt_text   = render_prompt(template_text, features=features, predictions=predictions, probs=probs)


    mime = guess_mime(image_path)
    image_data = None
    # Convert BMP to PNG in memory if needed
    if mime == "image/bmp":
        try:
            from PIL import Image
            import io
            with Image.open(image_path) as img:
                with io.BytesIO() as output:
                    img.save(output, format="PNG")
                    image_data = output.getvalue()
            mime = "image/png"
        except ImportError:
            raise RuntimeError("Pillow (PIL) is required to convert BMP images. Please install it with 'pip install pillow'.")
    else:
        with open(image_path, "rb") as f:
            image_data = f.read()
        if not mime.startswith("image/"):
            mime = "image/png"

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                {"role": "user", "parts": [{"text": system_text}]},
                {"role": "user", "parts": [
                    {"text": prompt_text},
                    {"inline_data": {"mime_type": mime, "data": image_data}}
                ]},
            ],
            config=genai.types.GenerateContentConfig(
                temperature=0.6,
                top_p=0.9,
            ),
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini generate_content failed: {e}") from e
    
import os
import base64
import requests

def analyze_with_ollama(
    image_path: str,
    features,
    predictions,
    probs,
    system_path: str = r"llm_testing\system.txt",
    prompt_template_path: str = r"llm_testing\prompt.txt",
    model: str = "llava:latest",
    *,
    stream: bool = False, 
    on_chunk=None, 
):
    """
    Buduje prompt z plików i wysyła go do modelu w Ollama (lokalnie).
    - Obsługuje multimodal (obrazy) dla modeli typu LLaVA/Moondream.
    - system_path: plik z opisem roli/systemu
    - prompt_template_path: plik z szablonem promptu
    Zwraca: treść odpowiedzi modelu (str).
    """

    if not os.path.exists(system_path):
        raise FileNotFoundError(f"System file not found: {system_path}")
    if not os.path.exists(prompt_template_path):
        raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    system_text   = load_text(system_path)
    template_text = load_text(prompt_template_path)
    prompt_text   = render_prompt(template_text, features=features, predictions=predictions, probs=probs)

    mime = guess_mime(image_path)
    if mime == "image/bmp":
        try:
            from PIL import Image
            import io
            with Image.open(image_path) as img:
                with io.BytesIO() as buf:
                    img.save(buf, format="PNG")
                    image_bytes = buf.getvalue()
            mime = "image/png"
        except ImportError:
            raise RuntimeError("Pillow jest wymagany do konwersji BMP → PNG (pip install pillow).")
    else:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt_text, "images": [image_b64]},
        ],
        "stream": stream,
        "options": {
            "temperature": 0.7, 
            "top_p": 0.9
        }
    }

    url = "http://localhost:11434/api/chat"
    # r = requests.post(url, json=response)
    # r.raise_for_status()
    # data = r.json()
    # return data.get("message", {}).get("content") or data.get("response")
    if stream:
        with requests.post(url, json=payload, stream=True) as r:
            r.raise_for_status()
            chunks = []
            for line in r.iter_lines():
                if not line:
                    continue
                # każda linia to JSON (NDJSON)
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = (obj.get("message") or {}).get("content") or obj.get("response")
                if piece:
                    chunks.append(piece)
                    if callable(on_chunk):
                        try:
                            on_chunk(piece)
                        except Exception:
                            pass
            return "".join(chunks)

    # tryb bez streamu: zwykły JSON (z fallbackiem na NDJSON, jeśli serwer jednak zastrumieniuje)
    r = requests.post(url, json=payload)
    r.raise_for_status()
    try:
        data = r.json()
        return (data.get("message") or {}).get("content") or data.get("response")
    except ValueError:
        # fallback: spróbuj sparsować ostatnią poprawną linię NDJSON
        last = None
        for ln in r.text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                last = json.loads(ln)
            except json.JSONDecodeError:
                continue
        if last:
            return (last.get("message") or {}).get("content") or last.get("response")
        raise RuntimeError("Nie udało się sparsować odpowiedzi Ollama (ani JSON, ani NDJSON).")
