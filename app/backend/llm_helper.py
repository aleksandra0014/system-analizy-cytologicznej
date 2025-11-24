import os
import json
import mimetypes
from string import Template
from pathlib import Path
import numpy as np
from google import genai
import base64
import requests
from PIL import Image
import io
from typing import Any, Mapping

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
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

_CLASS_MAP = {0: "HSIL", 1: "LSIL", 2: "NSIL"}

def _normalize_overall_class(value: Any) -> str:
    """Przyjmij int 0/1/2 lub str i zwróć etykietę HSIL/LSIL/NSIL."""
    if isinstance(value, int):
        if value not in _CLASS_MAP:
            raise ValueError(f"Invalid overall_class index: {value}")
        return _CLASS_MAP[value]
    if isinstance(value, str):
        v = value.strip().upper()
        if v in {"HSIL", "LSIL", "NSIL"}:
            return v
        raise ValueError(f"Invalid overall_class label: {value!r}")
    raise TypeError(f"overall_class must be int or str, got {type(value).__name__}")

def render_prompt(
    template_text: str,
    *,
    features: Mapping[str, Any],
    predictions: Mapping[str, Any],
    probs: Mapping[str, Any],
    overall_class: Any,
    overall_probs: Any,
) -> str:
    """
    Renderuje template z placeholderami:
      $FEATURES_JSON
      $PREDICTIONS_JSON
      $PROBS_JSON
      $OVERALL_CLASS
      $OVERALL_PROBS_JSON
    """
    features_clean = to_builtin(features)
    preds_clean    = to_builtin(predictions)
    probs_clean    = to_builtin(probs)
    overall_probs_clean = to_builtin(overall_probs)
    overall_class_label = _normalize_overall_class(overall_class)

    tpl = Template(template_text)
    return tpl.substitute(
        FEATURES_JSON=json.dumps(features_clean, ensure_ascii=False, indent=2),
        PREDICTIONS_JSON=json.dumps(preds_clean, ensure_ascii=False, indent=2),
        PROBS_JSON=json.dumps(probs_clean, ensure_ascii=False, indent=2),
        OVERALL_CLASS=overall_class_label,
        OVERALL_PROBS_JSON=json.dumps(overall_probs_clean, ensure_ascii=False, indent=2),
    )

BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PATH = BASE_DIR / "config/system.txt"
PROMPT_TEMPLATE_PATH = BASE_DIR / "config/prompt.txt"

try:
    from dotenv import load_dotenv
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
    else:
        load_dotenv()
except Exception:
    pass

API_KEY = os.getenv("API_KEY", os.getenv("api_key", ""))
def _parse_float_env(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

GEMINI_TEMPERATURE = _parse_float_env("GEMINI_TEMPERATURE", 0.6)
GEMINI_TOP_P = _parse_float_env("GEMINI_TOP_P", 0.9)
OLLAMA_TEMPERATURE = _parse_float_env("OLLAMA_TEMPERATURE", 0.6)
OLLAMA_TOP_P = _parse_float_env("OLLAMA_TOP_P", 0.9)

def analyze_with_gemini(image_path, features, predictions, probs, oveall_class, overall_probs, api_key=API_KEY, model="gemini-2.5-flash", temperature: float = GEMINI_TEMPERATURE, top_p: float = GEMINI_TOP_P):
    system_path = SYSTEM_PATH
    prompt_template_path = PROMPT_TEMPLATE_PATH

    if not system_path.exists():
        raise FileNotFoundError(f"System file not found: {system_path}")
    if not prompt_template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    system_text = system_path.read_text(encoding="utf-8")
    template_text = prompt_template_path.read_text(encoding="utf-8")
    prompt_text   = render_prompt(template_text, features=features, predictions=predictions, probs=probs, overall_class=oveall_class, overall_probs=overall_probs)


    mime = guess_mime(image_path)
    image_data = None
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

    api_key = API_KEY
    if not api_key:
        raise ValueError("Missing API key for Gemini/Google AI. Set API_KEY in app/backend/.env or pass api_key argument.")

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
                temperature=temperature,
                top_p=top_p,
            ),
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini generate_content failed: {e}") from e
    

def analyze_with_ollama(
    image_path: str,
    features,
    predictions,
    probs,
    oveall_class, 
    overall_probs,
    model: str = "llava:latest",
    *,
    stream: bool = False, 
    on_chunk=None, 
    temperature: float = OLLAMA_TEMPERATURE, top_p: float = OLLAMA_TOP_P
):
    """
    Buduje prompt z plików i wysyła go do modelu w Ollama (lokalnie).
    - Obsługuje multimodal (obrazy) dla modeli typu LLaVA/Moondream.
    - system_path: plik z opisem roli/systemu
    - prompt_template_path: plik z szablonem promptu
    Zwraca: treść odpowiedzi modelu (str).
    """
    system_path = SYSTEM_PATH
    prompt_template_path = PROMPT_TEMPLATE_PATH
    if not system_path.exists():
        raise FileNotFoundError(f"System file not found: {system_path}")
    if not prompt_template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    system_text = system_path.read_text(encoding="utf-8")
    template_text = prompt_template_path.read_text(encoding="utf-8")
    prompt_text   = render_prompt(template_text, features=features, predictions=predictions, probs=probs, overall_class=oveall_class, overall_probs=overall_probs)

    mime = guess_mime(image_path)
    if mime == "image/bmp":
        try:
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
        "num_predict": 2048,      
        "num_ctx": 8192,          
        "temperature": temperature,
        "top_p": top_p,
        "top_k": 40,
        "repeat_penalty": 1.05,   
        "repeat_last_n": 256
    }
    }

    # url = "http://localhost:11434/api/chat"
    url = f"{OLLAMA_URL}/api/chat"

    if stream:
        with requests.post(url, json=payload, stream=True) as r:
            r.raise_for_status()
            chunks = []
            for line in r.iter_lines():
                if not line:
                    continue
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

    r = requests.post(url, json=payload)
    r.raise_for_status()
    try:
        data = r.json()
        return (data.get("message") or {}).get("content") or data.get("response")
    except ValueError:
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
