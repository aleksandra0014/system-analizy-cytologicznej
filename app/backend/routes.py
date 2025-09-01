import os, shutil, tempfile, uuid, pathlib, datetime
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from llm_testing.test import *


app = FastAPI()

# === CORS (dopasuj origin do swojego frontu) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === STATIC ===
STATIC_DIR = pathlib.Path(__file__).parent / "static"
TMP_DIR = STATIC_DIR / "tmp"
os.makedirs(TMP_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def cleanup_tmp(hours: int = 24):
    """Usuwa pliki starsze niż X godzin z static/tmp."""
    now = datetime.datetime.now().timestamp()
    for p in TMP_DIR.glob("*"):
        try:
            if p.is_file() and (now - p.stat().st_mtime) > hours * 3600:
                p.unlink(missing_ok=True)
        except Exception:
            pass  # cicho ignoruj

@app.on_event("startup")
def _startup():
    cleanup_tmp()

@app.post("/process-image/")
async def process_image(request: Request, file: UploadFile = File(...)):
    # zapisz upload w temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # uruchom Twoją funkcję
    features_list, predict_fused, probs, df_preds, bbox_image_path = get_info(tmp_path, show_image=True)

    # skopiuj bbox do static/tmp z losową nazwą
    ext = os.path.splitext(bbox_image_path)[1] or ".png"
    out_name = f"{uuid.uuid4().hex}{ext}"
    out_path = TMP_DIR / out_name
    shutil.copyfile(bbox_image_path, out_path)

    # zbuduj publiczny URL (absolutny)
    # jeśli wolisz względny, użyj tylko: f"/static/tmp/{out_name}"
    public_url = request.url_for("static", path=f"tmp/{out_name}")

    # konwersje numpy -> JSON
    import numpy as np
    def convert_np(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_np(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_np(i) for i in obj]
        return obj

    return JSONResponse({
        "features_list": convert_np(features_list),
        "predict_fused": convert_np(predict_fused),
        "probs": convert_np(probs),
        "df_preds": convert_np(df_preds.to_dict(orient="records")),
        "bbox_public_url": str(public_url)  # <- to wysyłamy do frontu
    })