import json
import re
import os, shutil, tempfile, uuid, pathlib, datetime, sys

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd

from llm_testing.test import get_info, ARCHITECTURE, vgg_weights, gbm_model, label_encoder
from llm_testing.test_gemini import analyze_with_ollama, analyze_with_gemini
from classification.models import CytologyClassifier
from lime_helper import explainer, predict_fn
from xai_helper import gradcam_on_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = pathlib.Path(__file__).parent / "static"
TMP_DIR = STATIC_DIR / "tmp"
os.makedirs(TMP_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def cleanup_tmp(hours: int = 24):
    now = datetime.datetime.now().timestamp()
    for p in TMP_DIR.glob("*"):
        try:
            if p.is_file() and (now - p.stat().st_mtime) > hours * 3600:
                p.unlink(missing_ok=True)
        except Exception:
            pass

@app.on_event("startup")
def _startup():
    cleanup_tmp()

API_KEY = os.getenv("API_KEY", os.getenv("api_key", ''))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRADCAM_ARCH = ARCHITECTURE
GRADCAM_WEIGHTS = vgg_weights

gradcam_clf = CytologyClassifier(num_classes=3, architecture=GRADCAM_ARCH)
gradcam_clf.load(GRADCAM_WEIGHTS)      
gradcam_clf.model.to(device).eval()


def _local_path_from_static_url(image_url: str) -> pathlib.Path:
    fname = image_url.split("/")[-1]
    local = (TMP_DIR / fname).resolve()
    if not str(local).startswith(str(TMP_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not local.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return local

def _save_img_return_url(request: Request, img_bgr: np.ndarray) -> str:
    name = f"{uuid.uuid4().hex}.png"
    path = TMP_DIR / name
    cv2.imwrite(str(path), img_bgr)
    return str(request.url_for("static", path=f"tmp/{name}"))


@app.post("/process-image/")
async def process_image(request: Request, file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    (features_list, predict_fused, probs, df_preds,
     bbox_image_path, crop_paths) = get_info(tmp_path, show_image=True)

    response = analyze_with_gemini(
        bbox_image_path, features_list, predict_fused, probs, api_key=API_KEY
        # model='qwen2.5vl:7b', stream=True
    )

    ext = os.path.splitext(bbox_image_path)[1] or ".png"
    out_name = f"{uuid.uuid4().hex}{ext}"
    out_path = TMP_DIR / out_name
    shutil.copyfile(bbox_image_path, out_path)
    bbox_public_url = request.url_for("static", path=f"tmp/{out_name}")

    crop_public_urls = {}
    for idx, cpath in crop_paths.items():
        cext = os.path.splitext(cpath)[1] or ".png"
        cout_name = f"{uuid.uuid4().hex}{cext}"
        cout_path = TMP_DIR / cout_name
        try:
            shutil.copyfile(cpath, cout_path)
            crop_public_urls[str(idx)] = str(request.url_for("static", path=f"tmp/{cout_name}"))
        except Exception:
            continue

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

    def read_json(json_data: str | dict):
        if isinstance(json_data, dict):
            return json_data
        if isinstance(json_data, str):
            s = json_data.strip()
            s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.IGNORECASE)
            s = re.sub(r'\s*```$', '', s)
            return json.loads(s)
        raise TypeError(f"Unsupported type: {type(json_data)}")

    response_data = read_json(response)
    slide_summary = response_data.get("slide_summary", {}) if isinstance(response_data, dict) else {}
    overall_class = slide_summary.get("overall_class", "UNKNOWN")
    confidence = slide_summary.get("confidence", "?")
    explanation = slide_summary.get("explanation", "")

    return JSONResponse({
        "features_list": convert_np(features_list),
        "predict_fused": convert_np(predict_fused),
        "probs": convert_np(probs),
        "df_preds": convert_np(df_preds.to_dict(orient="records")),
        "bbox_public_url": str(bbox_public_url),
        "crop_public_urls": crop_public_urls,
        "response": response,
        "slide_summary_text": f"{overall_class} (confidence {confidence}) — {explanation}".strip(),
        "overall_class": overall_class
    })



class GradcamIn(BaseModel):
    image_url: str
    architecture: Optional[str] = None  




@app.post("/gradcam/")
async def gradcam(request: Request, payload: GradcamIn):

    local_path = _local_path_from_static_url(payload.image_url)
    arch = GRADCAM_ARCH 

    try:
        overlay, heatmap, activation, class_idx = gradcam_on_image(
            gradcam_clf, arch, local_path, device
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {e}")

    overlay_url = _save_img_return_url(request, overlay)
    heatmap_url = _save_img_return_url(request, heatmap)
    activation_url = _save_img_return_url(request, activation)

    class_names = {0: "HSIL", 1: "LSIL", 2: "NSIL"}
    predicted_class = class_names.get(class_idx, str(class_idx))

    return JSONResponse({
        "overlay_url": overlay_url,
        "heatmap_url": heatmap_url,
        "activation_url": activation_url,
        "predicted_class": predicted_class
    })


class LimeIn(BaseModel):
    cell_id: str  
    features: Dict

@app.post("/lime/")
async def lime_explain(request: Request, payload: LimeIn):
    
    row = payload.features.values()
    row = np.array(list(row))

    exp = explainer.explain_instance(
        data_row=row,
        predict_fn=predict_fn, 
        top_labels=len(label_encoder.classes_))

    # zapis do HTML
    html_str = exp.as_html()
    html_name = f"{uuid.uuid4().hex}.html"
    html_path = TMP_DIR / html_name
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    html_url = str(request.url_for("static", path=f"tmp/{html_name}"))
    return JSONResponse({"html_url": html_url})
