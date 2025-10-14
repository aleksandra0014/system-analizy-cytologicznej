import os, io, re, json, shutil, tempfile, uuid, pathlib, datetime, sys, time
from typing import Optional, Dict
import numpy as np
import cv2
import torch
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, APIRouter, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, EmailStr
from pymongo import ReturnDocument
from passlib.context import CryptContext
from jose import jwt, JWTError
from pydantic import BaseModel, Field


import os as _os
_os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_testing.test import get_info, ARCHITECTURE, vgg_weights, gbm_model, label_encoder
from llm_testing.test_gemini import analyze_with_gemini, analyze_with_ollama
from classification.models import CytologyClassifier
from lime_helper import explainer, predict_fn
from xai_helper import gradcam_on_image

from database import (
    db, slides_bucket, crops_bucket, xai_bucket,
    ensure_collections, COLL
)

# === AUTH ===
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-env")
JWT_ALG = "HS256"
ACCESS_TTL_SECONDS = 60 * 60 * 12
COOKIE_NAME = "lbc_session"
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(p: str) -> str: return pwd_ctx.hash(p)
def verify_password(p: str, h: str) -> bool: return pwd_ctx.verify(p, h)

def create_token(sub: str, role: str) -> str:
    now = int(time.time())
    payload = {"sub": sub, "role": role, "iat": now, "exp": now + ACCESS_TTL_SECONDS}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])

async def get_current_doctor(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = decode_token(token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = await db[COLL["lekarze"]].find_one({"email": email})
    if not user or user.get("aktywny") is False:
        raise HTTPException(status_code=403, detail="User inactive or not found")
    return {
        "email": user["email"],
        "imie": user.get("imie"),
        "nazwisko": user.get("nazwisko"),
        "rola": user.get("rola", "doctor"),
    }

# === MODEL/XAI ===
API_KEY = os.getenv("API_KEY", os.getenv("api_key", ""))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRADCAM_ARCH = ARCHITECTURE
GRADCAM_WEIGHTS = vgg_weights

gradcam_clf = CytologyClassifier(num_classes=3, architecture=GRADCAM_ARCH)
gradcam_clf.load(GRADCAM_WEIGHTS)
gradcam_clf.model.to(device).eval()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:3000", "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def file_url(request: Request, bucket: str, filename: str) -> str:
    return str(request.url_for("get_file_by_name", bucket_name=bucket, filename=filename))

async def gridfs_upload_bytes(bucket, filename: str, data: bytes, content_type: str) -> str:
    bio = io.BytesIO(data)
    await bucket.upload_from_stream(filename, bio, metadata={"contentType": content_type})
    return filename

async def gridfs_upload_disk(bucket, path: str, content_type: str = "image/png") -> str:
    with open(path, "rb") as f:
        data = f.read()
    uuid_name = f"{uuid.uuid4().hex}{pathlib.Path(path).suffix or '.png'}"
    return await gridfs_upload_bytes(bucket, uuid_name, data, content_type)

async def gridfs_upload_np_png(bucket, filename: str, img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="PNG encode failed")
    if not filename.endswith(".png"):
        filename = f"{filename}.png"
    return await gridfs_upload_bytes(bucket, filename, buf.tobytes(), "image/png")

async def gridfs_stream_by_name(bucket, filename: str) -> StreamingResponse:
    grid_out = await bucket.open_download_stream_by_name(filename)
    ctype = (grid_out.metadata or {}).get("contentType", "application/octet-stream")
    async def _iter():
        while True:
            chunk = await grid_out.readchunk()
            if not chunk: break
            yield chunk
    return StreamingResponse(_iter(), media_type=ctype)

files_router = APIRouter()
@files_router.get("/files/{bucket_name}/name/{filename}", name="get_file_by_name")
async def get_file_by_name(bucket_name: str, filename: str):
    if bucket_name == "slides": return await gridfs_stream_by_name(slides_bucket, filename)
    if bucket_name == "crops":  return await gridfs_stream_by_name(crops_bucket, filename)
    if bucket_name == "xai":    return await gridfs_stream_by_name(xai_bucket, filename)
    raise HTTPException(status_code=404, detail="Unknown bucket")
app.include_router(files_router)

@app.on_event("startup")
async def _startup():
    await ensure_collections()

# ===== AUTH ENDPOINTS =====
class RegisterIn(BaseModel):
    imie: str
    nazwisko: str
    email: str
    haslo: str
    rola: Optional[str] = "doctor"

class LoginIn(BaseModel):
    email: str
    haslo: str

@app.post("/auth/register")
async def register_user(payload: RegisterIn):
    exists = await db[COLL["lekarze"]].find_one({"email": payload.email})
    if exists:
        raise HTTPException(status_code=409, detail="Email already registered")
    now = datetime.datetime.utcnow()
    doc = {
        "lekarz_uid": uuid.uuid4().hex,
        "imie": payload.imie,
        "nazwisko": payload.nazwisko,
        "email": payload.email,
        "rola": payload.rola,
        "aktywny": True,
        "password_hash": hash_password(payload.haslo),
        "created_at": now,
    }
    await db[COLL["lekarze"]].insert_one(doc)
    return {"ok": True}

@app.post("/auth/login")
async def login(payload: LoginIn, response: Response):
    user = await db[COLL["lekarze"]].find_one({"email": payload.email})
    if not user or not user.get("password_hash") or not verify_password(payload.haslo, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(user["email"], user.get("rola", "doctor"))
    response.set_cookie(
        key=COOKIE_NAME, value=token, httponly=True, secure=False, samesite="lax", path="/",
        max_age=ACCESS_TTL_SECONDS,
    )
    return {"ok": True}

@app.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}

@app.get("/auth/me")
async def me(user=Depends(get_current_doctor)):
    return user

# ====== API (chronione) ======

class GradcamIn(BaseModel):
    crop_gridfs_name: Optional[str] = None
    image_url: Optional[str] = None
    architecture: Optional[str] = None

class LimeIn(BaseModel):
    komorka_uid: Optional[str] = None
    features: Optional[Dict[str, float]] = None
    top_labels: Optional[int] = None

class AddInfoIn(BaseModel):
    add_info: Optional[str] = None

@app.post("/process-image/")
async def process_image(
    request: Request,
    file: UploadFile = File(...),
    pacjent_id: Optional[str] = None,
    user=Depends(get_current_doctor),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    (features_list, predict_fused, probs, df_preds,
     bbox_image_path, crop_paths) = get_info(tmp_path, show_image=True)

    response = analyze_with_ollama(
        bbox_image_path, features_list, predict_fused, probs, model='qwen2.5vl:7b', stream=False,
        #api_key=API_KEY
    )

    now = datetime.datetime.now()
    pacjent_uid = pacjent_id or "UNKNOWN"

    await db[COLL["pacjenci"]].find_one_and_update(
        {"pacjent_uid": pacjent_uid},
        {"$setOnInsert": {"created_at": now}},
        upsert=True, return_document=ReturnDocument.AFTER
    )

    bbox_name = await gridfs_upload_disk(slides_bucket, bbox_image_path, "image/png")
    bbox_url  = file_url(request, "slides", bbox_name)

    slajd_uid = uuid.uuid4().hex
    slide_doc = {
        "slajd_uid": slajd_uid,
        "pacjent_uid": pacjent_uid,
        "created_at": now,
        "status": "processed",
        "overall_class": None,
        "slide_summary_text": None,
        "bbox_gridfs_name": bbox_name,
        "bbox_url": bbox_url,
        "add_info": None,  # ⬅️ NOWE POLE
    }
    await db[COLL["slajdy"]].insert_one(slide_doc)

    def convert_np(obj):
        import numpy as _np
        if isinstance(obj, _np.generic): return obj.item()
        if isinstance(obj, _np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert_np(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert_np(i) for i in obj]
        return obj

    def read_json(json_data: str | dict):
        if isinstance(json_data, dict): return json_data
        if isinstance(json_data, str):
            s = json_data.strip()
            s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.IGNORECASE)
            s = re.sub(r'\s*```$', '', s)
            return json.loads(s)
        raise TypeError(f"Unsupported type: {type(json_data)}")
    
    def _pick(d: dict, key):
        if not isinstance(d, dict): return None
        if key in d: return d.get(key)
        s = str(key)
        if s in d: return d.get(s)
        try:
            i = int(s)
            if i in d: return d.get(i)
        except Exception:
            pass
        return None

    _CLASS_ORDER = [str(c) for c in getattr(label_encoder, "classes_", ["HSIL","LSIL","NSIL"])]

    def _to_prob_map(p_raw):
        if isinstance(p_raw, dict):
            if "fused" in p_raw: p_raw = p_raw["fused"]
            else: return {str(k): float(v) for k, v in p_raw.items()}
        import numpy as _np
        arr = _np.asarray(p_raw).reshape(-1) if p_raw is not None else _np.asarray([])
        out = {}
        for i, v in enumerate(arr):
          key = _CLASS_ORDER[i] if i < len(_CLASS_ORDER) else str(i)
          try: out[str(key)] = float(v)
          except Exception: out[str(key)] = float(_np.float64(v))
        return out

    response_data = read_json(response)
    slide_summary = response_data.get("slide_summary", {}) if isinstance(response_data, dict) else {}
    overall_class = slide_summary.get("overall_class", "UNKNOWN")
    confidence = slide_summary.get("confidence", "?")
    explanation = slide_summary.get("explanation", "")
    slide_summary_text = f"{overall_class} (confidence {confidence}) — {explanation}".strip()

    await db[COLL["slajdy"]].update_one(
        {"slajd_uid": slajd_uid},
        {"$set": {"overall_class": overall_class, "slide_summary_text": slide_summary_text}}
    )

    cells_explanations = {}
    if isinstance(response_data, dict) and "cells" in response_data:
        cells_list = response_data.get("cells", [])
        if isinstance(cells_list, list):
            for cell_info in cells_list:
                if isinstance(cell_info, dict):
                    cell_id = str(cell_info.get("id", ""))
                    explanation_text = cell_info.get("explanation", "")
                    cells_explanations[cell_id] = {
                        "explanation": explanation_text,
                    }

    crop_public_urls: dict[str, str] = {}
    crop_gridfs_names: dict[str, str] = {}
    komorki_docs = []

    for cell_id, cpath in crop_paths.items():
        try:
            crop_name = await gridfs_upload_disk(crops_bucket, cpath, "image/png")
            curl = file_url(request, "crops", crop_name)
            cid = str(cell_id)

            p_raw        = _pick(probs, cid) or {}
            probs_map    = _to_prob_map(p_raw)

            features_map = _pick(features_list, cid) or {}
            features_map = {
                str(k): (float(v) if hasattr(v, "item") else (float(v) if isinstance(v, (int, float)) else v))
                for k, v in features_map.items()
            }

            predicted = _pick(predict_fused, cid)
            if hasattr(predicted, "item"): predicted = predicted.item()
            predicted = str(predicted) if predicted is not None else "—"

            # Pobranie explanation dla tej komórki
            cell_extra_info = cells_explanations.get(cid, {})
            cell_explanation = cell_extra_info.get("explanation", "")

            crop_public_urls[cid] = curl
            crop_gridfs_names[cid] = crop_name

            komorki_docs.append({
                "komorka_uid": f"{slajd_uid}:{cid}",
                "slajd_uid": slajd_uid,
                "pacjent_uid": pacjent_uid,
                "cell_id": cid,
                "klasa": predicted,
                "probs": probs_map,
                "features": features_map,
                "crop_gridfs_name": crop_name,
                "crop_url": curl,
                "created_at": now,
                "explanation": cell_explanation,
            })
        except Exception:
            continue

    if komorki_docs:
        await db[COLL["komorki"]].insert_many(komorki_docs)

    return JSONResponse({
        "slide_uid": slajd_uid,
        "pacjent_uid": pacjent_uid,
        "bbox_public_url": bbox_url,
        "crop_public_urls": crop_public_urls,
        "crop_gridfs_names": crop_gridfs_names,
        "predict_fused": convert_np(predict_fused),
        "probs": convert_np(probs),
        "features_list": convert_np(features_list),
        "slide_summary_text": slide_summary_text,
        "overall_class": overall_class,
        "add_info": None,
        "response_data": response_data.get("cells"),
        "response": response,
        "cells_explanations": cells_explanations,
    })

@app.post("/gradcam/")
async def gradcam(request: Request, payload: GradcamIn, user=Depends(get_current_doctor)):
    if payload.crop_gridfs_name:
        grid_out = await crops_bucket.open_download_stream_by_name(payload.crop_gridfs_name)
    elif payload.image_url:
        parts = payload.image_url.rstrip("/").split("/")
        try:
            bucket, name_kw, fname = parts[-3], parts[-2], parts[-1]
            assert name_kw == "name"
            bucket_obj = {"slides": slides_bucket, "crops": crops_bucket, "xai": xai_bucket}[bucket]
            grid_out = await bucket_obj.open_download_stream_by_name(fname)
        except Exception:
            raise HTTPException(status_code=400, detail="Unsupported image_url")
    else:
        raise HTTPException(status_code=400, detail="No image name/url provided")

    buf = io.BytesIO()
    while True:
        chunk = await grid_out.readchunk()
        if not chunk: break
        buf.write(chunk)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=500, detail="Cannot decode image")

    import tempfile as _tf, os as _os2
    fd, tmp_path = _tf.mkstemp(suffix=".png")
    _os2.close(fd)
    try:
        ok = cv2.imwrite(tmp_path, img_bgr)
        if not ok:
            raise HTTPException(status_code=500, detail="Cannot write temp image")
        overlay, heatmap, activation, class_idx = gradcam_on_image(
            gradcam_clf, ARCHITECTURE, tmp_path, device
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {e}")
    finally:
        try: _os2.remove(tmp_path)
        except Exception: pass

    ov_name = await gridfs_upload_np_png(xai_bucket, f"{uuid.uuid4().hex}.png", overlay)
    hm_name = await gridfs_upload_np_png(xai_bucket, f"{uuid.uuid4().hex}.png", heatmap)
    ac_name = await gridfs_upload_np_png(xai_bucket, f"{uuid.uuid4().hex}.png", activation)
    overlay_url = file_url(request, "xai", ov_name)
    heatmap_url = file_url(request, "xai", hm_name)
    activation_url = file_url(request, "xai", ac_name)

    class_names = {0: "HSIL", 1: "LSIL", 2: "NSIL"}
    predicted_class = class_names.get(class_idx, str(class_idx))

    await db[COLL["gradcam"]].insert_one({
        "komorka_uid": None,
        "created_at": datetime.datetime.utcnow(),
        "predicted_class": predicted_class,
        "overlay_gridfs_name": ov_name,
        "heatmap_gridfs_name": hm_name,
        "activation_gridfs_name": ac_name,
        "overlay_url": overlay_url,
        "heatmap_url": heatmap_url,
        "activation_url": activation_url
    })

    return JSONResponse({
        "overlay_url": overlay_url,
        "heatmap_url": heatmap_url,
        "activation_url": activation_url,
        "predicted_class": predicted_class
    })

@app.post("/lime/")
async def lime_explain(request: Request, payload: LimeIn, user=Depends(get_current_doctor)):
    features_map = payload.features
    if (not features_map) and payload.komorka_uid:
        kom = await db[COLL["komorki"]].find_one({"komorka_uid": payload.komorka_uid}, {"features": 1})
        if not kom: raise HTTPException(status_code=404, detail="Cell not found")
        features_map = kom.get("features", {})

    row = np.array(list(features_map.values()))
    exp = explainer.explain_instance(
        data_row=row, predict_fn=predict_fn,
        top_labels=payload.top_labels or len(label_encoder.classes_)
    )

    html_str = exp.as_html().encode("utf-8")
    html_name = f"{uuid.uuid4().hex}.html"
    await gridfs_upload_bytes(xai_bucket, html_name, html_str, "text/html")
    html_url = file_url(request, "xai", html_name)

    await db[COLL["lime"]].insert_one({
        "komorka_uid": payload.komorka_uid,
        "created_at": datetime.datetime.utcnow(),
        "html_gridfs_name": html_name,
        "html_url": html_url
    })

    return JSONResponse({"html_url": html_url})

@app.get("/slide/{slajd_uid}")
async def get_slide(slajd_uid: str, request: Request, user=Depends(get_current_doctor)):
    slide = await db[COLL["slajdy"]].find_one({"slajd_uid": slajd_uid})
    if not slide:
        raise HTTPException(status_code=404, detail="Slide not found")

    cur = db[COLL["komorki"]].find({"slajd_uid": slajd_uid})
    komorki = [doc async for doc in cur]

    predict_fused: dict[str, str | int | float] = {}
    probs: dict[str, dict] = {}
    features_list: dict[str, dict] = {}
    crop_public_urls: dict[str, str] = {}
    crop_gridfs_names: dict[str, str] = {}

    def _plain(v):
        try:
            import numpy as np
            if isinstance(v, np.generic): return v.item()
        except Exception:
            pass
        return v

    for k in komorki:
        cell_id = str(k.get("cell_id"))
        predict_fused[cell_id] = _plain(k.get("klasa", "—"))
        pmap = k.get("probs") or {}
        probs[cell_id] = {"fused": {str(kk): _plain(vv) for kk, vv in pmap.items()}}
        fmap = k.get("features") or {}
        features_list[cell_id] = {str(kk): _plain(vv) for kk, vv in fmap.items()}
        if k.get("crop_url"): crop_public_urls[cell_id] = k["crop_url"]
        if k.get("crop_gridfs_name"): crop_gridfs_names[cell_id] = k["crop_gridfs_name"]

    return JSONResponse({
        "slide_uid": slide.get("slajd_uid"),
        "pacjent_uid": slide.get("pacjent_uid"),
        "bbox_public_url": slide.get("bbox_url"),
        "overall_class": slide.get("overall_class"),
        "slide_summary_text": slide.get("slide_summary_text"),
        "predict_fused": predict_fused,
        "probs": probs,
        "features_list": features_list,
        "crop_public_urls": crop_public_urls,
        "crop_gridfs_names": crop_gridfs_names,
        "add_info": slide.get("add_info"),   # ⬅️ NOWE
    })

@app.patch("/slide/{slajd_uid}/add-info")
async def update_add_info(slajd_uid: str, payload: AddInfoIn, user=Depends(get_current_doctor)):
    res = await db[COLL["slajdy"]].find_one_and_update(
        {"slajd_uid": slajd_uid},
        {"$set": {"add_info": payload.add_info}},
        return_document=ReturnDocument.AFTER
    )
    if not res:
        raise HTTPException(status_code=404, detail="Slide not found")
    return {"ok": True, "add_info": res.get("add_info")}

@app.get("/patients")
async def list_patients(user=Depends(get_current_doctor)):
    cur = db[COLL["pacjenci"]].find({}, {"_id": 0, "pacjent_uid": 1, "created_at": 1}).sort("pacjent_uid", 1)
    out = []
    async for d in cur:
        ct = d.get("created_at")
        out.append({"pacjent_uid": d.get("pacjent_uid"), "created_at": ct.isoformat() if isinstance(ct, datetime.datetime) else None})
    return {"patients": out}

@app.get("/patient/{pacjent_uid}/slides")
async def list_slides_for_patient(pacjent_uid: str, user=Depends(get_current_doctor)):
    cur = db[COLL["slajdy"]].find(
        {"pacjent_uid": pacjent_uid},
        {"_id": 0, "slajd_uid": 1, "created_at": 1, "status": 1, "overall_class": 1, "add_info": 1}
    ).sort("created_at", -1)
    out = []
    async for d in cur:
        ct = d.get("created_at")
        out.append({
            "slajd_uid": d.get("slajd_uid"),
            "status": d.get("status"),
            "overall_class": d.get("overall_class"),
            "created_at": ct.isoformat() if isinstance(ct, datetime.datetime) else None,
            "add_info": d.get("add_info"),
        })
    return {"slides": out}


class CorrectClassRequest(BaseModel):
    klasa_corrected: str = Field(..., pattern="^(HSIL|LSIL|NSIL)$")

@app.patch("/cell/{komorka_uid}/correct-class")
async def correct_cell_class(
    komorka_uid: str,
    body: CorrectClassRequest,  
    user=Depends(get_current_doctor),
):

    try:
        result = await db[COLL["komorki"]].update_one(
            {"komorka_uid": komorka_uid},
            {
                "$set": {
                    "klasa_corrected": body.klasa_corrected,
                    "corrected_at": datetime.datetime.utcnow(),
                    "corrected_by": user.get("username") or user.get("email"),
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail=f"Cell {komorka_uid} not found")
        
        return {
            "status": "ok",
            "komorka_uid": komorka_uid,
            "klasa_corrected": body.klasa_corrected
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update: {str(e)}")