import io, os, uuid, pathlib, datetime
import numpy as np
import cv2
import torch

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from app.backend.api.deps import get_current_doctor
from app.backend.schemas import GradcamIn, LimeIn
from app.backend.database import mongo
from app.backend.lime_helper import explainer, predict_fn
from app.backend.xai_helper import gradcam_on_image
from app.backend.prediction_helpers import ARCHITECTURE, CNN_MODEL_PATH, label_encoder
from classification.models import CytologyClassifier

router = APIRouter(tags=["xai"])
files_router = APIRouter(tags=["files"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_gradcam_clf = None
def get_gradcam_clf():
    global _gradcam_clf
    if _gradcam_clf is None:
        clf = CytologyClassifier(num_classes=len(getattr(label_encoder, "classes_", [0,1,2])), architecture=ARCHITECTURE)
        clf.load(CNN_MODEL_PATH)
        clf.model.to(device).eval()
        _gradcam_clf = clf
    return _gradcam_clf

def file_url(request: Request, bucket: str, filename: str) -> str:
    return str(request.url_for("get_file_by_name", bucket_name=bucket, filename=filename))

async def gridfs_upload_bytes(bucket, filename: str, data: bytes, content_type: str) -> str:
    bio = io.BytesIO(data)
    await bucket.upload_from_stream(filename, bio, metadata={"contentType": content_type})
    return filename

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

@files_router.get("/files/{bucket_name}/name/{filename}", name="get_file_by_name")
async def get_file_by_name(bucket_name: str, filename: str):
    if bucket_name == "slides": return await gridfs_stream_by_name(mongo.slides_bucket, filename)
    if bucket_name == "crops":  return await gridfs_stream_by_name(mongo.crops_bucket, filename)
    if bucket_name == "xai":    return await gridfs_stream_by_name(mongo.xai_bucket, filename)
    raise HTTPException(status_code=404, detail="Unknown bucket")

@router.post("/gradcam/")
async def gradcam(request: Request, payload: GradcamIn, user=Depends(get_current_doctor)):
    if payload.crop_gridfs_name:
        grid_out = await mongo.crops_bucket.open_download_stream_by_name(payload.crop_gridfs_name)
    elif payload.image_url:
        parts = payload.image_url.rstrip("/").split("/")
        try:
            bucket, name_kw, fname = parts[-3], parts[-2], parts[-1]
            assert name_kw == "name"
            bucket_obj = {"slides": mongo.slides_bucket, "crops": mongo.crops_bucket, "xai": mongo.xai_bucket}[bucket]
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

    # zapis tymczasowy aby użyć Twojej funkcji gradcam_on_image
    import tempfile as _tf, os as _os2
    fd, tmp_path = _tf.mkstemp(suffix=".png")
    _os2.close(fd)
    try:
        ok = cv2.imwrite(tmp_path, img_bgr)
        if not ok:
            raise HTTPException(status_code=500, detail="Cannot write temp image")
        clf = get_gradcam_clf()
        overlay, heatmap, activation, class_idx = gradcam_on_image(
            clf, ARCHITECTURE, tmp_path, device
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {e}")
    finally:
        try: _os2.remove(tmp_path)
        except Exception: pass

    ov_name = await gridfs_upload_np_png(mongo.xai_bucket, f"{uuid.uuid4().hex}.png", overlay)
    hm_name = await gridfs_upload_np_png(mongo.xai_bucket, f"{uuid.uuid4().hex}.png", heatmap)
    ac_name = await gridfs_upload_np_png(mongo.xai_bucket, f"{uuid.uuid4().hex}.png", activation)
    overlay_url = file_url(request, "xai", ov_name)
    heatmap_url = file_url(request, "xai", hm_name)
    activation_url = file_url(request, "xai", ac_name)

    class_names = {0: "HSIL", 1: "LSIL", 2: "NSIL"}
    predicted_class = class_names.get(class_idx, str(class_idx))

    await mongo.db[mongo.COLL["gradcam"]].insert_one({
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

@router.post("/lime/")
async def lime_explain(request: Request, payload: LimeIn, user=Depends(get_current_doctor)):
    features_map = payload.features
    if (not features_map) and payload.komorka_uid:
        kom = await mongo.db[mongo.COLL["komorki"]].find_one({"komorka_uid": payload.komorka_uid}, {"features": 1})
        if not kom: raise HTTPException(status_code=404, detail="Cell not found")
        features_map = kom.get("features", {})

    if not features_map:
        raise HTTPException(status_code=400, detail="No features provided")

    row = np.array(list(features_map.values()))
    exp = explainer.explain_instance(
        data_row=row, predict_fn=predict_fn,
        top_labels=payload.top_labels or len(getattr(label_encoder, "classes_", [])) or 3
    )

    html_str = exp.as_html().encode("utf-8")
    html_name = f"{uuid.uuid4().hex}.html"
    await gridfs_upload_bytes(mongo.xai_bucket, html_name, html_str, "text/html")
    html_url = file_url(request, "xai", html_name)

    await mongo.db[mongo.COLL["lime"]].insert_one({
        "komorka_uid": payload.komorka_uid,
        "created_at": datetime.datetime.utcnow(),
        "html_gridfs_name": html_name,
        "html_url": html_url
    })

    return JSONResponse({"html_url": html_url})
