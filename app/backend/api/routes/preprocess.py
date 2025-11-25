import os, re, json, shutil, tempfile, uuid, pathlib, datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, File, UploadFile, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from pymongo import ReturnDocument
import torch

from app.backend.api.deps import get_current_doctor
from app.backend.database import mongo
from app.backend.prediction_helpers import get_info, label_encoder
from app.backend.llm_helper import analyze_with_ollama, analyze_with_gemini
from app.backend.prediction_helpers import API_KEY
from classification_slide.attention_models import AttentionMIL, predict_attention
router = APIRouter(tags=["preprocess"])

load_dotenv()

model_mil = AttentionMIL(
    input_dim=3,
    hidden_dim=128,
    num_classes=len(['HSIL', 'LSIL', 'NSIL']),
    dropout=0.5
)
model_mil.load_state_dict(torch.load(os.getenv("ATTENTION_MODEL"), map_location=torch.device('cpu')))

def file_url(request: Request, bucket: str, filename: str) -> str:
    return str(request.url_for("get_file_by_name", bucket_name=bucket, filename=filename))

async def gridfs_upload_bytes(bucket, filename: str, data: bytes, content_type: str) -> str:
    import io as _io
    bio = _io.BytesIO(data)
    await bucket.upload_from_stream(filename, bio, metadata={"contentType": content_type})
    return filename

async def gridfs_upload_disk(bucket, path: str, content_type: str = "image/png") -> str:
    with open(path, "rb") as f:
        data = f.read()
    uuid_name = f"{uuid.uuid4().hex}{pathlib.Path(path).suffix or '.png'}"
    return await gridfs_upload_bytes(bucket, uuid_name, data, content_type)

@router.post("/process-image/")
async def process_image(
    request: Request,
    file: UploadFile = File(...),
    patient_uid: Optional[str] = None,
    user=Depends(get_current_doctor),
    gemini=False
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    (features_list, predict_fused, probs, probs_list, df_preds, bbox_image_path, crop_paths) = get_info(tmp_path, show_image=True)
    
    _CLASS_ORDER = [str(c) for c in getattr(label_encoder, "classes_", ["HSIL","LSIL","NSIL"])]

    pred, attn, probability = predict_attention(
        model_mil,
        probs_list,
        'test',
        torch.device("cpu"),
        class_names=_CLASS_ORDER,
        visualize=False
    )

    if gemini:
        response = analyze_with_gemini(
                    bbox_image_path, features_list, predict_fused, probs, pred, probability
                )
    else: 
        response = analyze_with_ollama(
                        bbox_image_path, features_list, predict_fused, probs_list, pred, probability, model='qwen2.5vl:7b', stream=False,
                    )
        
    prob_dict = None
    if probability is not None:
        prob_dict = {
            _CLASS_ORDER[i]: float(probability[i]) 
            for i in range(len(probability))
        }
    now = datetime.datetime.utcnow()
    patient_uid = patient_uid or "UNKNOWN"

    await mongo.db[mongo.COLL["patients"]].find_one_and_update(
        {"patient_uid": patient_uid},
        {"$setOnInsert": {"created_at": now}},
        upsert=True, return_document=ReturnDocument.AFTER
    )

    bbox_name = await gridfs_upload_disk(mongo.slides_bucket, bbox_image_path, "image/png")
    bbox_url  = file_url(request, "slides", bbox_name)

    slide_uid = uuid.uuid4().hex
    slide_doc = {
        "slide_uid": slide_uid,
        "patient_uid": patient_uid,
        "created_at": now,
        "status": "processed",
        "overall_class": None,
        "slide_summary_text": None,
        "bbox_gridfs_name": bbox_name,
        "bbox_url": bbox_url,
        "add_info": None,
        "probability": prob_dict,
        "access": [{
            "doctor_uid": user["doctor_uid"],
            "role": "owner",
            "granted_by": user["doctor_uid"],
            "granted_at": now,
            "revoked_at": None,
            "active": True,
            "note": "" 
        }]
    }
    await mongo.db[mongo.COLL["slides"]].insert_one(slide_doc)

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

    response_data = read_json(response) if response else {}
    slide_summary = response_data.get("slide_summary", {}) if isinstance(response_data, dict) else {}

    
    overall_class = _CLASS_ORDER[pred] if pred is not None else "UNKNOWN"
    confidence = probability.max() if probability is not None else "?"
    explanation = slide_summary.get("explanation", "")
    slide_summary_text = f"{overall_class} (confidence {confidence:.3f}) — {explanation}".strip()

    await mongo.db[mongo.COLL["slides"]].update_one(
        {"slide_uid": slide_uid},
        {"$set": {"overall_class": overall_class, "slide_summary_text": slide_summary_text}}
    )

    cells_explanations = {}
    if isinstance(response_data, dict) and "cells" in response_data:
        cells_list = response_data.get("cells", [])
        if isinstance(cells_list, list):
            for cell_info in cells_list:
                if isinstance(cell_info, dict):
                    cid = str(cell_info.get("id", ""))
                    cells_explanations[cid] = {"explanation": cell_info.get("explanation", "")}

    crop_public_urls: dict[str, str] = {}
    crop_gridfs_names: dict[str, str] = {}
    cell_docs = []

    for cell_id, cpath in (crop_paths or {}).items():
        try:
            crop_name = await gridfs_upload_disk(mongo.crops_bucket, cpath, "image/png")
            curl = file_url(request, "crops", crop_name)
            cid = str(cell_id)

            p_raw      = _pick(probs, cid) or {}
            probs_map  = _to_prob_map(p_raw)

            features_map = _pick(features_list, cid) or {}
            features_map = {
                str(k): (float(v) if hasattr(v, "item") else (float(v) if isinstance(v, (int, float)) else v))
                for k, v in features_map.items()
            }

            predicted = _pick(predict_fused, cid)
            if hasattr(predicted, "item"): predicted = predicted.item()
            predicted = str(predicted) if predicted is not None else "—"

            cell_explanation = (cells_explanations.get(cid) or {}).get("explanation", "")

            crop_public_urls[cid] = curl
            crop_gridfs_names[cid] = crop_name

            cell_docs.append({
                "cell_uid": f"{slide_uid}:{cid}",
                "slide_uid": slide_uid,
                "patient_uid": patient_uid,
                "cell_id": cid,
                "class": predicted,
                "probs": probs_map,
                "features": features_map,
                "crop_gridfs_name": crop_name,
                "crop_url": curl,
                "created_at": now,
                "explanations": {"explanation": cell_explanation},
            })
        except Exception:
            continue

    if cell_docs:
        await mongo.db[mongo.COLL["cells"]].insert_many(cell_docs)

    return JSONResponse({
        "slide_uid": slide_uid,
        "patient_uid": patient_uid,
        "bbox_public_url": bbox_url,
        "crop_public_urls": crop_public_urls,
        "crop_gridfs_names": crop_gridfs_names,
        "predict_fused": convert_np(predict_fused),
        "probs": convert_np(probs),
        "features_list": convert_np(features_list),
        "slide_summary_text": slide_summary_text,
        "overall_class": overall_class,
        "add_info": None,
        "response_data": response_data.get("cells") if isinstance(response_data, dict) else None,
        "response": response,
        "cells_explanations": cells_explanations,
        "probability": prob_dict,
    })