import datetime
from fastapi import APIRouter, Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from pymongo import ReturnDocument

from app.backend.api.deps import get_current_doctor
from app.backend.schemas import AddInfoIn, CorrectClassRequest
from app.backend.database import mongo

router = APIRouter(tags=["cells"])

@router.get("/slide/{slajd_uid}")
async def get_slide(slajd_uid: str, request: Request, user=Depends(get_current_doctor)):
    slide = await mongo.db[mongo.COLL["slajdy"]].find_one({"slajd_uid": slajd_uid})
    if not slide:
        raise HTTPException(status_code=404, detail="Slide not found")

    cur = mongo.db[mongo.COLL["komorki"]].find({"slajd_uid": slajd_uid})
    komorki = [doc async for doc in cur]

    predict_fused: dict[str, str | int | float] = {}
    probs: dict[str, dict] = {}
    features_list: dict[str, dict] = {}
    crop_public_urls: dict[str, str] = {}
    crop_gridfs_names: dict[str, str] = {}
    cells_explanations: dict[str, dict] = {}  # ← DODANE

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
        
        explanation = k.get("explanation", "")
        if explanation:
            cells_explanations[cell_id] = {"explanation": explanation}

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
        "add_info": slide.get("add_info"),
        "cells_explanations": cells_explanations, 
    })

@router.patch("/slide/{slajd_uid}/add-info")
async def update_add_info(slajd_uid: str, payload: AddInfoIn, user=Depends(get_current_doctor)):
    res = await mongo.db[mongo.COLL["slajdy"]].find_one_and_update(
        {"slajd_uid": slajd_uid},
        {"$set": {"add_info": payload.add_info}},
        return_document=ReturnDocument.AFTER
    )
    if not res:
        raise HTTPException(status_code=404, detail="Slide not found")
    return {"ok": True, "add_info": res.get("add_info")}

@router.patch("/cell/{komorka_uid}/correct-class")
async def correct_cell_class(
    komorka_uid: str,
    body: CorrectClassRequest,
    user=Depends(get_current_doctor),
):
    result = await mongo.db[mongo.COLL["komorki"]].update_one(
        {"komorka_uid": komorka_uid},
        {
            "$set": {
                "klasa_corrected": body.klasa_corrected,
                "corrected_at": datetime.datetime.utcnow(),
                "corrected_by": user.get("email"),
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

from fastapi import status

@router.delete("/slide/{slajd_uid}/cell/{cell_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_cell_by_slide_and_cellid(
    slajd_uid: str,
    cell_id: str,
    user=Depends(get_current_doctor),
):
    slide = await mongo.db[mongo.COLL["slajdy"]].find_one({"slajd_uid": slajd_uid})
    if not slide:
        raise HTTPException(status_code=404, detail=f"Slide {slajd_uid} not found")

    doc = await mongo.db[mongo.COLL["komorki"]].find_one(
        {"slajd_uid": slajd_uid, "cell_id": str(cell_id)},
        {"_id": 1, "komorka_uid": 1}
    )
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Cell with cell_id={cell_id} on slide={slajd_uid} not found"
        )

    await mongo.db[mongo.COLL["komorki"]].delete_one({"komorka_uid": doc["komorka_uid"]})
