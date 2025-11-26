import datetime
from fastapi import APIRouter, Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from pymongo import ReturnDocument

from app.backend.api.deps import get_current_doctor
from app.backend.schemas import AddInfoIn, CorrectClassRequest
from app.backend.database import mongo 

router = APIRouter(tags=["cells"])

def _plain(v):
    try:
        import numpy as np
        if isinstance(v, np.generic): 
            return v.item()
    except ImportError:
        pass
    except Exception:
        pass
    return v

@router.get("/slide/{slide_uid}")
async def get_slide(slide_uid: str, request: Request, user=Depends(get_current_doctor)):
    slide = await mongo.db[mongo.COLL["slides"]].find_one({"slide_uid": slide_uid})
    if not slide:
        raise HTTPException(status_code=404, detail="Slide not found")

    cur = mongo.db[mongo.COLL["cells"]].find({"slide_uid": slide_uid})
    cells = [doc async for doc in cur]

    predict_fused: dict[str, str] = {}
    probs: dict[str, dict] = {}
    features_list: dict[str, dict] = {}
    crop_public_urls: dict[str, str] = {}
    crop_gridfs_names: dict[str, str] = {}
    cells_explanations: dict[str, dict] = {}
    
    lime_explanations: dict[str, dict] = {} 
    gradcam_explanations: dict[str, dict] = {} 

    for k in cells:
        cell_id = str(k.get("cell_id"))
        
        predict_fused[cell_id] = _plain(k.get("class", "—"))
        
        pmap = k.get("probs") or {}
        probs[cell_id] = {"fused": {str(kk): _plain(vv) for kk, vv in pmap.items()}}
        
        fmap = k.get("features") or {}
        features_list[cell_id] = {str(kk): _plain(vv) for kk, vv in fmap.items()}
        
        if k.get("crop_url"): crop_public_urls[cell_id] = k["crop_url"]
        if k.get("crop_gridfs_name"): crop_gridfs_names[cell_id] = k["crop_gridfs_name"]

        explanation = k.get("explanation", "")
        if explanation:
            cells_explanations[cell_id] = {"explanation": explanation}
        
        lime_data = k.get("lime_data")
        if lime_data:
            lime_explanations[cell_id] = {
                "html_url": lime_data.get("html_url"),
                "html_gridfs_name": lime_data.get("html_gridfs_name"),
            }
            
        gradcam_data = k.get("gradcam_data")
        if gradcam_data:
            gradcam_explanations[cell_id] = {
                "overlay_url": gradcam_data.get("overlay_url"),
                "heatmap_url": gradcam_data.get("heatmap_url"),
            }

    return JSONResponse({
        "slide_uid": slide.get("slide_uid"),
        "patient_uid": slide.get("patient_uid"), 
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
        "lime_explanations": lime_explanations, 
        "gradcam_explanations": gradcam_explanations,
        "probability": slide.get("probability")
    })

@router.patch("/slide/{slide_uid}/add-info")
async def update_add_info(slide_uid: str, payload: AddInfoIn, user=Depends(get_current_doctor)):
    res = await mongo.db[mongo.COLL["slides"]].find_one_and_update(
        {"slide_uid": slide_uid},
        {"$set": {"add_info": payload.add_info}},
        return_document=ReturnDocument.AFTER
    )
    if not res:
        raise HTTPException(status_code=404, detail="Slide not found")
    return {"ok": True, "add_info": res.get("add_info")}

@router.patch("/cell/{cell_uid}/correct-class")
async def correct_cell_class(
    cell_uid: str, 
    body: CorrectClassRequest,
    user=Depends(get_current_doctor),
):
    result = await mongo.db[mongo.COLL["cells"]].update_one(
        {"cell_uid": cell_uid},
        {
            "$set": {
                "class_corrected": body.class_corrected, 
                "corrected_at": datetime.datetime.utcnow(),
                "corrected_by": user.get("email"),
            }
        }
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=f"Cell {cell_uid} not found")
    return {
        "status": "ok",
        "cell_uid": cell_uid,
        "class_corrected": body.class_corrected 
    }

@router.delete("/slide/{slide_uid}/cell/{cell_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_cell_by_slide_and_cellid(
    slide_uid: str,
    cell_id: str,
    user=Depends(get_current_doctor),
):
    slide = await mongo.db[mongo.COLL["slides"]].find_one({"slide_uid": slide_uid})
    if not slide:
        raise HTTPException(status_code=404, detail=f"Slide {slide_uid} not found")

    doc = await mongo.db[mongo.COLL["cells"]].find_one(
        {"slide_uid": slide_uid, "cell_id": str(cell_id)},
        {"_id": 1, "cell_uid": 1}
    )
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Cell with cell_id={cell_id} on slide={slide_uid} not found"
        )
    
    await mongo.db[mongo.COLL["cells"]].delete_one({"cell_uid": doc["cell_uid"]})