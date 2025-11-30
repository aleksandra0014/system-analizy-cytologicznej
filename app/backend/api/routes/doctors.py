import datetime
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from pymongo import ReturnDocument

from app.backend.api.deps import get_current_doctor
from app.backend.schemas import AddInfoIn, CorrectClassRequest
from app.backend.database import mongo

router = APIRouter(tags=["doctors"])

@router.post("/slide/{slide_uid}/share-by-email", status_code=status.HTTP_200_OK)
async def share_slide_by_email(
    slide_uid: str,
    email: str,
    user=Depends(get_current_doctor),
):
    user_uid = user.get("doctor_uid")
    
    slide_doc = await mongo.db[mongo.COLL["slides"]].find_one(
        {"slide_uid": slide_uid},
        {"_id": 0, "slide_uid": 1, "access": 1}
    )
    if not slide_doc:
        raise HTTPException(status_code=404, detail="Slide not found")

    access = slide_doc.get("access", [])
    user_access = next(
        (item for item in access if item.get("doctor_uid") == user_uid and item.get("active")),
        None
    )
    role = user_access.get("role") if user_access else None

    if user.get("role") != "admin" and role != "owner":
        raise HTTPException(status_code=403, detail="Tylko właściciel slajdu lub admin może udostępniać")

    doctor_doc = await mongo.db[mongo.COLL["doctors"]].find_one(
        {"email": email},
        {"_id": 0, "doctor_uid": 1, "email": 1}
    )
    if not doctor_doc or not doctor_doc.get("doctor_uid"):
        raise HTTPException(status_code=404, detail="Lekarz o podanym e-mailu nie istnieje")

    target_uid = doctor_doc["doctor_uid"]
    now = datetime.datetime.utcnow()

    new_access_detail = {
        "doctor_uid": target_uid,
        "role": "viewer",
        "active": True,
        "granted_by": user_uid,
        "granted_at": now,
        "revoked_at": None,
        "note": ""
    }

    await mongo.db[mongo.COLL["slides"]].update_one(
        {"slide_uid": slide_uid, "access.doctor_uid": target_uid},
        {"$pull": {"access": {"doctor_uid": target_uid}}}
    )

    await mongo.db[mongo.COLL["slides"]].update_one(
        {"slide_uid": slide_uid},
        {"$push": {"access": new_access_detail}},
        upsert=False
    )

    return {
        "status": "ok",
        "slide_uid": slide_uid,
        "doctor_uid": target_uid,
        "role": "viewer",
        "active": True
    }