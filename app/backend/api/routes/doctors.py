import datetime
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from pymongo import ReturnDocument

from app.backend.api.deps import get_current_doctor
from app.backend.schemas import AddInfoIn, CorrectClassRequest
from app.backend.database import mongo

router = APIRouter(tags=["doctors"])

@router.post("/slide/{slajd_uid}/share-by-email", status_code=status.HTTP_200_OK)
async def share_slide_by_email(
    slajd_uid: str,
    email: str,
    user=Depends(get_current_doctor),
):
    slide = await mongo.db[mongo.COLL["slajdy"]].find_one(
        {"slajd_uid": slajd_uid},
        {"_id": 0, "slajd_uid": 1, "lekarz_uid": 1}
    )
    if not slide:
        raise HTTPException(status_code=404, detail="Slide not found")

    role_doc = await mongo.db[mongo.COLL["access"]].find_one(
        {"slajd_uid": slajd_uid, "lekarz_uid": user.get("lekarz_uid"), "active": True},
        {"_id": 0, "rola": 1}
    )

    role = role_doc.get("rola") if role_doc else None

    if user.get("rola") != "admin" and role != "owner":
        raise HTTPException(status_code=403, detail="Tylko właściciel slajdu lub admin może udostępniać")

    lekarz = await mongo.db[mongo.COLL["lekarze"]].find_one(
        {"email": email},
        {"_id": 0, "lekarz_uid": 1, "email": 1}
    )
    if not lekarz or not lekarz.get("lekarz_uid"):
        raise HTTPException(status_code=404, detail="Lekarz o podanym e-mailu nie istnieje")

    target_uid = lekarz["lekarz_uid"]
    now = datetime.datetime.utcnow()

    await mongo.db[mongo.COLL["access"]].update_one(
        {"slajd_uid": slajd_uid, "lekarz_uid": target_uid},
        {"$set": {
            "rola": "viewer",
            "active": True,
            "granted_by": user.get("lekarz_uid"),
            "granted_at": now,
            "revoked_at": None,
            "note":""
        }},
        upsert=True
    )

    return {
        "status": "ok",
        "slajd_uid": slajd_uid,
        "lekarz_uid": target_uid,
        "rola": "viewer",
        "active": True
    }

