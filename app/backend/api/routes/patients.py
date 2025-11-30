import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.backend.api.deps import get_current_doctor
from app.backend.database import mongo

router = APIRouter(tags=["patients"])

class PatientOut(BaseModel):
    patient_uid: str
    created_at: str | None

class SlideOut(BaseModel):
    slide_uid: str
    status: str | None = None
    overall_class: str | None = None
    created_at: str | None = None
    add_info: Dict[str, Any] | str | None = None

class SlidesResponse(BaseModel):
    slides: List[SlideOut]
    info: Optional[str] = None
    
@router.get("/patients")
async def list_patients(user=Depends(get_current_doctor)) -> dict[str, list[PatientOut]]:
    cur = mongo.db[mongo.COLL["patients"]].find({}, {"_id": 0, "patient_uid": 1, "created_at": 1}).sort("patient_uid", 1)
    out: list[PatientOut] = []
    async for d in cur:
        ct = d.get("created_at")
        out.append(PatientOut(
            patient_uid=d.get("patient_uid"),
            created_at=ct.isoformat() if isinstance(ct, datetime.datetime) else None
        ))
    return {"patients": out}

@router.get("/patient/{patient_uid}/slides", response_model=SlidesResponse)
async def list_slides_for_patient(
    patient_uid: str,
    user = Depends(get_current_doctor)
):

    pipeline = [
        {"$match": {
            "patient_uid": patient_uid,
            "access": {
                "$elemMatch": {
                    "doctor_uid": user["doctor_uid"], 
                    "active": True
                }
            }
        }},
        {"$project": {
            "_id": 0,
            "slide_uid": 1,
            "created_at": 1,
            "status": 1,
            "overall_class": 1,
            "add_info": 1,
        }},
        {"$sort": {"created_at": -1}}
    ]

    cur = mongo.db[mongo.COLL["slides"]].aggregate(pipeline)

    out: list[SlideOut] = []
    async for d in cur:
        ct = d.get("created_at")
        out.append(
            SlideOut(
                slide_uid=d.get("slide_uid"),
                status=d.get("status"),
                overall_class=d.get("overall_class"),
                created_at=ct.isoformat() if isinstance(ct, datetime.datetime) else None,
                add_info=d.get("add_info"),
            )
        )
    
    if not out:
        patient_has_slides = await mongo.db[mongo.COLL["slides"]].find_one(
            {"patient_uid": patient_uid}, {"_id": 1}
        )
        if patient_has_slides:
            return {"slides": [], "info": "Slides exist for this patient, but no active access granted to the doctor."}
        else:
            return {"slides": [], "info": "No slides found for this patient."}
            
    return {"slides": out}