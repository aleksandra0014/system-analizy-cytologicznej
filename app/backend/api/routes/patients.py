import datetime
from typing import List
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.backend.api.deps import get_current_doctor
from app.backend.database import mongo

router = APIRouter(tags=["patients"])

class PatientOut(BaseModel):
    pacjent_uid: str
    created_at: str | None

class SlideOut(BaseModel):
    slajd_uid: str
    status: str | None = None
    overall_class: str | None = None
    created_at: str | None = None
    add_info: dict | str | None = None

@router.get("/patients")
async def list_patients(user=Depends(get_current_doctor)) -> dict[str, list[PatientOut]]:
    cur = mongo.db[mongo.COLL["pacjenci"]].find({}, {"_id": 0, "pacjent_uid": 1, "created_at": 1}).sort("pacjent_uid", 1)
    out: list[PatientOut] = []
    async for d in cur:
        ct = d.get("created_at")
        out.append(PatientOut(
            pacjent_uid=d.get("pacjent_uid"),
            created_at=ct.isoformat() if isinstance(ct, datetime.datetime) else None
        ))
    return {"patients": out}

@router.get("/patient/{pacjent_uid}/slides")
async def list_slides_for_patient(
    pacjent_uid: str,
    user = Depends(get_current_doctor)
) -> dict[str, List[SlideOut]]:

 
    accessible_ids: list[str] = await mongo.db[mongo.COLL["access"]].distinct(
        "slajd_uid",
        {
            "lekarz_uid": user["lekarz_uid"],
            "active": True,
        },
    )

    if not accessible_ids:
        return {"slides": [], "info": "No accessible slides for this doctor."}
    access_filter = {"slajd_uid": {"$in": accessible_ids}}

    base_filter = {"pacjent_uid": pacjent_uid}
    if access_filter:
        base_filter.update(access_filter)

    cur = mongo.db[mongo.COLL["slajdy"]].find(
        base_filter,
        {
            "_id": 0,
            "slajd_uid": 1,
            "created_at": 1,
            "status": 1,
            "overall_class": 1,
            "add_info": 1,
        },
    ).sort("created_at", -1)

    out: list[SlideOut] = []
    async for d in cur:
        ct = d.get("created_at")
        out.append(
            SlideOut(
                slajd_uid=d.get("slajd_uid"),
                status=d.get("status"),
                overall_class=d.get("overall_class"),
                created_at=ct.isoformat() if isinstance(ct, datetime.datetime) else None,
                add_info=d.get("add_info"),
            )
        )
    return {"slides": out}
