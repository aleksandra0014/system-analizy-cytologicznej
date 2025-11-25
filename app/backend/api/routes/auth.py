from fastapi import APIRouter, HTTPException, Response, Depends
import uuid, datetime
from pydantic import BaseModel,  Field

from app.backend.core.settings import settings
from app.backend.core.security import hash_password, verify_password, create_token
from app.backend.database import mongo
from app.backend.api.deps import get_current_doctor

router = APIRouter(prefix="/auth", tags=["auth"])

class RegisterIn(BaseModel):
    name: str
    surname: str
    email: str
    role: str = "doctor"
    password: str = Field(min_length=8)

class LoginIn(BaseModel):
    email: str
    password: str

@router.post("/register")
async def register_user(payload: RegisterIn):
    exists = await mongo.db[mongo.COLL["doctors"]].find_one({"email": payload.email})
    if exists:
        raise HTTPException(status_code=409, detail="Email already registered")
    now = datetime.datetime.utcnow()
    doc = {
        "doctor_uid": uuid.uuid4().hex,
        "name": payload.name,
        "surname": payload.surname,
        "email": payload.email,
        "role": payload.role,
        "active": True,
        "password_hash": hash_password(payload.password),
        "created_at": now,
    }
    await mongo.db[mongo.COLL["doctors"]].insert_one(doc)
    return {"ok": True}

@router.post("/login")
async def login(payload: LoginIn, response: Response):
    user = await mongo.db[mongo.COLL["doctors"]].find_one({"email": payload.email})
    if not user or not user.get("password_hash") or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(user["email"], user.get("role", "doctor"))
    response.set_cookie(
        key=settings.COOKIE_NAME,
        value=token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path="/",
        max_age=settings.ACCESS_TTL_SECONDS,
    )
    return {"ok": True}

@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(settings.COOKIE_NAME, path="/")
    return {"ok": True}

@router.get("/me")
async def me(user=Depends(get_current_doctor)):
    return user
