from fastapi import Depends, HTTPException, Request
from jose import JWTError
from app.backend.core.settings import settings
from app.backend.core.security import decode_token
from app.backend.database import mongo

async def get_current_doctor(request: Request):
    token = request.cookies.get(settings.COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = decode_token(token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = await mongo.db[mongo.COLL["doctors"]].find_one({"email": email})
    if not user or user.get("active") is False:
        raise HTTPException(status_code=403, detail="User inactive or not found")
    return {
        "email": user["email"],
        "name": user.get("name"),
        "surname": user.get("surname"),
        "role": user.get("role", "doctor"),
        "doctor_uid": user["doctor_uid"],
    }
