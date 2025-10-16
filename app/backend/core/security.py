import time
from jose import jwt
from passlib.context import CryptContext
from app.backend.core.settings import settings

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(p: str) -> str:
    return _pwd_ctx.hash(p)

def verify_password(p: str, h: str) -> bool:
    return _pwd_ctx.verify(p, h)

def create_token(sub: str, role: str) -> str:
    now = int(time.time())
    payload = {"sub": sub, "role": role, "iat": now, "exp": now + settings.ACCESS_TTL_SECONDS}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALG)

def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALG])
