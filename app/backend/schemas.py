from typing import Optional, Dict
from pydantic import BaseModel, Field, EmailStr

class RegisterIn(BaseModel):
    imie: str
    nazwisko: str
    email: EmailStr
    haslo: str = Field(min_length=8)
    rola: Optional[str] = "doctor"

class LoginIn(BaseModel):
    email: EmailStr
    haslo: str

class GradcamIn(BaseModel):
    crop_gridfs_name: Optional[str] = None
    image_url: Optional[str] = None
    architecture: Optional[str] = None  

class LimeIn(BaseModel):
    komorka_uid: Optional[str] = None
    features: Optional[Dict[str, float]] = None
    top_labels: Optional[int] = None

class AddInfoIn(BaseModel):
    add_info: Optional[str] = None

class CorrectClassRequest(BaseModel):
    klasa_corrected: str = Field(..., pattern="^(HSIL|LSIL|NSIL)$")
