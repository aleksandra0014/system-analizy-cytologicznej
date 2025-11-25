from typing import Optional, Dict
from pydantic import BaseModel, Field, EmailStr

class RegisterIn(BaseModel):
    name: str
    surname: str
    email: EmailStr
    password: str = Field(min_length=8)
    role: Optional[str] = "doctor"

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class GradcamIn(BaseModel):
    cell_uid: Optional[str] = None
    crop_gridfs_name: Optional[str] = None
    image_url: Optional[str] = None
    architecture: Optional[str] = None  

class LimeIn(BaseModel):
    cell_uid: Optional[str] = None
    features: Optional[Dict[str, float]] = None
    top_labels: Optional[int] = None

class AddInfoIn(BaseModel):
    add_info: Optional[str] = None

class CorrectClassRequest(BaseModel):
    class_corrected: str = Field(..., pattern="^(HSIL|LSIL|NSIL)$")
