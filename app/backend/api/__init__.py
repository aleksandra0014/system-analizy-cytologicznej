from fastapi import APIRouter
from .routes import auth, patients, cells, preprocess, xai, doctors

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(patients.router)
api_router.include_router(cells.router)
api_router.include_router(doctors.router)
api_router.include_router(preprocess.router)
api_router.include_router(xai.files_router)  # endpoint do pobierania z GridFS
api_router.include_router(xai.router)
