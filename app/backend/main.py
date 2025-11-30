import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.backend.core.settings import settings
from app.backend.core.logging import configure_logging
from app.backend.core.errors import http_exception_handler, generic_exception_handler
from app.backend.api import api_router
from app.backend.database.mongo import connect, disconnect, ensure_collections

def build_app() -> FastAPI:
    configure_logging()

    app = FastAPI(title="Cytology API", version="1.0.0")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"]
    )

    # Lifespan
    @app.on_event("startup")
    async def _startup():
        await connect()
        await ensure_collections()

    @app.on_event("shutdown")
    async def _shutdown():
        await disconnect()

    # Handlery wyjątków
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Routery
    app.include_router(api_router)

    return app

app = build_app()

if __name__ == "__main__":
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=settings.PORT, reload=True)

