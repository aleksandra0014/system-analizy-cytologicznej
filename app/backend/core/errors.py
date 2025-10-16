from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

logger = logging.getLogger(__name__)

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning("HTTPException %s %s", exc.status_code, exc.detail)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse({"detail": "Internal Server Error"}, status_code=500)
