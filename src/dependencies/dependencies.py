import os
from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from src.middlewares.auth_middleware import get_api_key 

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(request: Request):
    """
    Função de dependência para verificar a API Key.
    """
    correct_api_key = get_api_key() 

    sent_api_key = await api_key_header(request)

    if not sent_api_key or sent_api_key != correct_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciais inválidas: API Key ausente ou incorreta."
        )

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in ["/", "/openapi.json", "/docs", "/redoc"]:
            response = await call_next(request)
            return response

        try:
            await verify_api_key(request)
            response = await call_next(request)
            return response
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail}
            )