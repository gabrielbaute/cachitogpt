"""
FastAPI Application Factory module
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.settings.config import Config
from app.backend.api.include_routes import include_routes
from app.backend.api.routes.ui_routes import router as ui_router

def create_app(config: Config) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance
    """
    app: FastAPI = FastAPI(
        title=config.MODEL_NAME,
        description="CachitoGPT: un modelo experimental.",
        version=config.MODEL_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
                                
    # Mount static files (only if directory exists)
    if config.STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=Config.API_ALLOW_ORIGINS,
        allow_credentials=Config.API_ALLOW_CREDENTIALS,
        allow_methods=Config.API_ALLOW_METHODS,
        allow_headers=Config.API_ALLOW_HEADERS,
    )

    include_routes(app)
    app.include_router(ui_router)

    return app