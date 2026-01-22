"""
Helper to load all the api routes.
"""

from fastapi import FastAPI

from app.backend.api.routes import health_router
from app.backend.api.routes import cachito_router

def include_routes(app: FastAPI) -> None:
    """Include all API routes in the FastAPI application.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    app.include_router(health_router, tags=["Health"])
    app.include_router(cachito_router, tags=["Cachito"])