"""
Module for UI routes using Jinja2
"""
from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.settings.config import Config

# Instanciamos el router y las plantillas
router = APIRouter(tags=["UI"])
templates = Jinja2Templates(directory=str(Config.UI_DIR))

# --- CONTEXT PROCESSOR ---
def global_context(request: Request):
    """
    Función que define las variables globales.
    """
    return {
        "app_name": Config.MODEL_NAME,
        "version": Config.MODEL_VERSION,
        "current_date": datetime.now().strftime("%Y-%m-%d")
    }

# En lugar de un decorador @, añadimos la función a la lista de procesadores
templates.context_processors.append(global_context)

@router.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """
    Renderiza la página principal e inyecta los metadatos del modelo activo.
    """
    from app.backend.api.routes.cachito_routes import service
    
    # Extraemos la arquitectura y parámetros del generador
    model_info = service.generator.model_config
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "info": model_info
    })