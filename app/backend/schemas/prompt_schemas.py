from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict
from datetime import datetime


class ChatCompletionRequest(BaseModel):
    """
    Modelo de peticiones al servicio

    Attributes:
        model(str): modelo a usar
        message(str): mensaje a enviar al modelo
        max_tokens(int): número máximo de tokens a generar
        temperature(float): temperatura/aleatoriedad del modelo
        top_p(float):
        top_k(int):
    """
    model: str = "cachito_2"
    message: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    "model": "cachito_2",
                    "message": "Hola, ¿cómo estás?",
                    "max_tokens": 100,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 40
                }
            ]
        }
    )

# --- Respuesta (Response) ---
class ChatCompletionResponse(BaseModel):
    """
    Resuesta del chat completions.

    Attributes:
        id(str): identificador
        response(str): respuesta del modelo
        created(datetime): fecha de creación
        model(str): modelo usado
        prompt_tokens(int): tokens usados en el prompt
        response_tokens(int): tokens usados en la respuesta
        total_tokens(int): tokens totales usados
    """
    id: str = "chatcmpl-cachito"
    response: str
    created: datetime
    model: str
    prompt_tokens: int
    response_tokens: int
    total_tokens: int

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "chatcmpl-cachito",
                    "response": "Hola, ¿cómo estás?",
                    "created": datetime.now(),
                    "model": "cachito_2",
                    "prompt_tokens": 10,
                    "response_tokens": 5,
                    "total_tokens": 15
                }
            ]
        }
        )