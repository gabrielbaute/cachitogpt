from pydantic import BaseModel

class CachitoResponse(BaseModel):
    """
    Respuesta del modelo

    Keywords:
        id: Nombre del modelo
        tokens_del_mensaje: Número de tokens del mensaje
        tokens_generados: Número de tokens generados
        texto_generado: Texto generado por el modelo
    """
    id: str
    tokens_del_mensaje: int
    tokens_generados: int
    texto_generado: str