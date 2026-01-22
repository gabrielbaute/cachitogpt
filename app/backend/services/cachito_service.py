import logging
from pathlib import Path
from datetime import datetime
from app.generator import TextGenerator
from app.schemas.model_response import CachitoResponse
from app.backend.schemas import ChatCompletionRequest, ChatCompletionResponse

class CachitoService:
    def __init__(self):
        self.logger = logging.getLogger("CachitoService")
        self.generator = TextGenerator(model_name="cachito_2")
        self.logger.info("CachitoService initialized and TextGenerator loaded.")

    def generate_response(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Genera una respuesta del modelo a partir de un prompt de entrada.

        Args:
            request(ChatCompletionRequest): La solicitud que contiene el prompt.

        Returns:
            ChatCompletionResponse: La respuesta generada por el modelo.
        """
        try:
            result = self.generator.generate_with_top_filters(
                prompt=request.message,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p
            )
            
            response = ChatCompletionResponse(
                id="chatcmpl-cachito",
                response=result.texto_generado,
                created=datetime.now(),
                model=request.model,
                prompt_tokens=result.tokens_del_mensaje,
                response_tokens=result.tokens_generados,
                total_tokens=result.tokens_del_mensaje + result.tokens_generados
            )
            return response
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}", exc_info=True)
            raise
