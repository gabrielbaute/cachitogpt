from fastapi import APIRouter, HTTPException
from app.backend.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.backend.services import CachitoService

router = APIRouter(prefix="/v1")
service = CachitoService()

@router.post("/chat/completions", summary="Ruta de consultas al modelo y chat",  response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):    
    try:
        response = service.generate_response(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")