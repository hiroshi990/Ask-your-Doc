from fastapi import APIRouter

from app.models.schemas import ChatRequest, ChatResponse
from app.src.rag_service import RAGService

router = APIRouter()
rag_service = RAGService()


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:

    return rag_service.chat(request)
