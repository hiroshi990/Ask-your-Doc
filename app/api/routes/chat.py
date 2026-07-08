from fastapi import APIRouter

from app.models.schemas import ChatRequest, EvaluatedResponse
from app.evals.eval import EvaluationService

router = APIRouter()
rag_service = EvaluationService()


@router.post("/chat", response_model=EvaluatedResponse)
async def chat(request: ChatRequest) -> EvaluatedResponse:

    return await rag_service.run_evaluation(request)
