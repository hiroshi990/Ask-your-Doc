from fastapi import APIRouter

from app.models.schemas import ChatRequest, EvaluatedResponse
from app.evals.eval import EvaluationService

router = APIRouter()
rag_service = EvaluationService()


@router.post("", response_model=EvaluatedResponse)
def chat(request: ChatRequest) -> EvaluatedResponse:

    return rag_service.run_evaluation(request)
