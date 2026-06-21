import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any,Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics import _answer_relevance
from ragas.llms import LangchainLLMWrapper
from openai import OpenAI

from app.src.rag_service import RAGService
from app.config import Settings, get_settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    EvaluatedResponse
)

logger = logging.getLogger(__name__)

class RAGASEvaluator:
    def __init__(self,settings:Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.pipeline = RAGService
        self.result_dir : Path = self.settings.result_dir
        self.client = OpenAI(
            api_key=(self.settings.openai_api_key))
        self.ragas_llm = LangchainLLMWrapper(self.client)
        self.metrics =[faithfulness,_answer_relevance]

        for metric in self.metrics:
            metric.llm = self.ragas_llm

    def _run_rag_service(self,question:ChatRequest) -> ChatResponse:
        result = self.pipeline.chat(question)

        return result
    
    def run(self,request:ChatRequest) -> dict[str,Any]:
        logger.info("Running RAG pipeline")
        output = self._run_rag_service(request)
        query = output.query
        contexts = output.retrieved_chunks
        answer= output.answer

        ragas_dataset = Dataset.from_dict({
            "question":query,
            "answer":answer,
            "contexts":contexts
        })
        
        logger.info('Running Evaluation')
        ragas_result = evaluate(
            dataset = ragas_dataset,
            metrics = self.metrics
        )

        scores = ragas_result.to_pandas().to_dict(orient='list')

        return scores



        


        

        

        



        

