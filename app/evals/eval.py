from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree
from dotenv import load_dotenv
from typing import Optional
load_dotenv()
import json
from openai import OpenAI
from app.src import RAGService
from app.models.schemas import ChatRequest,ChatResponse,EvaluatedResponse



class EvaluationService:
    def __init__(self) -> None:
        self.pipeline = RAGService()
        self.langsmith_client = Client()
        self.judge_llm = OpenAI()
    
    def _score_faithfulness(self,response:ChatResponse) -> float:
        context = [c.text for c in response.retrieved_chunks]
        answer = response.answer
        prompt = f"""You are evaluating whether an AI generated answer is fully
        supported by the given context.Score between 0.0 to 1.0 how well the 
        answer is supported by the context.Respond only with JSON: {{"score":<float>}}
        
        Context:
        {context}
        
        Answer:
        {answer}
        """
        faithfulness_score = self.judge_llm.chat.completions.create(
            model = "gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0,
            response_format={"type":"json_object"}
        )
        score_faithfulness = json.loads(faithfulness_score.choices[0].message.content)["score"]  # pyright: ignore[reportArgumentType]
        return score_faithfulness

    def _score_relevancy(self,response:ChatResponse) -> float:
        question = response.query
        answer = response.answer
        prompt = f"""Checks whether the answer actually addresses the question asked,
        independent of whether it's factually correct.Score from 0.0 to 1.0 how relevant
        the answer is to the question. Respond only with JSON: {{'score':<float>}}

        Question: {question}
        Answer:{answer}
        """
        relevancy_score = self.judge_llm.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        score_relevancy = json.loads(relevancy_score.choices[0].message.content)["score"]  # pyright: ignore[reportArgumentType]
        return score_relevancy

    @traceable(name = "Evaluation")
    def run_evaluation(self,request:ChatRequest) -> EvaluatedResponse:
        result = self.pipeline.chat(request)
        faithfulness_score = self._score_faithfulness(result)
        relevance_score = self._score_relevancy(result)

        run = get_current_run_tree()
        if run is not None:
            self.langsmith_client.create_feedback(
                run_id=run.id, key = "faithfulness", score = faithfulness_score
            )
            self.langsmith_client.create_feedback(
                run_id = run.id, key ="answer_relevancy", score = relevance_score
            )
        
        return EvaluatedResponse(
            query=result.query,
            answer=result.answer,
            citations=result.citations,
            faithfulness_score=faithfulness_score,
            answer_relevancy=relevance_score,
            retrieved_chunks=result.retrieved_chunks,

        )