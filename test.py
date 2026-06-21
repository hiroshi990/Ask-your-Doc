"""
Minimal end-to-end test for RAGAS evaluation (faithfulness + answer_relevancy).

Pipeline used for this test (intentionally simple, no vector store):
    1. A tiny in-memory "knowledge base" of text snippets
    2. BGE (sentence-transformers) embeds the KB once, and embeds each query
    3. Retrieval = np.dot(query_vector, doc_vectors) — plain cosine similarity
       since BGE embeddings are already L2-normalized
    4. gpt-4o-mini generates the answer from the retrieved context
    5. RAGAS scores the answer for faithfulness + answer_relevancy

Run:
    pip install ragas openai sentence-transformers numpy langchain-openai datasets
    export OPENAI_API_KEY=sk-...
    python test_ragas_eval.py
"""

import os
from typing import Any
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from app.ingestion.embedder import BGEEmbedder
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI


# ─── 1. Tiny knowledge base ────────────────────────────────────────────────

KNOWLEDGE_BASE = [
    "Our return policy allows returns within 30 days of purchase with a valid receipt. "
    "Damaged or defective items are eligible for a full refund.",

    "Standard shipping within the country takes 3-5 business days. "
    "International shipping typically takes 7-14 business days depending on the destination.",

    "We accept Visa, Mastercard, American Express, and PayPal as payment methods. "
    "Cash on delivery is not currently supported.",

    "To reset your password, go to the login page and click 'Forgot Password'. "
    "An email with reset instructions will be sent to your registered email address.",

    "Our customer support team is available Monday to Friday, 9 AM to 6 PM EST. "
    "You can reach us via email at support@example.com or through live chat on our website.",
]


# ─── 2. Embedder (BGE) ──────────────────────────────────────────────────────

# class BGEEmbedder:
#     def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
#         self.model = SentenceTransformer(model_name)

#     def embed_passages(self, texts: list[str]) -> np.ndarray:
#         """No prefix needed for passages — BGE recommendation."""
#         return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

#     def embed_query(self, query: str) -> np.ndarray:
#         """BGE recommends a prefix for queries to align the embedding space."""
#         prefixed = f"Represent this sentence for searching relevant passages: {query}"
#         return self.model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)


# ─── 3. Retrieval — plain np.dot, no vector store ──────────────────────────

def retrieve_top_k(
    query_vector: np.ndarray,
    doc_vectors: np.ndarray,
    documents: list[str],
    top_k: int = 2,
) -> list[str]:
    """
    Since BGE embeddings are L2-normalized, dot product == cosine similarity.
    doc_vectors shape: (num_docs, dim)
    query_vector shape: (dim,)
    """
    scores = np.dot(doc_vectors, query_vector)  # shape: (num_docs,)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in top_indices]


# ─── 4. Generation (gpt-4o-mini) ───────────────────────────────────────────

def generate_answer(client: OpenAI, question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"- {c}" for c in contexts)

    prompt = f"""Answer the question using ONLY the context below. \
If the answer isn't in the context, say you don't have that information.

Context:
{context_block}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


# ─── 5. Tiny RAG pipeline tying it together ────────────────────────────────

class MiniRAGPipeline:
    def __init__(self):
        self.embedder = BGEEmbedder()
        self.openai_client = OpenAI()  # reads OPENAI_API_KEY from env

        self.documents = KNOWLEDGE_BASE
        self.doc_vectors = self.embedder.embed_texts(self.documents)

    def query(self, question: str, top_k: int = 2) -> dict[str, Any]:
        query_vector = self.embedder.embed_query(question)
        contexts = retrieve_top_k(query_vector, self.doc_vectors, self.documents, top_k)
        answer = generate_answer(self.openai_client, question, contexts)

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }


# ─── 6. RAGAS scoring (faithfulness + answer_relevancy only) ──────────────

def score_with_ragas(results: list[dict[str, Any]]) -> dict[str,Any]:
    judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

    metrics = [faithfulness, answer_relevancy]
    for m in metrics:
        m.llm = judge_llm

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
    })

    eval_result = evaluate(dataset=dataset, metrics=metrics)
    eval_result = eval_result.to_pandas().to_dict(orient='list')
    return eval_result


# ─── 7. Run the test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this script.")

    test_questions = [
        "What is the return policy for damaged items?",
        "How long does international shipping take?",
        "Can I pay with cash on delivery?",
        "How do I reset my password?",
    ]

    print("Building pipeline (loading BGE embedder)...")
    pipeline = MiniRAGPipeline()

    print(f"\nRunning {len(test_questions)} test queries...\n")
    results = []
    for q in test_questions:
        out = pipeline.query(q)
        results.append(out)
        print(f"Q: {out['question']}")
        print(f"A: {out['answer']}")
        print(f"Retrieved {len(out['contexts'])} context chunk(s)")
        print("-" * 60)

    print("\nScoring with RAGAS (faithfulness + answer_relevancy)...\n")
    eval_result = score_with_ragas(results)

    df = eval_result
    

    print("\n=== Aggregate Scores ===")
    print(f"Mean Faithfulness:     {df['faithfulness'].mean():.4f}")
    print(f"Mean Answer Relevancy: {df['answer_relevancy'].mean():.4f}")