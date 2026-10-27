
"""
Evaluate the basic_RAG employee-handbook application with RAGAS.

Install dependencies:
    pip install "ragas==0.2.15" "datasets>=2.14" \
        "langchain-core" "openai" "sentence-transformers" "python-dotenv"

Run from the repository root:
    python evaluate_rag.py

Uses:
- Existing document retrieval from step3_search.py (search + reranker)
- Existing answer generation from step4_answer.py
- Existing embedding model from helpers.py
- LM Studio (or another OpenAI-compatible endpoint) for evaluation

Run with --no-rerank to evaluate search-only retrieval for comparison.
"""

import json
import os
import sys
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from helpers import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    LLM_EVAL_BASE_URL,
    LLM_EVAL_MODEL,
    LLM_MAX_TOKENS,
    RERANKER_MODEL,
    get_embedder,
    get_llm_eval,
)
from step3_search import TOP_K, load_store, search
from step4_answer import SYSTEM_PROMPT, build_prompt


load_dotenv()

OUTPUT_PATH = Path("ragas_results.json")

# Retrieval uses the same two passes as the app: similarity search, then the
# reranker. Run with --no-rerank to score the search-only version instead, so
# the two can be compared on the same questions.
USE_RERANKER = "--no-rerank" not in sys.argv


# Add more test questions here as needed.
# Keep reference answers consistent with the actual employee handbook.
EVAL_CASES = [
    {
        "question": "What should an employee do if they notice an error in their recorded work hours?",
        "reference": (
            "The employee must inform their supervisor immediately "
            "so the error or omission can be corrected."
        ),
    },
    {
        "question": "Who is authorized to provide employment references?",
        "reference": (
            "All reference requests must be directed to the designated "
            "person identified as [TITLE]. No other manager, team leader, "
            "or employee is authorized to provide references."
        ),
    },
    {
        "question": "When must an employee receive overtime approval?",
        "reference": (
            "A supervisor must authorize overtime in writing before "
            "it is worked, except in an emergency."
        ),
    },
    {
        "question": "Can unused vacation time carry over to the next benefit year?",
        "reference": (
            "Unused accrued vacation can carry forward to the next benefit "
            "year, but total accrued vacation cannot exceed twice the "
            "annual vacation allotment."
        ),
    },
    {
        "question": "What is the capital of Japan?",
        "reference": (
            "The employee handbook does not provide information "
            "about the capital of Japan."
        ),
    },
]


class ExistingEmbedder(Embeddings):
    """Adapt the existing SentenceTransformer model for RAGAS."""

    def __init__(self):
        self.model = get_embedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=len(texts) > 1,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
        )
        return vector.tolist()


def generate_answer(question, hits):
    """Generate an answer using the existing answer-generation logic."""
    prompt = build_prompt(question, hits)

    response = get_llm_eval().chat.completions.create(
        model=LLM_EVAL_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=LLM_MAX_TOKENS,
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError(
            f"The model returned an empty answer for {question!r}. "
            "Try increasing LLM_MAX_TOKENS."
        )

    return answer.strip()


def main():
    # LM Studio accepts a placeholder API key.
    # The OpenAI-compatible client requires a non-empty key.
    if not OPENAI_API_KEY or OPENAI_API_KEY.lower() == "none":
        os.environ["OPENAI_API_KEY"] = "lm-studio"

    store_path = Path("vector_store.json")

    if not store_path.exists():
        raise SystemExit(
            "vector_store.json was not found. Run these first:\n"
            "  python step1_extract_and_chunk.py\n"
            "  python step2_embed.py"
        )

    # Use the configured evaluation LLM for RAGAS judge calls.
    evaluator_llm = llm_factory(
        model=LLM_EVAL_MODEL,
        base_url=LLM_EVAL_BASE_URL,
    )

    # Reuse the same cached embedding model as the RAG pipeline.
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        ExistingEmbedder()
    )

    store = load_store()

    questions = []
    answers = []
    contexts = []
    references = []

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Reranker: {RERANKER_MODEL if USE_RERANKER else 'off'}")
    print(f"Evaluation model: {LLM_EVAL_MODEL}")
    print(f"Evaluation endpoint: {LLM_EVAL_BASE_URL}")
    print(f"Evaluating {len(EVAL_CASES)} questions...\n")

    for index, case in enumerate(EVAL_CASES, start=1):
        question = case["question"]

        # Retrieve passages using the existing RAG pipeline.
        hits = search(question, store, top_k=TOP_K, use_reranker=USE_RERANKER)

        retrieved_contexts = [
            entry["text"] for entry, _score in hits
        ]

        # Generate an answer using the existing prompt and LLM.
        answer = generate_answer(question, hits)

        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved_contexts)
        references.append(case["reference"])

        pages = ", ".join(
            str(entry["metadata"]["page"])
            for entry, _score in hits
        )

        print(f"{index}. {question}")
        print(f"   Retrieved pages: {pages}")
        print(f"   Answer: {answer[:220].replace(chr(10), ' ')}")
        print()

    # Build the dataset expected by RAGAS.
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "reference": references,
        }
    )

    print(
        "Running RAGAS metrics. "
        "This makes additional calls to the evaluation LLM...\n"
    )

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    print("\nRAGAS scores:")
    print(result)

    # Save per-question scores to JSON.
    try:
        result.to_pandas().to_json(
            OUTPUT_PATH,
            orient="records",
            indent=2,
        )
        print(f"\nSaved per-question scores to {OUTPUT_PATH}")

    except Exception as error:
        print(f"\nCould not save per-question scores: {error}")
        with OUTPUT_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                {"summary": str(result)},
                file,
                indent=2,
            )
        print(f"Saved summary to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
