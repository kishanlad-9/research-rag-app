"""
rag_pipeline.py
---------------
Orchestrates the full RAG workflow:
PDF → Chunks → Embeddings → FAISS → Retrieval → Prompt → LLM → Answer
"""

from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document

from config import TOP_K_RESULTS
from llm_handler import generate_answer, get_llm
from pdf_processor import process_pdf
from prompt_template import build_rag_prompt
from utils import get_logger
from vector_store import (
    build_vector_store,
    delete_vector_store,
    load_vector_store,
    save_vector_store,
    similarity_search,
    vector_store_exists,
)

logger = get_logger(__name__)


@dataclass
class IngestResult:
    success:     bool
    message:     str
    page_count:  int = 0
    chunk_count: int = 0


@dataclass
class QueryResult:
    success:        bool
    answer:         str
    retrieved_docs: list[tuple[Document, float]] = field(default_factory=list)
    error:          Optional[str] = None


# ── Ingest Pipeline ───────────────────────────────────────────────────────────

def ingest_pdf(pdf_path: str, force_rebuild: bool = False) -> IngestResult:
    """Extract PDF → chunk → embed → save to FAISS."""
    if force_rebuild:
        delete_vector_store()

    try:
        chunks, page_count, _ = process_pdf(pdf_path)
        store = build_vector_store(chunks)
        save_vector_store(store)

        return IngestResult(
            success=True,
            message=f"✅ Processed **{page_count}** pages → **{len(chunks)}** chunks → stored in FAISS.",
            page_count=page_count,
            chunk_count=len(chunks),
        )
    except (FileNotFoundError, ValueError) as exc:
        return IngestResult(success=False, message=f"❌ {exc}")
    except Exception as exc:
        logger.exception("Ingest failed: %s", exc)
        return IngestResult(success=False, message=f"❌ Unexpected error: {exc}")


# ── Query Pipeline ────────────────────────────────────────────────────────────

def query_rag(
    question:   str,
    provider:   str,
    model_name: str,
    api_key:    str,
    top_k:      int = TOP_K_RESULTS,
) -> QueryResult:
    """Search FAISS → build prompt → call LLM → return answer."""

    if not question.strip():
        return QueryResult(success=False, answer="", error="Question cannot be empty.")

    if not vector_store_exists():
        return QueryResult(
            success=False, answer="",
            error="No document processed yet. Upload a PDF and click Process PDF first.",
        )

    if not api_key or not api_key.strip():
        return QueryResult(
            success=False, answer="",
            error="API key is missing. Please enter your API key in the sidebar.",
        )

    try:
        store     = load_vector_store()
        retrieved = similarity_search(store, question, top_k=top_k)

        if not retrieved:
            return QueryResult(
                success=False, answer="",
                error="No relevant content found in the document for your question.",
            )

        prompt = build_rag_prompt(question, retrieved)
        llm    = get_llm(provider=provider, model_name=model_name, api_key=api_key)
        answer = generate_answer(llm, prompt)

        return QueryResult(success=True, answer=answer, retrieved_docs=retrieved)

    except (ValueError, RuntimeError) as exc:
        return QueryResult(success=False, answer="", error=str(exc))
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        return QueryResult(success=False, answer="", error=f"Unexpected error: {exc}")
