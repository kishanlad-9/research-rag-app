"""
vector_store.py
---------------
FAISS vector store — build, save, load, search.
"""

import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import FAISS_INDEX_PATH, TOP_K_RESULTS
from embeddings import get_embedding_model
from utils import get_logger

logger = get_logger(__name__)


def build_vector_store(documents: list[Document]) -> FAISS:
    if not documents:
        raise ValueError("Cannot build vector store from empty document list.")
    logger.info("Building FAISS index from %d documents…", len(documents))
    store = FAISS.from_documents(documents, get_embedding_model())
    logger.info("FAISS index built.")
    return store


def save_vector_store(store: FAISS, path: str = FAISS_INDEX_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    store.save_local(path)
    logger.info("FAISS index saved to '%s'.", path)


def load_vector_store(path: str = FAISS_INDEX_PATH) -> FAISS:
    if not Path(path).exists():
        raise FileNotFoundError("No FAISS index found. Please upload and process a PDF first.")
    store = FAISS.load_local(path, get_embedding_model(), allow_dangerous_deserialization=True)
    logger.info("FAISS index loaded.")
    return store


def vector_store_exists(path: str = FAISS_INDEX_PATH) -> bool:
    return Path(path).exists() and any(Path(path).iterdir())


def delete_vector_store(path: str = FAISS_INDEX_PATH) -> None:
    if Path(path).exists():
        shutil.rmtree(path)
        logger.info("FAISS index deleted.")


def similarity_search(store: FAISS, query: str, top_k: int = TOP_K_RESULTS) -> list[tuple[Document, float]]:
    results = store.similarity_search_with_score(query, k=top_k)
    return [(doc, float(1 / (1 + score))) for doc, score in results]
