"""
embeddings.py
-------------
Local HuggingFace embedding model — no API calls needed.
"""

from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME
from utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model: %s", model_name)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": _select_device()},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
        logger.info("Embedding model loaded.")
        return embeddings
    except Exception as exc:
        raise RuntimeError(f"Failed to load embedding model: {exc}") from exc


def _select_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
