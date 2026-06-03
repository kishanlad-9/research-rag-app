"""
pdf_processor.py
----------------
PDF extraction and chunking.
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE
from utils import clean_text, get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(pdf_path: str) -> tuple[str, int]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ValueError(f"Could not open PDF — it may be corrupted: {exc}") from exc

    page_count = len(reader.pages)
    if page_count == 0:
        raise ValueError("PDF has no pages.")

    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            pages_text.append(text)
        except Exception as exc:
            logger.warning("Skipping page %d: %s", i + 1, exc)

    full_text = clean_text("\n\n".join(pages_text))

    if not full_text.strip():
        raise ValueError(
            "No readable text found. The PDF may be scanned/image-based."
        )

    logger.info("Extracted %d chars from %d pages.", len(full_text), page_count)
    return full_text, page_count


def chunk_text(text: str, source_name: str = "document") -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": source_name,
                "chunk_index": idx,
                "chunk_total": len(raw_chunks),
            },
        )
        for idx, chunk in enumerate(raw_chunks)
    ]
    logger.info("Created %d chunks from '%s'.", len(documents), source_name)
    return documents


def process_pdf(pdf_path: str) -> tuple[list[Document], int, int]:
    text, page_count = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, source_name=Path(pdf_path).name)
    return chunks, page_count, len(text)
