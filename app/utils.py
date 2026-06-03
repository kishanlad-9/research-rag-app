"""
utils.py
--------
Shared helper utilities.
"""

import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def validate_pdf_file(file) -> tuple[bool, str]:
    if file is None:
        return False, "No file provided."
    ext = Path(file.name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type '{ext}'. Only PDF files are accepted."
    size_mb = file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB."
    header = file.read(4)
    file.seek(0)
    if header != b"%PDF":
        return False, "File does not appear to be a valid PDF."
    return True, "File is valid."


def save_uploaded_file(uploaded_file) -> Optional[str]:
    try:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="rag_") as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name
    except Exception:
        return None


def clean_text(raw: str) -> str:
    raw = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    lines = [line.rstrip() for line in raw.splitlines()]
    return "\n".join(lines).strip()


def format_file_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def truncate_text(text: str, max_chars: int = 400) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"
