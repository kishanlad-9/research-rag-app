"""
prompt_template.py
------------------
Anti-hallucination RAG prompt construction.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

RAG_TEMPLATE = """\
You are an expert AI research assistant. Answer questions ONLY using the \
context extracted from the research paper below.

STRICT RULES:
1. Use ONLY information present in the Context.
2. Do NOT use your own training knowledge.
3. If the answer is not in the context, say exactly:
   "The document does not contain enough information to answer this question."
4. Be concise and well-structured.

Context:
{context}

Question: {question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_TEMPLATE,
)


def format_context(retrieved_docs: list[tuple[Document, float]]) -> str:
    sections = []
    for rank, (doc, score) in enumerate(retrieved_docs, start=1):
        source  = doc.metadata.get("source", "unknown")
        chunk_i = doc.metadata.get("chunk_index", "?")
        header  = f"[Excerpt {rank} | Source: {source} | Chunk: {chunk_i} | Relevance: {score:.2%}]"
        sections.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(sections)


def build_rag_prompt(question: str, retrieved_docs: list[tuple[Document, float]]) -> str:
    context = format_context(retrieved_docs)
    return RAG_PROMPT.format(context=context, question=question)
