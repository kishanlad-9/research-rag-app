# 📄 Research RAG Assistant

Ask questions about research papers using AI — powered by your own API key.

## Features
- Upload any research paper PDF
- Choose AI provider: Google Gemini, OpenAI, or Anthropic Claude
- Bring your own API key — nothing is stored
- Semantic search using FAISS + sentence-transformers
- Anti-hallucination RAG prompt

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Deploy on Streamlit Cloud

See deployment guide in the project documentation.

## How to use
1. Enter your API key in the sidebar
2. Upload a PDF and click Process PDF
3. Ask questions about the paper
