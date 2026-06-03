# 📄 Research RAG Assistant

A Retrieval-Augmented Generation (RAG) application that lets you ask questions about research papers using AI.

## 🌟 Features

- Upload any research paper PDF
- Choose AI provider: Google Gemini, OpenAI, or Anthropic Claude
- Bring your own API key — nothing is stored on the server
- Semantic search using FAISS + sentence-transformers
- Anti-hallucination prompt — answers only from the document
- Chat history with source context display
- Export chat as Markdown

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| RAG Framework | LangChain |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| LLM Providers | Google Gemini, OpenAI, Anthropic |
| PDF Parsing | pypdf |
| Language | Python 3.11+ |

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## 🔑 How to Use

1. Open the app
2. Select your AI provider (Google Gemini / OpenAI / Anthropic)
3. Enter your API key in the sidebar
4. Upload a research paper PDF
5. Click Process PDF
6. Ask questions about the paper

## 🔗 Get API Keys

- Google Gemini: https://aistudio.google.com/app/apikey
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/settings/keys

## 📝 License

MIT License
