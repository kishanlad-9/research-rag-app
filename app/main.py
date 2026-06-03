"""
main.py
-------
Research RAG Assistant — API key based, cloud-ready.
Run locally:  streamlit run app/main.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from config import API_PROVIDERS, APP_SUBTITLE, APP_TITLE
from rag_pipeline import IngestResult, QueryResult, ingest_pdf, query_rag
from utils import format_file_size, save_uploaded_file, truncate_text, validate_pdf_file
from vector_store import delete_vector_store, vector_store_exists

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0e1117; --surface: #161b27; --border: #252d3d;
    --accent: #4f7aff; --accent2: #7c5cfc;
    --text: #dce3f0; --muted: #6b7a99;
    --success: #3ddc84; --error: #ff5c6a;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
    background: var(--bg);
}

.rag-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.rag-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    font-weight: 400;
    margin: 0;
    background: linear-gradient(135deg, #4f7aff 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.rag-header p { color: var(--muted); font-size: 1rem; margin: .4rem 0 0; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

.sidebar-label {
    font-size: .7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    font-weight: 600;
    margin-bottom: .4rem;
}

.status-ok {
    display: inline-flex; align-items: center; gap: .4rem;
    padding: .25rem .75rem; border-radius: 999px; font-size: .78rem;
    background: rgba(61,220,132,.15); color: var(--success);
    border: 1px solid rgba(61,220,132,.3);
}
.status-warn {
    display: inline-flex; align-items: center; gap: .4rem;
    padding: .25rem .75rem; border-radius: 999px; font-size: .78rem;
    background: rgba(255,92,106,.15); color: var(--error);
    border: 1px solid rgba(255,92,106,.3);
}
.status-info {
    display: inline-flex; align-items: center; gap: .4rem;
    padding: .25rem .75rem; border-radius: 999px; font-size: .78rem;
    background: rgba(79,122,255,.15); color: var(--accent);
    border: 1px solid rgba(79,122,255,.3);
}

.chat-turn { margin-bottom: 1.5rem; }
.chat-label { font-size: .7rem; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); font-weight: 600; margin-bottom: .4rem; }
.chat-user {
    background: #1c2340;
    border: 1px solid rgba(79,122,255,.25);
    border-radius: 12px 12px 4px 12px;
    padding: .9rem 1.1rem;
    max-width: 85%;
    margin-left: auto;
    font-size: .95rem;
}
.chat-ai {
    background: #111827;
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 4px;
    padding: .9rem 1.1rem;
    max-width: 90%;
    font-size: .95rem;
    line-height: 1.7;
}

.chunk-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 8px;
    padding: .8rem 1rem;
    margin-bottom: .6rem;
    font-size: .83rem;
    font-family: 'JetBrains Mono', monospace;
    color: #a0aec0;
    line-height: 1.5;
}
.chunk-meta { font-size: .72rem; color: var(--muted); margin-bottom: .4rem; font-family: 'DM Sans', sans-serif; }

.stat-row { display: flex; gap: .7rem; margin: .8rem 0; }
.stat-card {
    flex: 1; background: rgba(79,122,255,.08);
    border: 1px solid rgba(79,122,255,.2);
    border-radius: 8px; padding: .6rem .8rem; text-align: center;
}
.stat-num { font-size: 1.4rem; font-weight: 600; color: var(--accent); }
.stat-lbl { font-size: .68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }

.doc-banner {
    background: rgba(79,122,255,.08);
    border: 1px solid rgba(79,122,255,.25);
    border-radius: 8px;
    padding: .6rem 1rem;
    font-size: .85rem;
    color: #93b4ff;
    margin-bottom: 1.5rem;
}

.stButton > button { font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; border-radius: 8px !important; }
hr { border-color: var(--border) !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "chat_history":        [],
        "ingest_result":       None,
        "selected_provider":   "Google Gemini",
        "selected_model":      "gemini-2.5-flash-preview-05-20",
        "api_key":             "",
        "pdf_name":            None,
        "show_context":        True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="rag-header">
    <h1>📄 Research RAG Assistant</h1>
    <p>{APP_SUBTITLE}</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # ── Provider ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">AI Provider</div>', unsafe_allow_html=True)
    provider = st.selectbox(
        "Provider",
        options=list(API_PROVIDERS.keys()),
        index=list(API_PROVIDERS.keys()).index(st.session_state.selected_provider),
        label_visibility="collapsed",
    )
    if provider != st.session_state.selected_provider:
        st.session_state.selected_provider = provider
        st.session_state.selected_model = API_PROVIDERS[provider]["default"]
        st.rerun()

    provider_cfg = API_PROVIDERS[provider]

    # ── Model ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label" style="margin-top:.8rem">Model</div>', unsafe_allow_html=True)
    model = st.selectbox(
        "Model",
        options=provider_cfg["models"],
        index=provider_cfg["models"].index(st.session_state.selected_model)
              if st.session_state.selected_model in provider_cfg["models"] else 0,
        label_visibility="collapsed",
    )
    st.session_state.selected_model = model

    # ── API Key ───────────────────────────────────────────────────────────────
    st.markdown(f'<div class="sidebar-label" style="margin-top:.8rem">{provider_cfg["key_label"]}</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder=provider_cfg["key_prefix"],
        label_visibility="collapsed",
        help="Your key stays in this session only — never stored or sent anywhere except the AI provider.",
    )
    st.session_state.api_key = api_key

    if api_key.strip():
        st.markdown('<span class="status-ok">✓ API Key entered</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">✕ API Key required</span>', unsafe_allow_html=True)

    st.caption(f"[Get {provider} API key ↗]({provider_cfg['get_key_url']})")

    st.markdown("---")

    # ── PDF Upload ────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">Upload Research Paper</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop PDF here",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.caption(f"📎 **{uploaded_file.name}**  \nSize: {format_file_size(uploaded_file.size)}")

        if st.button("⚡ Process PDF", type="primary", use_container_width=True):
            is_valid, msg = validate_pdf_file(uploaded_file)
            if not is_valid:
                st.error(msg)
            else:
                with st.spinner("Reading and indexing PDF…"):
                    tmp_path = save_uploaded_file(uploaded_file)
                    if tmp_path:
                        result: IngestResult = ingest_pdf(tmp_path, force_rebuild=True)
                        st.session_state.ingest_result = result
                        if result.success:
                            st.session_state.pdf_name = uploaded_file.name
                            st.session_state.chat_history = []
                        try:
                            Path(tmp_path).unlink(missing_ok=True)
                        except Exception:
                            pass
                    else:
                        st.error("Failed to save file. Please try again.")

    # Ingest result display
    if st.session_state.ingest_result:
        res = st.session_state.ingest_result
        if res.success:
            st.success(res.message)
            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-card">
                    <div class="stat-num">{res.page_count}</div>
                    <div class="stat-lbl">Pages</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{res.chunk_count}</div>
                    <div class="stat-lbl">Chunks</div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.error(res.message)
    elif vector_store_exists():
        st.markdown('<span class="status-info">● Vector Store Ready</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Options ───────────────────────────────────────────────────────────────
    st.session_state.show_context = st.toggle(
        "Show retrieved context", value=st.session_state.show_context
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear DB", use_container_width=True):
            delete_vector_store()
            st.session_state.ingest_result = None
            st.session_state.pdf_name = None
            st.session_state.chat_history = []
            st.success("Cleared.")
    with col2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("""
    <hr>
    <div style='font-size:.7rem;color:#4a5568;text-align:center;line-height:1.8'>
        Research RAG Assistant<br>
        LangChain · FAISS · HuggingFace<br>
        🔑 Bring Your Own API Key
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════

# Active document banner
if st.session_state.pdf_name:
    st.markdown(f"""
    <div class="doc-banner">
        📄 <strong>{st.session_state.pdf_name}</strong>
        &nbsp;·&nbsp;
        {st.session_state.selected_provider} · <strong>{st.session_state.selected_model}</strong>
    </div>""", unsafe_allow_html=True)

# Chat history
for turn in st.session_state.chat_history:
    if turn["role"] == "user":
        st.markdown(f"""
        <div class="chat-turn">
            <div class="chat-label">You</div>
            <div class="chat-user">{turn["content"]}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-turn">
            <div class="chat-label">🤖 Assistant</div>
            <div class="chat-ai">{turn["content"]}</div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.show_context and turn.get("docs"):
            with st.expander(f"📚 Retrieved Context ({len(turn['docs'])} chunks)", expanded=False):
                for i, (doc, score) in enumerate(turn["docs"], 1):
                    st.markdown(f"""
                    <div class="chunk-card">
                        <div class="chunk-meta">
                            Chunk {i} &nbsp;·&nbsp;
                            {doc.metadata.get('source','?')} &nbsp;·&nbsp;
                            Relevance: {score:.1%}
                        </div>
                        {truncate_text(doc.page_content, 400)}
                    </div>""", unsafe_allow_html=True)

# Empty state
if not st.session_state.chat_history:
    if vector_store_exists():
        st.markdown("""
        <div style="text-align:center;padding:3rem 0;">
            <div style="font-size:3rem">💬</div>
            <div style="font-size:1.1rem;color:#6b7a99;margin:.5rem 0">Document ready — ask your first question below</div>
            <div style="font-size:.85rem;color:#4a5568">
                Try: "What is the main contribution of this paper?"
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;">
            <div style="font-size:4rem">📄</div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:#6b7a99;margin:.5rem 0">
                Upload a research paper to get started
            </div>
            <div style="font-size:.9rem;color:#4a5568;max-width:440px;margin:.5rem auto;line-height:1.7">
                1. Enter your API key in the sidebar<br>
                2. Upload a PDF and click ⚡ Process PDF<br>
                3. Ask questions about the paper
            </div>
        </div>""", unsafe_allow_html=True)

# ── Question Input ────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
col_q, col_btn = st.columns([5, 1])

with col_q:
    question = st.text_input(
        "Question",
        placeholder="e.g. What methodology did the authors use?",
        label_visibility="collapsed",
    )
with col_btn:
    ask = st.button("Ask ➤", type="primary", use_container_width=True)

# ── Handle Question ───────────────────────────────────────────────────────────
if ask and question.strip():
    if not st.session_state.api_key.strip():
        st.error("Please enter your API key in the sidebar first.", icon="🔑")
    elif not vector_store_exists():
        st.error("Please upload and process a PDF first.", icon="📄")
    else:
        st.session_state.chat_history.append({
            "role": "user", "content": question, "docs": []
        })

        with st.spinner(f"🔍 Searching · 🤖 Generating with {st.session_state.selected_model}…"):
            result: QueryResult = query_rag(
                question=question,
                provider=st.session_state.selected_provider,
                model_name=st.session_state.selected_model,
                api_key=st.session_state.api_key,
            )

        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": result.answer if result.success else f"⚠️ {result.error}",
            "docs":    result.retrieved_docs if result.success else [],
        })
        st.rerun()

# ── Export ────────────────────────────────────────────────────────────────────
if st.session_state.chat_history:
    lines = [f"# Research RAG — Chat Export\n\nDocument: {st.session_state.pdf_name or 'Unknown'}\n"]
    for t in st.session_state.chat_history:
        label = "Q" if t["role"] == "user" else "A"
        lines.append(f"\n**{label}:** {t['content']}\n")
    st.download_button(
        label="⬇️ Export Chat",
        data="\n".join(lines),
        file_name="rag_chat_export.md",
        mime="text/markdown",
    )
