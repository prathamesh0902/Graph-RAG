"""Streamlit UI for a multi-transcript Podcast RAG application.
Run: streamlit run app.py
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Podcast RAG", page_icon="🎙️", layout="wide")

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.main-header {font-size:2rem;font-weight:700;margin-bottom:.2rem}
.subtitle {color:#777;font-size:.95rem;margin-bottom:1.2rem}
.chunk-box {background:#f8f9fa;border-left:3px solid #667eea;padding:.75rem 1rem;border-radius:0 8px 8px 0;font-size:.85rem;color:#333;margin-bottom:.5rem;white-space:pre-wrap}
.source-tag {display:inline-block;background:#667eea;color:white;border-radius:8px;padding:2px 8px;font-size:.75rem;margin:2px}
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ────────────────────────────────────────────────────────


def init_state():
    defaults = {
        "messages": [], "rag": None, "ingested": False,
        "ingest_stats_list": [], "active_sources": [],
        "provider": "groq", "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        # Unique per Streamlit session; persists across reruns in that session.
        "persist_dir": str(Path(os.getenv("RAG_BASE_DIR", "./data/podcast_rag")) / uuid.uuid4().hex),
        "processed_files": [], "rag_config": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def current_api_key() -> str:
    return st.session_state.groq_api_key if st.session_state.provider == "groq" else st.session_state.openai_api_key

# ─── LLM Call ─────────────────────────────────────────────────────────────────


def call_llm(messages: list, context: str, provider: str, api_key: str, active_sources: list) -> str:
    if not context or context.strip() == "No relevant excerpts found.":
        return "I couldn't find relevant information in the indexed transcripts."
    system_prompt = f"""You are a podcast transcript assistant. Answer ONLY from the transcript excerpts supplied below.
Available transcript sources: {', '.join(active_sources) if active_sources else 'unknown'}.
Rules:
1. Do not use outside knowledge or make unsupported assumptions.
2. If the excerpts do not answer the question, say: 'This topic is not covered in the available transcripts.'
3. Cite the source filename and timestamp for every substantive claim.
4. Distinguish what a speaker explicitly said from any summary or comparison.
5. If sources disagree, describe the disagreement rather than resolving it yourself.
6. Keep the answer clear and concise.

TRANSCRIPT EXCERPTS:
{context}"""
    chat_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[{"role": "system", "content": system_prompt}] + chat_messages,
            temperature=0.2, max_tokens=1200,
        )
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system_prompt}] + chat_messages,
            temperature=0.2, max_tokens=1200,
        )
    return response.choices[0].message.content or "The model returned an empty response."


def get_rag(chunk_size: int, overlap: int, top_k: int, top_n: int):
    from rag_pipeline import PodcastRAG
    config = (chunk_size, overlap, top_k, top_n)
    if st.session_state.rag is None or st.session_state.rag_config != config:
        st.session_state.rag = PodcastRAG(
            persist_dir=st.session_state.persist_dir,
            chunk_size=chunk_size, overlap=overlap,
            top_k_retrieve=top_k, top_n_rerank=top_n,
        )
        st.session_state.rag.load_state()
        st.session_state.rag_config = config
        st.session_state.active_sources = [
            value.get("source", "unknown")
            for value in st.session_state.rag._ingested_sources.values()
        ]
        st.session_state.ingested = bool(st.session_state.active_sources)
    return st.session_state.rag


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")
    st.session_state.provider = st.selectbox(
        "LLM Provider", ["groq", "openai"],
        index=0 if st.session_state.provider == "groq" else 1,
    )
    if st.session_state.provider == "groq":
        st.session_state.groq_api_key = st.text_input(
            "Groq API Key", value=st.session_state.groq_api_key, type="password",
            help="Can also be supplied through GROQ_API_KEY in your .env file.",
        )
    else:
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key", value=st.session_state.openai_api_key, type="password",
            help="Can also be supplied through OPENAI_API_KEY in your .env file.",
        )
    st.divider()
    with st.expander("RAG settings", expanded=False):
        chunk_size = st.slider("Chunk size (tokens)", 200, 800, 500, 50)
        overlap = st.slider("Overlap (tokens)", 0, 150, 80, 10)
        top_k = st.slider("Retrieve top-K", 5, 30, 20, 5)
        top_n = st.slider("Rerank to top-N", 2, 10, 5, 1)
    if overlap >= chunk_size:
        st.error("Overlap must be smaller than chunk size.")
        st.stop()

    st.divider()
    st.subheader("📂 Upload transcripts")
    uploaded_files = st.file_uploader(
        "Upload .txt or .srt files", type=["txt", "srt"], accept_multiple_files=True,
        help="Upload multiple transcripts and ask questions across all of them. Plain text and SubRip subtitle files are supported.",
    )

    if uploaded_files:
        # Content hashes prevent same-name/different-content files being skipped.
        pending = []
        for uploaded in uploaded_files:
            file_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
            if file_hash not in st.session_state.processed_files:
                pending.append((uploaded, file_hash))
        st.caption(f"{len(pending)} new file(s) detected.")
        if st.button("🚀 Process new transcripts", type="primary", use_container_width=True, disabled=not pending):
            rag = get_rag(chunk_size, overlap, top_k, top_n)
            progress = st.progress(0.0, text="Starting ingestion…")
            successes, failures = [], []
            new_stats = []
            for index, (uploaded, file_hash) in enumerate(pending):
                temp_path = None
                try:
                    content = uploaded.getvalue()
                    if not content:
                        raise ValueError("The uploaded file is empty.")
                    suffix = Path(uploaded.name).suffix.lower()
                    with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, prefix="podcast_", delete=False) as temp:
                        temp.write(content)
                        temp_path = Path(temp.name)
                    # Validate that bytes were actually written before passing the path to RAG.
                    if not temp_path.exists() or temp_path.stat().st_size == 0:
                        raise IOError("Could not write uploaded transcript to temporary storage.")
                    progress.progress(index / len(pending), text=f"Ingesting {uploaded.name}…")
                    stats = rag.ingest(str(temp_path), source_name=Path(uploaded.name).stem)
                    if int(stats.get("chunks", 0)) <= 0:
                        raise ValueError("Ingestion returned zero chunks.")
                    successes.append((uploaded.name, file_hash))
                    new_stats.append(stats)
                    st.write(f"✅ {uploaded.name}: {stats['chunks']} chunks")
                except Exception as exc:
                    failures.append((uploaded.name, str(exc)))
                    st.error(f"❌ {uploaded.name}: {type(exc).__name__}: {exc}")
                    st.exception(exc)
                finally:
                    if temp_path and temp_path.exists():
                        temp_path.unlink(missing_ok=True)
                progress.progress((index + 1) / len(pending), text=f"Processed {index + 1}/{len(pending)}")
            st.session_state.processed_files.extend(file_hash for _, file_hash in successes)
            st.session_state.ingest_stats_list.extend(new_stats)
            st.session_state.active_sources = [
                value.get("source", "unknown") for value in rag._ingested_sources.values()
            ]
            st.session_state.ingested = bool(st.session_state.active_sources)
            if successes:
                st.success(f"Successfully indexed {len(successes)} transcript(s).")
            if failures:
                st.warning(f"{len(failures)} transcript(s) failed. They can be retried.")
            st.rerun()

    if st.session_state.active_sources:
        st.divider()
        st.subheader("📚 Active transcripts")
        for source in st.session_state.active_sources:
            st.markdown(f'<span class="source-tag">🎙 {source}</span>', unsafe_allow_html=True)
    if st.session_state.ingest_stats_list:
        total_chunks = sum(int(s.get("chunks", 0)) for s in st.session_state.ingest_stats_list)
        st.metric("Chunks ingested this session", total_chunks)
    st.divider()
    col_a, col_b = st.columns(2)
    if col_a.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if col_b.button("Reset RAG", use_container_width=True):
        st.session_state.messages = []
        st.session_state.rag = None
        st.session_state.rag_config = None
        st.session_state.ingested = False
        st.session_state.ingest_stats_list = []
        st.session_state.active_sources = []
        st.session_state.processed_files = []
        st.session_state.persist_dir = str(Path(os.getenv("RAG_BASE_DIR", "./data/podcast_rag")) / uuid.uuid4().hex)
        st.rerun()


# ─── Main UI ───────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🎙️ Podcast RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-transcript chat · Dense + BM25 hybrid retrieval · Reciprocal Rank Fusion · Cross-encoder reranking</div>', unsafe_allow_html=True)

# Instantiate/restore RAG on every app run as needed, including after a browser refresh.
rag = get_rag(chunk_size, overlap, top_k, top_n)
if not st.session_state.ingested:
    st.info("Upload one or more TXT/SRT transcripts in the sidebar, then select Process new transcripts.")
else:
    st.markdown(f"**Active transcripts ({len(st.session_state.active_sources)}):** " + " · ".join(f"`{s}`" for s in st.session_state.active_sources))
    st.divider()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"📎 {len(message['sources'])} retrieved excerpts"):
                    for text, meta, score in message["sources"]:
                        st.markdown(
                            f"**{meta.get('source', '?')}** · {meta.get('start_time', 'N/A')} → {meta.get('end_time', 'N/A')} · score {score:.3f}\n\n{text}"
                        )
    # ── Suggestion Buttons ────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "What are the main themes discussed across all episodes?",
            "How do the guests' views on AI compare to each other?",
            "What did the speakers say about self-driving / autonomous vehicles?",
            "What are the key insights about the future of AI?",
        ]
        cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            if cols[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state["_prefill"] = sug
                st.rerun()

    # ── Chat Input (reads suggestion prefill if set) ───────────────────────────
    user_input = st.chat_input("Ask anything across all transcripts...")
    if not user_input and "_prefill" in st.session_state:
        user_input = st.session_state.pop("_prefill")

    if user_input:
        api_key = current_api_key().strip()
        if not api_key:
            st.error(f"Enter your {st.session_state.provider.upper()} API key in the sidebar.")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching transcript excerpts…"):
                    hits = rag.retrieve(user_input)
                    context = rag.format_context(hits)
                with st.spinner("Generating grounded answer…"):
                    answer = call_llm(
                        st.session_state.messages, context,
                        st.session_state.provider, api_key,
                        st.session_state.active_sources,
                    )
                st.markdown(answer)
                if hits:
                    with st.expander(f"📎 {len(hits)} retrieved excerpts"):
                        for text, meta, score in hits:
                            st.markdown(
                                f"**{meta.get('source', '?')}** · {meta.get('start_time', 'N/A')} → {meta.get('end_time', 'N/A')} · score {score:.3f}\n\n{text}"
                            )
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": hits})
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                st.error(error_message)
                st.exception(exc)
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_message}"})
