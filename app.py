"""Module 7: Streamlit interface for the Domain-Specific RAG Chatbot."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from filelock import Timeout

# Must run before importing auth: its USERS_FILE is bound from the
# AUTH_USERS_FILE env var at import time, so .env has to be loaded first.
load_dotenv()

import auth
import ocr
from document_loader import DocumentValidationError, MAX_FILE_SIZE_MB
from feedback import feedback_summary, log_feedback
from rag_pipeline import RAGPipeline

BASE_VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR") or "vector_store/saved_index")

EXAMPLE_QUESTIONS = [
    "📋 Summarize this document",
    "🔑 What are the key points?",
    "❓ What isn't covered here?",
]

st.set_page_config(page_title="Domain-Specific RAG Chatbot", page_icon="🤖", layout="wide")


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.hero-title {
    font-weight: 700;
    font-size: 2.1rem;
    line-height: 1.2;
    background: linear-gradient(90deg, #6366F1, #8B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.15rem;
}
.hero-subtitle { opacity: 0.72; font-size: 0.95rem; margin-bottom: 0.9rem; }

.badge-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.1rem; }
.badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.28rem 0.75rem; border-radius: 999px;
    border: 1px solid rgba(99, 102, 241, 0.35);
    color: #6366F1; font-size: 0.78rem; font-weight: 600;
    background: rgba(99, 102, 241, 0.07);
}

.source-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.3rem; }
.source-chip {
    display: flex; flex-direction: column; gap: 0.1rem;
    padding: 0.5rem 0.75rem; border-radius: 10px;
    border: 1px solid rgba(99, 102, 241, 0.18);
    background: rgba(99, 102, 241, 0.05);
    font-size: 0.8rem; min-width: 130px;
}
.source-chip .doc { font-weight: 600; }
.source-chip .meta { opacity: 0.68; font-size: 0.75rem; }
.source-chip .ocr-tag { color: #D97706; font-size: 0.7rem; font-weight: 700; }

.empty-state {
    text-align: center; padding: 2.75rem 1.5rem;
    border: 1.5px dashed rgba(99, 102, 241, 0.4);
    border-radius: 18px; background: rgba(99, 102, 241, 0.035);
    margin-top: 0.5rem;
}
.empty-state .icon { font-size: 2.6rem; margin-bottom: 0.6rem; }
.empty-state h3 { margin-bottom: 0.35rem; }
.empty-state p { opacity: 0.72; font-size: 0.92rem; }

.stButton>button { border-radius: 10px; transition: transform 0.06s ease-in-out; }
.stButton>button:hover { transform: translateY(-1px); }

.app-footer {
    margin-top: 2.5rem; padding-top: 1rem;
    border-top: 1px solid rgba(128, 128, 128, 0.18);
    font-size: 0.75rem; opacity: 0.55; text-align: center;
}

.login-card { max-width: 420px; margin: 2.5rem auto 0 auto; text-align: center; }
.login-card .icon { font-size: 2.8rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def require_login() -> str:
    """Gate the app behind a login form if any users are configured (auth.py).

    Returns the logged-in username, or "default" when no users.json exists -
    the app then runs exactly as it did before this feature, single-user,
    no prompt. Calls st.stop() to halt the script if not yet authenticated.
    """
    if not auth.login_required():
        return "default"

    if st.session_state.get("authenticated_user"):
        return st.session_state.authenticated_user

    inject_css()
    st.markdown(
        """
        <div class="login-card">
            <div class="icon">🤖</div>
            <div class="hero-title">Domain-Specific RAG Chatbot</div>
            <div class="hero-subtitle">Sign in to access your documents</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        with st.form("login_form", border=True):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True, type="primary")
        if submitted:
            if auth.authenticate(username, password):
                st.session_state.authenticated_user = username
                st.rerun()
            else:
                st.error("Incorrect username or password.")
    st.stop()


def vector_store_dir_for(username: str) -> Path:
    """Each logged-in user gets an isolated vector store subdirectory, so
    documents uploaded by one user are never visible to another. When login
    isn't configured, everyone shares the original top-level directory for
    backward compatibility with pre-login deployments."""
    if username == "default" and not auth.login_required():
        return BASE_VECTOR_STORE_DIR
    return BASE_VECTOR_STORE_DIR / username


def get_pipeline(username: str) -> RAGPipeline:
    if "pipeline" not in st.session_state or st.session_state.get("pipeline_owner") != username:
        st.session_state.pipeline = RAGPipeline()
        st.session_state.pipeline_owner = username
        store_dir = vector_store_dir_for(username)
        if (store_dir / "index.faiss").exists():
            try:
                st.session_state.pipeline.load(str(store_dir))
            except Exception:
                pass
    return st.session_state.pipeline


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_names" not in st.session_state:
        st.session_state.uploaded_names = []
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = {}
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def sidebar(pipeline: RAGPipeline, username: str) -> None:
    store_dir = vector_store_dir_for(username)
    with st.sidebar:
        if auth.login_required():
            with st.container(border=True):
                st.caption("Signed in as")
                st.markdown(f"**👤 {username}**")
                if st.button("Log out", use_container_width=True):
                    del st.session_state["authenticated_user"]
                    for key in ("pipeline", "pipeline_owner", "messages", "uploaded_names", "document_filter"):
                        st.session_state.pop(key, None)
                    st.rerun()

        with st.container(border=True):
            st.markdown("**📄 Document Upload**")
            st.caption(f"PDF files only, up to {MAX_FILE_SIZE_MB} MB each.")

            uploaded_files = st.file_uploader(
                "Upload one or more PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                key="uploader",
                label_visibility="collapsed",
            )

            enable_ocr = st.checkbox(
                "Enable OCR for scanned pages",
                value=False,
                help=(
                    "Optional extension for image-only/scanned PDF pages. "
                    "Requires the Tesseract OCR engine installed on the host "
                    "(see README) - has no effect otherwise."
                ),
            )
            if enable_ocr and not ocr.is_available():
                st.caption("⚠️ Tesseract isn't installed here, so OCR will be skipped.")

            col1, col2 = st.columns(2)
            with col1:
                process_clicked = st.button(
                    "Process", type="primary", use_container_width=True
                )
            with col2:
                clear_docs_clicked = st.button("Clear all", use_container_width=True)

            if process_clicked:
                if not uploaded_files:
                    st.warning("Please select at least one PDF file first.")
                else:
                    files_payload = [(f.name, f.getvalue(), f.size) for f in uploaded_files]
                    with st.status("Processing documents...", expanded=True) as status:
                        def on_progress(stage: str, _status=status) -> None:
                            _status.write(stage)

                        try:
                            num_chunks = pipeline.process_documents(
                                files_payload, enable_ocr=enable_ocr, on_progress=on_progress
                            )
                            pipeline.save(str(store_dir))
                            st.session_state.uploaded_names.extend(f.name for f in uploaded_files)
                            status.update(
                                label=f"Processed {len(uploaded_files)} file(s) into {num_chunks} chunks.",
                                state="complete",
                            )
                        except DocumentValidationError as exc:
                            status.update(label="Processing failed.", state="error")
                            st.error(str(exc))
                        except Timeout:
                            status.update(label="Processing failed.", state="error")
                            st.error(
                                "The document store is busy (another save is in progress). "
                                "Please try again in a moment."
                            )

            if clear_docs_clicked:
                pipeline.clear()
                st.session_state.uploaded_names = []
                st.session_state.messages = []
                st.toast("Cleared all documents and chat history.", icon="🗑️")
                st.rerun()

            document_names = pipeline.vector_store.document_names
            if document_names:
                st.markdown("**Indexed documents**")
                for name in document_names:
                    st.markdown(f"📎 `{name}`")

        document_names = pipeline.vector_store.document_names
        if len(document_names) > 1:
            with st.container(border=True):
                st.markdown("**🔍 Search scope**")
                if "document_filter" not in st.session_state:
                    st.session_state.document_filter = document_names
                else:
                    # Drop stale selections (e.g. a document removed by
                    # Clear all + reprocessing) so the widget never sees a
                    # value outside its current options.
                    st.session_state.document_filter = [
                        d for d in st.session_state.document_filter if d in document_names
                    ]
                # key="document_filter" binds this widget directly to
                # session_state - reading AND writing it manually as well
                # (the previous code did `st.session_state.document_filter =
                # st.multiselect(..., default=st.session_state.get(...))`)
                # creates a feedback loop where a click briefly gets
                # overwritten by the stale pre-click value, which is what
                # caused the flicker on removing an item.
                st.multiselect(
                    "Only answer from:",
                    options=document_names,
                    key="document_filter",
                    help="Restrict retrieval to a subset of the indexed documents.",
                    label_visibility="collapsed",
                )

        with st.container(border=True):
            st.markdown("**💬 Session**")
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        with st.expander("📊 Feedback insights"):
            summary = feedback_summary()
            if summary["total"] == 0:
                st.caption("No feedback logged yet - 👍/👎 an answer to start building this up.")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("👍 Helpful", summary["up"])
                col2.metric("👎 Not helpful", summary["down"])
                col3.metric("Helpful rate", f"{summary['helpful_rate']:.0%}")

        provider = os.getenv("LLM_PROVIDER", "groq")
        backend = pipeline.vector_store.embedding_backend if not pipeline.vector_store.is_empty else "—"
        st.caption(f"LLM provider: **{provider}** · Embeddings: **{backend}**")
        st.caption(
            "⚠️ Answers are generated from your documents but may still be "
            "incomplete or wrong. Verify high-stakes information yourself."
        )


def _extract_history(messages: list[dict], max_turns: int = 3) -> list[tuple[str, str]]:
    """Pull the last few (question, answer) turns for conversational follow-ups."""
    turns: list[tuple[str, str]] = []
    for i in range(len(messages) - 1):
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            turns.append((messages[i]["content"], messages[i + 1]["content"]))
    return turns[-max_turns:]


def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} source{'s' if len(sources) != 1 else ''}"):
        chips = []
        for s in sources:
            ocr_tag = '<span class="ocr-tag">🔎 via OCR</span>' if getattr(s, "via_ocr", False) else ""
            chips.append(
                f'<div class="source-chip"><span class="doc">{s.document}</span>'
                f'<span class="meta">page {s.page} · similarity {s.score:.2f}</span>{ocr_tag}</div>'
            )
        st.markdown(f'<div class="source-grid">{"".join(chips)}</div>', unsafe_allow_html=True)


def render_feedback(msg: dict) -> None:
    """👍/👎 feedback buttons for one assistant answer, logged via feedback.py."""
    if not msg.get("grounded", True):
        return  # nothing useful to rate on a refusal / upload-prompt message
    msg_id = msg["id"]
    given = st.session_state.feedback_given.get(msg_id)
    if given:
        st.caption("✅ Thanks for the feedback!" if given == "up" else "📝 Thanks - noted as not helpful.")
        return

    col1, col2, _ = st.columns([1, 1, 10])
    source_labels = [f"{s.document} p.{s.page}" for s in msg.get("sources", [])]
    with col1:
        if st.button("👍", key=f"fb_up_{msg_id}"):
            log_feedback(msg.get("question", ""), msg["content"], source_labels, "up")
            st.session_state.feedback_given[msg_id] = "up"
            st.rerun()
    with col2:
        if st.button("👎", key=f"fb_down_{msg_id}"):
            log_feedback(msg.get("question", ""), msg["content"], source_labels, "down")
            st.session_state.feedback_given[msg_id] = "down"
            st.rerun()


def render_assistant_message(msg: dict) -> None:
    st.markdown(msg["content"])
    if msg.get("sources"):
        render_sources(msg["sources"])
    if "response_time" in msg:
        st.caption(f"⏱ {msg['response_time']:.2f}s")
    render_feedback(msg)


def _resolve_document_filter(pipeline: RAGPipeline) -> set[str] | None:
    """None means 'search everything' (the fast path); a set narrows retrieval."""
    all_docs = pipeline.vector_store.document_names
    selected = st.session_state.get("document_filter")
    if selected is None or set(selected) == set(all_docs):
        return None
    return set(selected)


def render_header(pipeline: RAGPipeline) -> None:
    st.markdown('<div class="hero-title">🤖 Domain-Specific RAG Chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Ask questions about your documents - answers are grounded '
        "only in their content, with follow-ups and batched questions both supported.</div>",
        unsafe_allow_html=True,
    )

    provider = os.getenv("LLM_PROVIDER", "groq")
    doc_count = len(pipeline.vector_store.document_names)
    chunk_count = len(pipeline.vector_store.chunks)
    badges = [
        ("🧠", provider.capitalize()),
        ("📚", f"{doc_count} doc{'s' if doc_count != 1 else ''}"),
        ("🧩", f"{chunk_count} chunk{'s' if chunk_count != 1 else ''}"),
        ("🔎", "OCR ready" if ocr.is_available() else "OCR off"),
    ]
    badge_html = "".join(f'<span class="badge">{icon} {label}</span>' for icon, label in badges)
    st.markdown(f'<div class="badge-row">{badge_html}</div>', unsafe_allow_html=True)


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">📄➡️🤖</div>
            <h3>Upload a document to get started</h3>
            <p>Use the sidebar to upload one or more PDFs, then click <b>Process</b>.
            Try the sample "Policy.pdf" and "Handbook.pdf" in the <code>documents/</code> folder.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_example_questions() -> None:
    st.caption("Try asking:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, question in zip(cols, EXAMPLE_QUESTIONS):
        with col:
            if st.button(question, use_container_width=True, key=f"example_{question}"):
                st.session_state.pending_question = question


def render_footer() -> None:
    st.markdown(
        '<div class="app-footer">Domain-Specific RAG Chatbot · '
        "Built with Streamlit, FAISS, and Sentence Transformers</div>",
        unsafe_allow_html=True,
    )


def handle_question(question: str, pipeline: RAGPipeline) -> None:
    history = _extract_history(st.session_state.messages)
    document_filter = _resolve_document_filter(pipeline)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages and generating an answer..."):
            answer = pipeline.ask(question, history=history, document_filter=document_filter)
        msg = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": answer.text,
            "question": question,
            "sources": answer.sources,
            "response_time": answer.response_time_seconds,
            "grounded": answer.grounded,
        }
        st.session_state.messages.append(msg)
        render_assistant_message(msg)


def main() -> None:
    username = require_login()
    inject_css()
    init_state()
    pipeline = get_pipeline(username)
    sidebar(pipeline, username)

    render_header(pipeline)

    # st.chat_input always renders pinned to the bottom of the page regardless
    # of call order, so it's safe to resolve the pending question first and
    # let it inform whether the example-question row below should still show.
    pending = st.session_state.pop("pending_question", None)
    question = st.chat_input("Ask a question about your documents...") or pending

    if pipeline.vector_store.is_empty:
        render_empty_state()
    elif not st.session_state.messages and not question:
        render_example_questions()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_message(msg)
            else:
                st.markdown(msg["content"])

    if question:
        handle_question(question, pipeline)

    render_footer()


if __name__ == "__main__":
    main()
