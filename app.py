"""Module 7: Streamlit interface for the Domain-Specific RAG Chatbot."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Must run before importing auth: its USERS_FILE is bound from the
# AUTH_USERS_FILE env var at import time, so .env has to be loaded first.
load_dotenv()

import auth
from document_loader import DocumentValidationError, MAX_FILE_SIZE_MB
from feedback import log_feedback
from rag_pipeline import RAGPipeline

BASE_VECTOR_STORE_DIR = Path("vector_store/saved_index")

st.set_page_config(page_title="Domain-Specific RAG Chatbot", page_icon="📄", layout="wide")


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

    st.title("Domain-Specific RAG Chatbot")
    st.subheader("Sign in")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
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


def sidebar(pipeline: RAGPipeline, username: str) -> None:
    store_dir = vector_store_dir_for(username)
    with st.sidebar:
        if auth.login_required():
            st.caption(f"Signed in as **{username}**")
            if st.button("Log out", use_container_width=True):
                del st.session_state["authenticated_user"]
                for key in ("pipeline", "pipeline_owner", "messages", "uploaded_names", "document_filter"):
                    st.session_state.pop(key, None)
                st.rerun()
            st.divider()

        st.header("📄 Document Upload")
        st.caption(f"PDF files only, up to {MAX_FILE_SIZE_MB} MB each.")

        uploaded_files = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="uploader",
        )

        col1, col2 = st.columns(2)
        with col1:
            process_clicked = st.button("Process Documents", type="primary", use_container_width=True)
        with col2:
            clear_docs_clicked = st.button("Clear Documents", use_container_width=True)

        if process_clicked:
            if not uploaded_files:
                st.warning("Please select at least one PDF file first.")
            else:
                with st.spinner("Extracting, chunking, and embedding documents..."):
                    files_payload = [
                        (f.name, f.getvalue(), f.size) for f in uploaded_files
                    ]
                    try:
                        num_chunks = pipeline.process_documents(files_payload)
                        pipeline.save(str(store_dir))
                        st.session_state.uploaded_names.extend(
                            f.name for f in uploaded_files
                        )
                        st.success(f"Processed {len(uploaded_files)} file(s) into {num_chunks} chunks.")
                    except DocumentValidationError as exc:
                        st.error(str(exc))

        if clear_docs_clicked:
            pipeline.clear()
            st.session_state.uploaded_names = []
            st.session_state.messages = []
            st.success("Cleared all documents and chat history.")
            st.rerun()

        if st.session_state.uploaded_names:
            st.subheader("Indexed documents")
            for name in sorted(set(st.session_state.uploaded_names)):
                st.markdown(f"- {name}")

        document_names = pipeline.vector_store.document_names
        if len(document_names) > 1:
            st.subheader("Search scope")
            st.session_state.document_filter = st.multiselect(
                "Only answer from:",
                options=document_names,
                default=st.session_state.get("document_filter", document_names),
                help="Restrict retrieval to a subset of the indexed documents.",
            )

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        provider = os.getenv("LLM_PROVIDER", "groq")
        st.caption(f"LLM provider: **{provider}**")
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
    with st.expander(f"📎 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"- **{s.document}**, page {s.page} (similarity: {s.score:.2f})")


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


def main() -> None:
    username = require_login()
    init_state()
    pipeline = get_pipeline(username)
    sidebar(pipeline, username)

    st.title("Domain-Specific RAG Chatbot")
    st.caption(
        "Ask questions about the documents you've uploaded. Answers are grounded only in "
        "their content. Follow-up questions and pasted batches of questions are both supported."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_message(msg)
            else:
                st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your documents...")
    if question:
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


if __name__ == "__main__":
    main()
