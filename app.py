"""Module 7: Streamlit interface for the Domain-Specific RAG Chatbot."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from document_loader import DocumentValidationError, MAX_FILE_SIZE_MB
from rag_pipeline import RAGPipeline

load_dotenv()

VECTOR_STORE_DIR = Path("vector_store/saved_index")

st.set_page_config(page_title="Domain-Specific RAG Chatbot", page_icon="📄", layout="wide")


def get_pipeline() -> RAGPipeline:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
        if VECTOR_STORE_DIR.exists() and (VECTOR_STORE_DIR / "index.faiss").exists():
            try:
                st.session_state.pipeline.load(str(VECTOR_STORE_DIR))
            except Exception:
                pass
    return st.session_state.pipeline


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_names" not in st.session_state:
        st.session_state.uploaded_names = []


def sidebar(pipeline: RAGPipeline) -> None:
    with st.sidebar:
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
                        pipeline.save(str(VECTOR_STORE_DIR))
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


def main() -> None:
    init_state()
    pipeline = get_pipeline()
    sidebar(pipeline)

    st.title("Domain-Specific RAG Chatbot")
    st.caption(
        "Ask questions about the documents you've uploaded. Answers are grounded only in "
        "their content. Follow-up questions and pasted batches of questions are both supported."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])
            if msg["role"] == "assistant" and "response_time" in msg:
                st.caption(f"⏱ {msg['response_time']:.2f}s")

    question = st.chat_input("Ask a question about your documents...")
    if question:
        history = _extract_history(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant passages and generating an answer..."):
                answer = pipeline.ask(question, history=history)
            st.markdown(answer.text)
            render_sources(answer.sources)
            st.caption(f"⏱ {answer.response_time_seconds:.2f}s")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer.text,
                "sources": answer.sources,
                "response_time": answer.response_time_seconds,
            }
        )


if __name__ == "__main__":
    main()
