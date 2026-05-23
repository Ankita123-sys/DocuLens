from __future__ import annotations

import streamlit as st

from src.chat import answer_question
from src.config import get_settings
from src.document_processor import chunk_settings_for_file_size, process_upload
from src.launchdarkly_client import is_rag_enabled
from src.nvidia_client import NvidiaClient
from src.vector_store import SessionVectorStore

# Bump when ingestion/API changes so Streamlit reloads cached clients.
APP_CODE_VERSION = "3"


def init_session() -> None:
    if st.session_state.get("app_code_version") != APP_CODE_VERSION:
        for key in (
            "vector_store",
            "nvidia",
            "settings",
            "messages",
            "last_uploaded_source",
        ):
            st.session_state.pop(key, None)
        st.session_state.app_code_version = APP_CODE_VERSION

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        settings = get_settings()
        nvidia = NvidiaClient(settings)
        st.session_state.vector_store = SessionVectorStore(nvidia)
        st.session_state.nvidia = nvidia
        st.session_state.settings = settings
    if "last_uploaded_source" not in st.session_state:
        st.session_state.last_uploaded_source = None


def render_sidebar(rag_enabled: bool) -> None:
    st.sidebar.header("Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        store: SessionVectorStore = st.session_state.vector_store
        settings = st.session_state.settings
        for uploaded in uploaded_files:
            cache_key = f"ingested::{uploaded.name}::{uploaded.size}"
            if cache_key in st.session_state:
                continue
            try:
                size_mb = uploaded.size / (1024 * 1024)
                with st.sidebar.status(
                    f"Indexing {uploaded.name} ({size_mb:.1f} MB)...",
                    expanded=True,
                ) as status:
                    status.write("Reading file...")
                    file_bytes = uploaded.getvalue()

                    chunk_size, chunk_overlap = chunk_settings_for_file_size(
                        len(file_bytes),
                        settings.chunk_size,
                        settings.chunk_overlap,
                    )
                    status.write(
                        f"Extracting text (chunk size {chunk_size} chars)..."
                    )
                    chunks = process_upload(
                        uploaded.name,
                        file_bytes,
                        chunk_size,
                        chunk_overlap,
                    )
                    api_calls = (
                        (len(chunks) + settings.embed_batch_size - 1)
                        // settings.embed_batch_size
                    )
                    status.write(
                        f"Created {len(chunks)} chunks "
                        f"({api_calls} embedding batches, "
                        f"{settings.embed_concurrency} parallel). This may take a few minutes."
                    )

                    progress = st.sidebar.progress(0.0)

                    def on_progress(done: int, total: int) -> None:
                        progress.progress(done / total)
                        status.write(f"Embedded {done}/{total} chunks")

                    added = store.add_chunks(chunks, on_progress=on_progress)
                    progress.empty()
                    status.update(
                        label=f"Indexed {uploaded.name}",
                        state="complete",
                        expanded=False,
                    )

                st.session_state.last_uploaded_source = uploaded.name
                st.session_state[cache_key] = True
                st.sidebar.success(f"Indexed {uploaded.name} ({added} chunks)")
            except Exception as exc:  # noqa: BLE001 - show user-facing upload errors
                st.sidebar.error(f"{uploaded.name}: {exc}")

    sources = st.session_state.vector_store.list_sources()
    st.sidebar.metric("Documents indexed", len(sources))
    if sources:
        st.sidebar.caption("Indexed files")
        for name in sources:
            marker = " (latest)" if name == st.session_state.last_uploaded_source else ""
            st.sidebar.write(f"- {name}{marker}")

    st.sidebar.divider()
    st.sidebar.header("LaunchDarkly")
    mode = "RAG (all documents)" if rag_enabled else "Non-RAG (latest upload only)"
    st.sidebar.info(f"Flag `rag-enabled`: **{'ON' if rag_enabled else 'OFF'}**")
    st.sidebar.caption(f"Active mode: {mode}")
    if not rag_enabled and st.session_state.last_uploaded_source:
        st.sidebar.caption(
            f"Non-RAG scope: `{st.session_state.last_uploaded_source}`"
        )


def main() -> None:
    st.set_page_config(page_title="DocuLens", page_icon="📄", layout="wide")
    init_session()

    settings = st.session_state.settings
    try:
        rag_enabled = is_rag_enabled(settings)
    except Exception as exc:  # noqa: BLE001
        st.error(f"LaunchDarkly error: {exc}")
        st.stop()

    st.title("DocuLens")
    st.caption("AI powered Document analyser and query answering app")

    render_sidebar(rag_enabled)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander("Retrieved sources"):
                    for citation in message["citations"]:
                        st.write(f"- {citation}")

    question = st.chat_input("Ask a question about your documents...")
    if not question:
        return

    store: SessionVectorStore = st.session_state.vector_store
    if store.document_count == 0:
        st.warning("Upload at least one document before asking a question.")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    source_filter = None
    mode_label = "RAG — search all uploaded documents"
    if not rag_enabled:
        source_filter = st.session_state.last_uploaded_source
        if not source_filter:
            st.error("No document uploaded yet for non-RAG mode.")
            return
        mode_label = f"Non-RAG — search only `{source_filter}`"

    with st.spinner("Retrieving relevant excerpts..."):
        hits = store.query(question, top_k=settings.top_k, source_filter=source_filter)

    if not hits:
        answer = (
            "I could not find relevant excerpts in the selected document scope. "
            "Try rephrasing or uploading more content."
        )
        citations: list[str] = []
    else:
        with st.spinner("Generating answer..."):
            answer, citations = answer_question(
                st.session_state.nvidia,
                question,
                hits,
                mode_label,
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
        if citations:
            with st.expander("Retrieved sources"):
                for citation in citations:
                    st.write(f"- {citation}")


if __name__ == "__main__":
    main()
