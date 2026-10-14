"""FinDoc AI — Financial Document Intelligence Copilot.

Streamlit application providing document ingestion, vector indexing,
grounded question answering, and verifiable page-level citations.
"""

import json
from pathlib import Path
import streamlit as st

from findoc.config import get_settings
from findoc.generation.answer_service import AnswerService
from findoc.generation.provider import get_llm_provider
from findoc.ingestion.service import IngestionService
from findoc.retrieval.retriever import RetrievalService
from findoc.storage.embeddings import get_embedding_service
from findoc.storage.metadata_store import MetadataStore
from findoc.storage.vector_store import ChromaVectorStore

# Page configuration
st.set_page_config(
    page_title="FinDoc AI — Financial Document Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium look & feel
st.markdown(
    """
    <style>
    /* Metric & Card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-mock {
        display: inline-block;
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-live {
        display: inline-block;
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .citation-card {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #38bdf8;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-top: 10px;
        margin-bottom: 12px;
    }
    .citation-meta {
        font-size: 0.82rem;
        color: #38bdf8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .citation-quote {
        font-size: 0.92rem;
        color: #e2e8f0;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_app_services():
    """Initialize and cache application backend services."""
    settings = get_settings()
    settings.ensure_directories()

    meta_store = MetadataStore(settings.metadata_db_path)
    vstore = ChromaVectorStore(settings.chroma_dir)
    embedder = get_embedding_service(
        model_name=settings.embedding_model_name,
        mock=False,
    )
    provider = get_llm_provider(
        provider_name=settings.llm_provider,
        api_key=settings.gemini_api_key,
        model_name=settings.llm_model,
    )
    ingestion = IngestionService(
        metadata_store=meta_store,
        vector_store=vstore,
        embedding_service=embedder,
        chunk_size=settings.findoc_chunk_size,
        chunk_overlap=settings.findoc_chunk_overlap,
        max_file_mb=settings.findoc_max_file_mb,
    )
    retriever = RetrievalService(embedder, vstore)
    answer_service = AnswerService(retriever, provider)

    return settings, meta_store, vstore, ingestion, answer_service, provider


# Initialize services
settings, meta_store, vstore, ingestion, answer_service, provider = get_app_services()

# Sidebar Navigation
st.sidebar.title("FinDoc AI")
st.sidebar.caption("Financial Document Intelligence Copilot")

is_mock = "mock" in provider.model_mode.lower()
if is_mock:
    st.sidebar.markdown(
        '<span class="badge-mock">MOCK MODE ACTIVE</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Zero-key local test mode. Output is deterministic.")
else:
    st.sidebar.markdown(
        f'<span class="badge-live">{provider.model_mode.upper()}</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Live LLM connected. Excerpts are sent to configured provider.")

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Document Library", "Ask FinDoc", "About & Privacy"],
    index=0,
)

# -------------------------------------------------------------
# SCREEN 1: OVERVIEW
# -------------------------------------------------------------
if page == "Overview":
    st.title("FinDoc AI — Financial Document Intelligence")
    st.write(
        "A document-grounded AI copilot that indexes financial disclosures, annual reports, "
        "and policy terms page-by-page. FinDoc AI produces verifiable answers backed by exact "
        "page citations, and abstains when evidence is insufficient."
    )

    docs = meta_store.list_documents()
    total_docs = len(docs)
    total_chunks = vstore.count()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_docs}</div>
                <div class="metric-label">Indexed Documents</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_chunks}</div>
                <div class="metric-label">Searchable Chunks in Chroma</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        mode_label = "Mock (Local Deterministic)" if is_mock else "Live Provider"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 1.4rem; padding-top: 6px;">{mode_label}</div>
                <div class="metric-label">Model Runtime Mode</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("How It Works")
    st.markdown(
        """
        1. **Page-Aware Ingestion**: Upload a PDF or text file. FinDoc AI parses each page separately with PyMuPDF, preserving physical page numbers.
        2. **Local Embeddings**: Text chunks are embedded locally using `all-MiniLM-L6-v2` and indexed into persistent Chroma storage.
        3. **Grounded Retrieval**: When you ask a question, FinDoc AI searches only your selected documents and compiles a structured evidence pack.
        4. **Citation Verification**: The LLM synthesizes an answer, and the backend verifies that every cited passage genuinely exists in the evidence pack.
        5. **Explicit Abstention**: If the uploaded documents do not contain evidence for your question, FinDoc AI explicitly tells you rather than hallucinating.
        """
    )

    st.info(
        "🛡️ **Portfolio Notice**: This is an educational demonstration prototype. Use only public or synthetic documents. "
        "Do not upload confidential personal or business financial data."
    )

# -------------------------------------------------------------
# SCREEN 2: DOCUMENT LIBRARY
# -------------------------------------------------------------
elif page == "Document Library":
    st.title("Document Library & Ingestion")
    st.write(
        "Upload PDF, TXT, or Markdown documents. Uploaded files are validated, hashed for duplicate "
        "detection, and indexed with page-level provenance."
    )

    uploaded_files = st.file_uploader(
        "Upload Financial Documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help=f"Allowed formats: PDF, TXT, MD. Max size: {settings.findoc_max_file_mb} MB per file.",
    )

    if uploaded_files:
        if st.button("Process & Index Uploaded Files", type="primary"):
            progress_bar = st.progress(0)
            status_container = st.container()

            for i, uploaded_file in enumerate(uploaded_files):
                file_bytes = uploaded_file.read()
                filename = uploaded_file.name

                with status_container:
                    try:
                        doc_rec, is_dup = ingestion.ingest_file(filename, file_bytes)
                        if is_dup:
                            st.info(f"ℹ️ **{filename}**: Content identical to already indexed document `{doc_rec.display_name}`. Skipped duplicate re-indexing.")
                        elif doc_rec.status == "warning":
                            st.warning(f"⚠️ **{filename}**: Indexed with warning: {doc_rec.warning_message}")
                        else:
                            page_info = f"{doc_rec.page_count} pages" if doc_rec.page_count else "Text file"
                            st.success(f"✅ **{filename}**: Successfully indexed {page_info} ({doc_rec.chunk_count} searchable chunks).")
                    except Exception as e:
                        st.error(f"❌ **{filename}**: Ingestion failed: {e}")

                progress_bar.progress((i + 1) / len(uploaded_files))

            st.rerun()

    st.markdown("---")
    st.subheader("Indexed Documents")
    documents = meta_store.list_documents()

    if not documents:
        st.write("No documents indexed yet. Upload a document above to get started.")
    else:
        for doc in documents:
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                type_badge = doc.file_type.upper()
                pages_badge = f"{doc.page_count} pages" if doc.page_count else "Plain Text"
                st.markdown(f"**{doc.display_name}** `[{type_badge} | {pages_badge} | {doc.chunk_count} chunks]`")
                st.caption(f"ID: `{doc.document_id}` | Added: {doc.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                if doc.warning_message:
                    st.caption(f"⚠️ *{doc.warning_message}*")
            with col2:
                status_color = "green" if doc.status == "indexed" else ("orange" if doc.status == "warning" else "red")
                st.markdown(f":{status_color}[● {doc.status.upper()}]")
            with col3:
                # Safe delete confirmation toggle
                delete_key = f"del_{doc.document_id}"
                confirm_key = f"confirm_{doc.document_id}"
                if st.button("Delete", key=delete_key):
                    st.session_state[confirm_key] = True

                if st.session_state.get(confirm_key, False):
                    st.warning("Confirm removal?")
                    c_col1, c_col2 = st.columns(2)
                    with c_col1:
                        if st.button("Yes", key=f"yes_{doc.document_id}"):
                            ingestion.delete_document(doc.document_id)
                            st.session_state[confirm_key] = False
                            st.success(f"Removed {doc.display_name}")
                            st.rerun()
                    with c_col2:
                        if st.button("No", key=f"no_{doc.document_id}"):
                            st.session_state[confirm_key] = False
                            st.rerun()
            st.divider()

# -------------------------------------------------------------
# SCREEN 3: ASK FINDOC
# -------------------------------------------------------------
elif page == "Ask FinDoc":
    st.title("Ask FinDoc — Grounded Q&A")
    st.write("Ask natural-language questions across your selected documents with verified source citations.")

    documents = meta_store.list_documents()
    if not documents:
        st.warning("Please upload and index at least one document in the Document Library before asking questions.")
    else:
        doc_options = {doc.display_name: doc.document_id for doc in documents}
        
        col_sel1, col_sel2 = st.columns([4, 1])
        with col_sel1:
            selected_names = st.multiselect(
                "Select Documents to Query (Mandatory Scope Filter)",
                options=list(doc_options.keys()),
                default=list(doc_options.keys()),
                help="Retrieval will strictly query only the selected documents.",
            )
        with col_sel2:
            st.write("")
            st.write("")
            if st.button("Select All Docs"):
                st.session_state["selected_docs"] = list(doc_options.keys())

        selected_ids = [doc_options[name] for name in selected_names]

        # Sample question shortcuts
        st.caption("Quick example questions:")
        q_cols = st.columns(3)
        sample_q = ""
        with q_cols[0]:
            if st.button("What fees are described?"):
                sample_q = "What fees or charges are described in the document?"
        with q_cols[1]:
            if st.button("What is the stated revenue?"):
                sample_q = "What revenue or income figures are stated in the document?"
        with q_cols[2]:
            if st.button("Unanswerable: 2035 forecast?"):
                sample_q = "What is the company's projected revenue for 2035?"

        user_question = st.text_area(
            "Your Question",
            value=sample_q if sample_q else "",
            placeholder="e.g. What is the annual maintenance fee, and what conditions apply?",
            height=90,
        )

        with st.expander("⚙️ Advanced Retrieval Settings"):
            top_k = st.slider("Top-K Passages to Retrieve", min_value=1, max_value=10, value=5)
            show_debug = st.checkbox("Show diagnostic chunk IDs and search scores", value=False)

        if st.button("Submit Question", type="primary", disabled=not selected_ids or not user_question.strip()):
            with st.spinner("Retrieving evidence and synthesizing grounded answer..."):
                result = answer_service.answer_question(
                    question=user_question,
                    document_ids=selected_ids,
                    top_k=top_k,
                )

            st.markdown("---")
            # Answer status banner
            if result.status == "answered":
                st.success("✅ **Answer Grounded in Retrieved Evidence**")
            elif result.status == "insufficient_evidence":
                st.warning("⚠️ **Insufficient Evidence Found in Selected Documents**")
            else:
                st.error("❌ **Error During Answer Synthesis**")

            # Answer text
            st.markdown(f"### Answer\n{result.answer}")

            # Citations
            if result.citations:
                st.markdown("### Verified Source Citations")
                st.caption("Every citation below was verified by backend records against the current evidence pack.")

                for idx, cit in enumerate(result.citations):
                    page_label = f"Page {cit.page_number}" if cit.page_number is not None else "Text File"
                    debug_info = f" | Chunk: `{cit.chunk_id}`" if show_debug else ""
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-meta">[{idx + 1}] {cit.source_name} ({page_label}){debug_info}</div>
                            <div class="citation-quote">"{cit.quote}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Limitations and Transparency
            if result.limitations:
                with st.expander("ℹ️ Notes & System Disclaimers", expanded=False):
                    for lim in result.limitations:
                        st.write(f"- {lim}")

            # Export Report
            st.markdown("---")
            col_exp1, col_exp2 = st.columns([1, 4])
            with col_exp1:
                export_json = json.dumps(result.model_dump(), indent=2)
                st.download_button(
                    label="Download JSON Report",
                    data=export_json,
                    file_name="findoc_answer_report.json",
                    mime="application/json",
                )

# -------------------------------------------------------------
# SCREEN 4: ABOUT & PRIVACY
# -------------------------------------------------------------
elif page == "About & Privacy":
    st.title("About FinDoc AI & Privacy Architecture")
    st.markdown(
        """
        ### Architecture & Design
        FinDoc AI is built on a 5-step controlled RAG architecture:
        - **PyMuPDF**: Extracts text page-by-page to guarantee traceable page numbers.
        - **Chunking**: Uses character windowing with natural paragraph/sentence boundary snapping.
        - **Local Embeddings**: Embeds queries and chunks with `sentence-transformers/all-MiniLM-L6-v2`.
        - **Persistent Chroma Index**: Stores chunk vectors with document ID metadata for strict filtered search.
        - **Backend Citation Validation**: The model cites chunk/evidence IDs; application code checks each ID against the evidence pack and derives display filenames and page numbers from database records.

        ### Trust and Privacy Boundaries
        - **Learning/Portfolio Prototype**: This system is an assistive document reading tool, not financial, legal, or investment advice.
        - **Synthetic / Public Data**: Only test with public annual reports or synthetic disclosures.
        - **Model Provider Transparency**:
        """
    )
    if is_mock:
        st.info("Currently running in **MOCK MODE**. No data is sent over the network or to third-party model providers.")
    else:
        st.warning(f"Currently running with live provider **{provider.model_mode}**. User questions and retrieved excerpts are sent to the configured provider API.")
