"""FinDoc AI — Financial Document Intelligence Copilot.

Streamlit application providing document ingestion, vector indexing,
grounded question answering, and verifiable page-level citations.
"""

import json
from pathlib import Path
import sys

# Ensure 'src' directory is on sys.path for Streamlit Cloud and local runs without PYTHONPATH
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st

from findoc.comparison.service import ComparisonService
from findoc.config import get_settings
from findoc.evaluation.runner import (
    EvaluationRunner,
    export_evaluation_markdown,
    load_evaluation_dataset,
)
from findoc.generation.answer_service import AnswerService
from findoc.generation.provider import get_llm_provider
from findoc.ingestion.service import IngestionService
from findoc.reporting import export_answer_markdown, export_comparison_markdown
from findoc.retrieval.retriever import RetrievalService
from findoc.schemas import EvalSummary
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
    .badge-pass {
        display: inline-block;
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-fail {
        display: inline-block;
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-notrun {
        display: inline-block;
        background: rgba(148, 163, 184, 0.2);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.4);
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
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
    # Warm up embedding model at startup so user interactions never stall or drop WebSocket connection
    try:
        embedder.embed_query("FinDoc startup warmup")
    except Exception:
        pass
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
    comparison_service = ComparisonService(retriever, provider, meta_store)
    eval_runner = EvaluationRunner(
        ingestion_service=ingestion,
        retrieval_service=retriever,
        answer_service=answer_service,
        metadata_store=meta_store,
    )

    return settings, meta_store, vstore, ingestion, answer_service, provider, comparison_service, eval_runner


# Initialize services
settings, meta_store, vstore, ingestion, answer_service, provider, comparison_service, eval_runner = get_app_services()

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
    ["Overview", "Document Library", "Ask FinDoc", "Compare Documents", "Evaluation & Diagnostics", "About & Privacy"],
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
            col_exp1, col_exp2, _ = st.columns([1, 1, 3])
            with col_exp1:
                export_json = json.dumps(result.model_dump(), indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=export_json,
                    file_name="findoc_answer_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with col_exp2:
                export_md = export_answer_markdown(user_question, result)
                st.download_button(
                    label="📄 Download Markdown",
                    data=export_md,
                    file_name="findoc_answer_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

# -------------------------------------------------------------
# SCREEN 4: COMPARE DOCUMENTS
# -------------------------------------------------------------
elif page == "Compare Documents":
    st.title("Compare Documents — Grounded Comparison")
    st.write(
        "Select Document A and Document B to compare terms, conditions, fees, or metrics. "
        "Each finding requires verifiable evidence from both sides without assuming unretrieved "
        "facts are absent from the entire document."
    )

    documents = meta_store.list_documents()
    if len(documents) < 2:
        st.warning("Please upload and index at least TWO documents in the Document Library to perform comparisons.")
    else:
        doc_options = {doc.display_name: doc.document_id for doc in documents}
        doc_names = list(doc_options.keys())

        col_a, col_b = st.columns(2)
        with col_a:
            doc_a_name = st.selectbox("Document A (Reference / Baseline)", options=doc_names, index=0)
        with col_b:
            doc_b_name = st.selectbox("Document B (Comparison Target)", options=doc_names, index=min(1, len(doc_names) - 1))

        if doc_a_name == doc_b_name:
            st.error("Document A and Document B must be distinct documents. Please choose different documents.")

        st.caption("Quick comparison topics:")
        top_cols = st.columns(3)
        comp_sample = ""
        with top_cols[0]:
            if st.button("Compare fee terms & waivers"):
                comp_sample = "Compare the fee terms, maintenance charges, and any balance waiver requirements."
        with top_cols[1]:
            if st.button("Compare eligibility criteria"):
                comp_sample = "Compare customer eligibility requirements, qualifications, and restrictions."
        with top_cols[2]:
            if st.button("Compare reporting periods"):
                comp_sample = "Compare the reporting periods, dates, and accounting cycles covered."

        comp_question = st.text_area(
            "Comparison Topic / Question",
            value=comp_sample if comp_sample else "",
            placeholder="e.g. Compare the annual maintenance fee terms and penalty provisions between Document A and Document B.",
            height=80,
        )

        with st.expander("⚙️ Advanced Comparison Settings"):
            comp_top_k = st.slider("Top-K passages per document", min_value=1, max_value=8, value=4)

        if st.button("Run Grounded Comparison", type="primary", disabled=doc_a_name == doc_b_name or not comp_question.strip()):
            doc_a_id = doc_options[doc_a_name]
            doc_b_id = doc_options[doc_b_name]

            with st.spinner("Retrieving evidence independently from Document A and Document B..."):
                comp_result, pack_a, pack_b = comparison_service.compare_documents(
                    document_a_id=doc_a_id,
                    document_b_id=doc_b_id,
                    question=comp_question,
                    top_k=comp_top_k,
                )

            st.markdown("---")
            if comp_result.status == "complete":
                st.success("✅ **Comparison Complete (Evidence Verified on Both Sides)**")
            elif comp_result.status == "partial_evidence":
                st.warning("⚠️ **Partial Evidence (One document lacked candidate passages)**")
            else:
                st.error("❌ **Insufficient Evidence or Comparison Error**")

            st.markdown(f"### Findings for: *{comp_result.question}*")
            if not comp_result.findings:
                st.write("No substantiated comparison findings could be derived from the retrieved passages.")
            else:
                for idx, finding in enumerate(comp_result.findings):
                    cat = finding.category.upper()
                    cat_color = "green" if cat == "SAME" else ("blue" if cat == "DIFFERENT" else ("orange" if "ONLY" in cat else "gray"))
                    st.markdown(f"#### {idx + 1}. :{cat_color}[● {cat}]: {finding.topic}")
                    st.write(finding.summary)
                    if finding.caveat:
                        st.caption(f"⚠️ *{finding.caveat}*")

                    # Side-by-side evidence
                    e_col_a, e_col_b = st.columns(2)
                    with e_col_a:
                        st.caption(f"**Evidence from {doc_a_name}:**")
                        if not finding.evidence_a:
                            st.write("*No direct passage cited.*")
                        else:
                            for chunk_id in finding.evidence_a:
                                item = pack_a.get_by_chunk_id(chunk_id)
                                if item:
                                    page_lbl = f"Page {item.page_number}" if item.page_number else "Text File"
                                    st.markdown(
                                        f'<div class="citation-card"><div class="citation-meta">{item.source_name} ({page_lbl})</div><div class="citation-quote">"{item.text[:140]}..."</div></div>',
                                        unsafe_allow_html=True,
                                    )
                    with e_col_b:
                        st.caption(f"**Evidence from {doc_b_name}:**")
                        if not finding.evidence_b:
                            st.write("*No direct passage cited.*")
                        else:
                            for chunk_id in finding.evidence_b:
                                item = pack_b.get_by_chunk_id(chunk_id)
                                if item:
                                    page_lbl = f"Page {item.page_number}" if item.page_number else "Text File"
                                    st.markdown(
                                        f'<div class="citation-card"><div class="citation-meta">{item.source_name} ({page_lbl})</div><div class="citation-quote">"{item.text[:140]}..."</div></div>',
                                        unsafe_allow_html=True,
                                    )
                    st.divider()

            if comp_result.limitations:
                with st.expander("ℹ️ Comparison Limitations & Caveats", expanded=False):
                    for lim in comp_result.limitations:
                        st.write(f"- {lim}")

            # Export button
            col_comp_exp1, col_comp_exp2, _ = st.columns([1, 1, 3])
            with col_comp_exp1:
                comp_json = json.dumps(comp_result.model_dump(), indent=2)
                st.download_button(
                    label="📥 Export JSON",
                    data=comp_json,
                    file_name="findoc_comparison_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with col_comp_exp2:
                name_a = meta_store.get_document(doc_a_id).display_name if meta_store.get_document(doc_a_id) else doc_a_id
                name_b = meta_store.get_document(doc_b_id).display_name if meta_store.get_document(doc_b_id) else doc_b_id
                comp_md = export_comparison_markdown(comp_result, name_a, name_b)
                st.download_button(
                    label="📄 Export Markdown",
                    data=comp_md,
                    file_name="findoc_comparison_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

# -------------------------------------------------------------
# SCREEN 5: EVALUATION & DIAGNOSTICS
# -------------------------------------------------------------
elif page == "Evaluation & Diagnostics":
    st.title("Evaluation Suite & Quality Diagnostics")
    st.write(
        "Benchmark retrieval recall (Hit@K), citation-ID provenance, explicit abstention on "
        "unanswerable questions, and prompt-injection defense against a fixed 18-case synthetic financial dataset."
    )

    eval_summary: EvalSummary = st.session_state.get("eval_summary")

    if eval_summary is None:
        st.markdown(
            """
            <div class="metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0; color:#f8fafc;">Benchmark Execution Status</h3>
                    <span class="badge-notrun">NOT RUN</span>
                </div>
                <p style="color:#94a3b8; margin-top:10px; margin-bottom:0;">
                    In accordance with FinDoc AI engineering rules and <code>AGENTS.md</code>, no fabricated 
                    pass rates, simulated scores, or placeholder percentages are displayed until the real 
                    evaluation engine has executed against local test documents.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            #### What This Benchmark Measures:
            - **Retrieval Hit@K**: Measures whether the ground-truth evidence excerpt appears in top-k candidate chunks for answerable queries.
            - **Citation-ID Validity**: Validates that 100% of generated citations reference real retrieved chunk IDs from the active query evidence pack.
            - **Abstention Accuracy**: Verifies that unanswerable queries (e.g. unmentioned fiscal years, non-existent branches) return `insufficient_evidence` instead of guessing.
            - **Prompt Injection Defense**: Validates that system override commands and document-embedded instructions are neutralized as untrusted data.
            """
        )

        if st.button("🚀 Run Live Evaluation Benchmark", type="primary", use_container_width=True):
            progress_bar = st.progress(0.0)
            status_box = st.empty()

            def on_progress(current: int, total: int, res):
                progress_bar.progress(current / total)
                status_box.markdown(f"Evaluating `{res.case_id}` ({current}/{total}): **{res.question[:55]}...**")

            with st.spinner("Executing benchmark across synthetic financial disclosures..."):
                try:
                    summary = eval_runner.run_evaluation(on_progress=on_progress)
                    st.session_state["eval_summary"] = summary
                    st.success("Benchmark completed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Evaluation benchmark execution failed: {e}")
    else:
        # Display KPI Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{eval_summary.overall_pass_rate * 100:.1f}%</div>
                    <div class="metric-label">Overall Pass Rate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{eval_summary.retrieval_hit_rate * 100:.1f}%</div>
                    <div class="metric-label">Retrieval Hit@K</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{eval_summary.citation_validity_rate * 100:.1f}%</div>
                    <div class="metric-label">Citation Validity</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{eval_summary.abstention_rate * 100:.1f}%</div>
                    <div class="metric-label">Abstention Accuracy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{eval_summary.injection_resistance_rate * 100:.1f}%</div>
                    <div class="metric-label">Injection Defense</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 2])
        with sub_col1:
            st.write(f"**Passed Cases:** {eval_summary.passed_cases} / {eval_summary.total_cases}")
        with sub_col2:
            st.write(f"**Average Query Latency:** {eval_summary.avg_latency_ms:.1f} ms")
        with sub_col3:
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button("🔄 Rerun Benchmark", use_container_width=True):
                    st.session_state["eval_summary"] = None
                    st.rerun()
            with btn_col2:
                report_json = eval_summary.model_dump_json(indent=2)
                st.download_button(
                    label="📥 Export JSON",
                    data=report_json,
                    file_name="findoc_evaluation_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with btn_col3:
                report_md = export_evaluation_markdown(eval_summary)
                st.download_button(
                    label="📄 Export Markdown",
                    data=report_md,
                    file_name="findoc_evaluation_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        st.divider()

        # Filter by category
        categories = ["All"] + sorted(list({c.category for c in eval_summary.case_results}))
        selected_cat = st.selectbox("Filter test cases by category:", categories, index=0)

        filtered_cases = (
            eval_summary.case_results
            if selected_cat == "All"
            else [c for c in eval_summary.case_results if c.category == selected_cat]
        )

        st.markdown(f"### Diagnostics Breakdown ({len(filtered_cases)} cases)")

        for c in filtered_cases:
            status_badge = (
                '<span class="badge-pass">PASS</span>'
                if c.passed
                else '<span class="badge-fail">FAIL</span>'
            )
            hit_badge = "✅ Hit" if c.retrieval_hit else "❌ Miss"
            cit_badge = "✅ Valid Citations" if c.citation_validity else "❌ Invalid Citation"
            abs_badge = "✅ Abstention OK" if c.abstention_correct else "❌ Abstention Fail"
            inj_badge = "✅ Protected" if c.injection_resistant else "❌ Injected"

            with st.expander(f"{c.case_id} — {c.category.upper()} | {c.question[:65]}...", expanded=not c.passed):
                header_cols = st.columns([1, 1, 1, 1, 1])
                with header_cols[0]:
                    st.markdown(f"**Outcome:** {status_badge}", unsafe_allow_html=True)
                with header_cols[1]:
                    st.write(f"**Retrieval:** {hit_badge}")
                with header_cols[2]:
                    st.write(f"**Citations:** {cit_badge}")
                with header_cols[3]:
                    st.write(f"**Abstention:** {abs_badge}")
                with header_cols[4]:
                    st.write(f"**Safety:** {inj_badge}")

                st.write(f"**Question:** {c.question}")
                st.write(f"**Expected Status:** `{c.expected_status}` | **Actual Status:** `{c.actual_status}` | **Latency:** `{c.latency_ms:.1f} ms`")
                st.write(f"**Model Answer / Explanation:** {c.actual_answer}")
                if c.notes:
                    st.caption(f"**Benchmark Notes:** {c.notes}")

        st.info(f"**Evaluation Methodology Notice:** {eval_summary.disclaimer}")

# -------------------------------------------------------------
# SCREEN 6: ABOUT & PRIVACY
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
