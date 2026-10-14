"""Report formatting and export utilities for FinDoc AI.

Generates clean, structured Markdown reports for validated Q&A answers,
two-document comparison matrices, and diagnostic summaries.
"""

from typing import Optional
from findoc.schemas import AnswerResult, ComparisonResult


def export_answer_markdown(question: str, result: AnswerResult) -> str:
    """Format an AnswerResult and user query into a clean Markdown report."""
    status_label = {
        "answered": "ANSWERED (GROUNDED IN RETRIEVED EVIDENCE)",
        "insufficient_evidence": "INSUFFICIENT EVIDENCE (ABSTAINED)",
        "error": "PROCESSING ERROR",
    }.get(result.status, result.status.upper())

    md = [
        "# FinDoc AI — Grounded Answer Report",
        "",
        f"**Question:** {question}  ",
        f"**Status:** `{status_label}`  ",
        f"**Execution Mode:** `{result.model_mode}`  ",
        "",
        "## Answer",
        result.answer,
        "",
    ]

    if result.citations:
        md.extend([
            "## Verified Source Citations",
            "",
            "| # | Source File | Location | Verified Chunk ID | Supporting Quote |",
            "|---|---|---|---|---|",
        ])
        for idx, cit in enumerate(result.citations, start=1):
            page_str = f"Page {cit.page_number}" if cit.page_number is not None else "Text File"
            clean_quote = cit.quote.replace("\n", " ").replace("|", "\\|")
            md.append(f"| {idx} | {cit.source_name} | {page_str} | `{cit.chunk_id}` | \"{clean_quote}\" |")
        md.append("")

    if result.limitations:
        md.extend([
            "## Notes & Limitations",
            "",
        ])
        for lim in result.limitations:
            md.append(f"- {lim}")
        md.append("")

    md.extend([
        "---",
        "*FinDoc AI is an assistive document reading prototype, not financial, legal, accounting, or investment advice.*",
    ])

    return "\n".join(md)


def export_comparison_markdown(
    result: ComparisonResult,
    doc_a_name: Optional[str] = None,
    doc_b_name: Optional[str] = None,
) -> str:
    """Format a ComparisonResult into a structured Markdown comparison matrix."""
    a_label = doc_a_name or result.document_a_id
    b_label = doc_b_name or result.document_b_id

    md = [
        "# FinDoc AI — Two-Document Comparison Report",
        "",
        f"**Topic / Question:** {result.question}  ",
        f"**Document A:** {a_label} (`{result.document_a_id}`)  ",
        f"**Document B:** {b_label} (`{result.document_b_id}`)  ",
        f"**Comparison Status:** `{result.status.upper()}`  ",
        "",
        "## Findings Matrix",
        "",
        "| Category | Topic | Summary | Document A Evidence | Document B Evidence | Caveat |",
        "|---|---|---|---|---|---|",
    ]

    for f in result.findings:
        cat_badge = f.category.upper()
        ev_a_str = ", ".join(f"`{e}`" for e in f.evidence_a) if f.evidence_a else "*None*"
        ev_b_str = ", ".join(f"`{e}`" for e in f.evidence_b) if f.evidence_b else "*None*"
        caveat_str = f.caveat if f.caveat else "—"
        summary_clean = f.summary.replace("\n", " ").replace("|", "\\|")

        row = f"| **{cat_badge}** | {f.topic} | {summary_clean} | {ev_a_str} | {ev_b_str} | {caveat_str} |"
        md.append(row)

    md.append("")

    if result.limitations:
        md.extend([
            "## Methodological Caveats",
            "",
        ])
        for lim in result.limitations:
            md.append(f"- {lim}")
        md.append("")

    md.extend([
        "---",
        "*Important Notice: Comparison findings are derived strictly from retrieved passages. Absence in retrieved passages does not verify absence from an entire document.*",
    ])

    return "\n".join(md)
