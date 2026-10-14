"""Two-document comparison service for FinDoc AI.

Performs independent dual retrieval, prevents cross-source citation contamination,
and validates structured comparison schemas.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from findoc.errors import FinDocError
from findoc.generation.prompts import (
    COMPARISON_SYSTEM_PROMPT,
    build_comparison_user_prompt,
)
from findoc.generation.provider import BaseLLMProvider
from findoc.retrieval.evidence import EvidenceItem, EvidencePack
from findoc.retrieval.retriever import RetrievalService
from findoc.schemas import (
    ComparisonCategory,
    ComparisonFinding,
    ComparisonResult,
    ComparisonStatus,
)
from findoc.storage.metadata_store import MetadataStore


class ComparisonService:
    """Orchestrates independent document comparison with strict source isolation."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        provider: BaseLLMProvider,
        metadata_store: MetadataStore,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.provider = provider
        self.metadata_store = metadata_store

    def compare_documents(
        self,
        document_a_id: str,
        document_b_id: str,
        question: str,
        top_k: int = 5,
    ) -> Tuple[ComparisonResult, EvidencePack, EvidencePack]:
        """Compare two documents independently on a focused question.

        Returns:
            Tuple of (ComparisonResult, evidence_pack_a, evidence_pack_b)

        Raises:
            FinDocError: If documents are identical or invalid.
        """
        doc_a_id = document_a_id.strip() if document_a_id else ""
        doc_b_id = document_b_id.strip() if document_b_id else ""

        if not doc_a_id or not doc_b_id:
            raise FinDocError("Both Document A and Document B must be selected for comparison.")

        if doc_a_id == doc_b_id:
            raise FinDocError("Document A and Document B must be distinct documents.")

        clean_q = question.strip() if question else ""
        if not clean_q:
            raise FinDocError("Comparison topic / question cannot be empty.")

        doc_a = self.metadata_store.get_document(doc_a_id)
        doc_b = self.metadata_store.get_document(doc_b_id)

        if not doc_a:
            raise FinDocError(f"Document A '{doc_a_id}' not found in metadata store.")
        if not doc_b:
            raise FinDocError(f"Document B '{doc_b_id}' not found in metadata store.")

        # 1. Independent Retrieval: Search A only, then B only
        pack_a = self.retrieval_service.retrieve(clean_q, [doc_a_id], top_k=top_k)
        pack_b = self.retrieval_service.retrieve(clean_q, [doc_b_id], top_k=top_k)

        # Check evidence presence
        has_a = pack_a.has_usable_evidence
        has_b = pack_b.has_usable_evidence

        if not has_a and not has_b:
            result = ComparisonResult(
                status="insufficient_evidence",
                document_a_id=doc_a_id,
                document_b_id=doc_b_id,
                question=clean_q,
                findings=[],
                limitations=[
                    "Neither Document A nor Document B returned relevant candidate passages for this question."
                ],
            )
            return result, pack_a, pack_b

        # 2. Format source-side evidence with clear prefixes (A1, A2 vs B1, B2)
        ev_a_text, a_id_map = self._format_side_evidence(pack_a, prefix="A")
        ev_b_text, b_id_map = self._format_side_evidence(pack_b, prefix="B")

        user_prompt = build_comparison_user_prompt(
            question=clean_q,
            doc_a_name=doc_a.display_name,
            evidence_a_text=ev_a_text,
            doc_b_name=doc_b.display_name,
            evidence_b_text=ev_b_text,
        )

        # 3. Call provider
        try:
            raw_response = self.provider.generate(
                user_prompt=user_prompt,
                system_instruction=COMPARISON_SYSTEM_PROMPT,
            )
        except Exception as e:
            result = ComparisonResult(
                status="error",
                document_a_id=doc_a_id,
                document_b_id=doc_b_id,
                question=clean_q,
                findings=[],
                limitations=[f"Provider error during comparison: {e}"],
            )
            return result, pack_a, pack_b

        # 4. Safely parse JSON structure
        parsed = self._parse_json_safely(raw_response)
        if not parsed:
            result = ComparisonResult(
                status="error",
                document_a_id=doc_a_id,
                document_b_id=doc_b_id,
                question=clean_q,
                findings=[],
                limitations=["Model output did not adhere to required comparison JSON schema."],
            )
            return result, pack_a, pack_b

        # 5. Validate findings and enforce source-side citation isolation
        validated_findings: List[ComparisonFinding] = []
        limitations: List[str] = [
            str(lim) for lim in parsed.get("limitations", []) if lim
        ]

        raw_findings = parsed.get("findings", [])
        if isinstance(raw_findings, list):
            for item in raw_findings:
                if not isinstance(item, dict):
                    continue

                raw_cat = str(item.get("category", "unclear")).lower()
                category: ComparisonCategory = (
                    raw_cat if raw_cat in ("same", "different", "only_in_a", "only_in_b", "unclear")  # type: ignore
                    else "unclear"
                )
                topic = str(item.get("topic", "Comparison finding")).strip() or "Finding"
                summary = str(item.get("summary", "")).strip() or "Comparison summary"
                caveat = item.get("caveat")

                # Validate evidence_a citations
                valid_a_ids: List[str] = []
                for ev_id in item.get("evidence_a", []):
                    clean_id = str(ev_id).strip("[]").strip()
                    if clean_id in a_id_map:
                        valid_a_ids.append(a_id_map[clean_id])
                    elif clean_id in pack_a.chunk_ids:
                        valid_a_ids.append(clean_id)
                    else:
                        limitations.append(
                            f"Discarded invalid/cross-contaminated Document A citation '{clean_id}'."
                        )

                # Validate evidence_b citations
                valid_b_ids: List[str] = []
                for ev_id in item.get("evidence_b", []):
                    clean_id = str(ev_id).strip("[]").strip()
                    if clean_id in b_id_map:
                        valid_b_ids.append(b_id_map[clean_id])
                    elif clean_id in pack_b.chunk_ids:
                        valid_b_ids.append(clean_id)
                    else:
                        limitations.append(
                            f"Discarded invalid/cross-contaminated Document B citation '{clean_id}'."
                        )

                # Integrity check: "different" and "same" normally require evidence from both sides
                if category in ("same", "different") and (not valid_a_ids or not valid_b_ids):
                    category = "unclear"
                    caveat = (
                        "Finding reclassified as unclear: claimed comparison lacked verified "
                        "evidence from both documents."
                    )

                validated_findings.append(
                    ComparisonFinding(
                        category=category,
                        topic=topic,
                        summary=summary,
                        evidence_a=valid_a_ids,
                        evidence_b=valid_b_ids,
                        caveat=caveat,
                    )
                )

        # 6. Determine final status
        if not has_a or not has_b:
            status: ComparisonStatus = "partial_evidence"
            limitations.append(
                "One of the compared documents yielded no candidate passages for this question."
            )
        elif not validated_findings:
            status = "insufficient_evidence"
        else:
            status = "complete"

        result = ComparisonResult(
            status=status,
            document_a_id=doc_a_id,
            document_b_id=doc_b_id,
            question=clean_q,
            findings=validated_findings,
            limitations=limitations,
        )

        return result, pack_a, pack_b

    @staticmethod
    def _format_side_evidence(
        pack: EvidencePack,
        prefix: str,
    ) -> Tuple[str, Dict[str, str]]:
        """Format evidence items with side-specific IDs (e.g. A1, A2 or B1, B2) and return map to chunk_ids."""
        if not pack.items:
            return "NO EVIDENCE AVAILABLE.", {}

        blocks: List[str] = []
        id_map: Dict[str, str] = {}

        for idx, item in enumerate(pack.items):
            side_id = f"{prefix}{idx + 1}"
            id_map[side_id] = item.chunk_id
            page_str = f"Page {item.page_number}" if item.page_number is not None else "Text File"
            blocks.append(
                f"[{side_id}] ({item.source_name}, {page_str})\n{item.text}"
            )

        return "\n\n".join(blocks), id_map

    @staticmethod
    def _parse_json_safely(raw_text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON object safely."""
        if not raw_text or not raw_text.strip():
            return None
        clean = raw_text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        try:
            parsed = json.loads(clean)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        match = re.search(r"(\{.*\})", clean, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return None
