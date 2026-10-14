"""Answer generation service for FinDoc AI.

Orchestrates retrieval, evidence gating, LLM prompt assembly, schema validation,
and backend citation provenance verification.
"""

import json
from typing import Any, Dict, List, Optional

from findoc.errors import GenerationError
from findoc.generation.prompts import QA_SYSTEM_PROMPT, build_qa_user_prompt
from findoc.generation.provider import BaseLLMProvider
from findoc.retrieval.evidence import EvidencePack
from findoc.retrieval.retriever import RetrievalService
from findoc.schemas import AnswerResult, AnswerStatus, Citation


class AnswerService:
    """Orchestrates grounded question answering with strict evidence validation."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        provider: BaseLLMProvider,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.provider = provider

    def answer_question(
        self,
        question: str,
        document_ids: List[str],
        top_k: int = 5,
    ) -> AnswerResult:
        """Execute end-to-end grounded document Q&A.

        Args:
            question: User's question.
            document_ids: Selected document IDs to query.
            top_k: Maximum candidate passages to retrieve.

        Returns:
            Validated AnswerResult with verified citations.
        """
        # 1. Retrieve candidates
        evidence_pack = self.retrieval_service.retrieve(
            question=question,
            document_ids=document_ids,
            top_k=top_k,
        )

        # 2. Evidence Gate: if no evidence found, abstain without calling the LLM
        if not evidence_pack.has_usable_evidence:
            return AnswerResult(
                status="insufficient_evidence",
                answer="The selected documents do not contain sufficient evidence to answer this question.",
                citations=[],
                retrieved_chunk_ids=[],
                model_mode=self.provider.model_mode,
                limitations=["No relevant candidate passages were retrieved from the selected documents."],
            )

        # 3. Build prompt and invoke provider
        user_prompt = build_qa_user_prompt(
            question=question,
            formatted_evidence=evidence_pack.format_for_prompt(),
        )

        try:
            raw_response = self.provider.generate(
                user_prompt=user_prompt,
                system_instruction=QA_SYSTEM_PROMPT,
            )
        except Exception as e:
            return AnswerResult(
                status="error",
                answer=f"Error occurred during answer generation: {e}",
                citations=[],
                retrieved_chunk_ids=evidence_pack.chunk_ids,
                model_mode=self.provider.model_mode,
                limitations=[f"Provider failure: {type(e).__name__}"],
            )

        # 4. Safely parse JSON structure
        parsed_data = self._parse_json_safely(raw_response)
        if not parsed_data:
            return AnswerResult(
                status="error",
                answer="Model output was malformed and did not adhere to the required JSON schema.",
                citations=[],
                retrieved_chunk_ids=evidence_pack.chunk_ids,
                model_mode=self.provider.model_mode,
                limitations=["Output schema parsing failure."],
            )

        # 5. Extract answer fields
        status_raw = str(parsed_data.get("status", "answered")).lower()
        status: AnswerStatus = "insufficient_evidence" if "insufficient" in status_raw else "answered"
        answer_text = str(parsed_data.get("answer", "")).strip()
        if not answer_text:
            answer_text = "No substantive answer text returned by model."

        limitations: List[str] = [
            str(lim) for lim in parsed_data.get("limitations", []) if lim
        ]

        # 6. Backend Citation Verification:
        # Never accept model-generated page numbers or unretrieved IDs
        verified_citations: List[Citation] = []
        raw_citations = parsed_data.get("citations", [])

        if isinstance(raw_citations, list):
            for raw_cit in raw_citations:
                if not isinstance(raw_cit, dict):
                    continue
                cited_id = str(raw_cit.get("id", "")).strip()
                quote = str(raw_cit.get("quote", "")).strip()

                citation = evidence_pack.to_citation(cited_id, quote)
                if citation:
                    verified_citations.append(citation)
                else:
                    limitations.append(
                        f"Discarded unverified citation ID '{cited_id}' (not found in current evidence pack)."
                    )

        # If model claimed answered but provided zero verified citations, note in limitations
        if status == "answered" and not verified_citations and evidence_pack.has_usable_evidence:
            limitations.append("Answer contains claims without traceable passage citations.")

        return AnswerResult(
            status=status,
            answer=answer_text,
            citations=verified_citations,
            retrieved_chunk_ids=evidence_pack.chunk_ids,
            model_mode=self.provider.model_mode,
            limitations=limitations,
        )

    @staticmethod
    def _parse_json_safely(raw_text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON object from raw model string."""
        if not raw_text or not raw_text.strip():
            return None

        clean = raw_text.strip()
        # Strip markdown fences if present
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

        # Try regex search for first { ... }
        import re
        match = re.search(r"(\{.*\})", clean, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        return None
