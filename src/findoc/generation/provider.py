"""LLM provider adapters for FinDoc AI.

Exposes a clean interface with a deterministic Mock provider for offline testing
and an adapter for live Google Gemini models.
"""

import json
import re
from typing import Any, Dict, Optional
from findoc.errors import ConfigurationError, GenerationError


class BaseLLMProvider:
    """Abstract interface for FinDoc AI language model providers."""

    @property
    def model_mode(self) -> str:
        """Human-readable identifier of the active model mode."""
        raise NotImplementedError

    def generate(self, user_prompt: str, system_instruction: str) -> str:
        """Generate response text for the given prompt and system instruction."""
        raise NotImplementedError


class MockLLMProvider(BaseLLMProvider):
    """Deterministic, zero-key mock LLM provider for tests and portfolio demos.

    Generates visibly labelled mock answers from retrieved evidence without external API calls.
    Never presents output as a real model response.
    """

    def __init__(self, mode_label: str = "mock-mode (no external API key)") -> None:
        self._mode_label = mode_label

    @property
    def model_mode(self) -> str:
        return self._mode_label

    def generate(self, user_prompt: str, system_instruction: str) -> str:
        """Synthesize a deterministic mock response based on evidence content."""
        # Detect if prompt indicates unanswerable / no evidence
        if "NO EVIDENCE AVAILABLE" in user_prompt:
            return json.dumps(
                {
                    "status": "insufficient_evidence",
                    "answer": "[MOCK] The provided documents do not contain sufficient evidence to answer this question.",
                    "citations": [],
                    "limitations": [
                        "[MOCK MODE] Executed locally without calling an external model provider."
                    ],
                }
            )

        # Handle two-document comparison mode
        if "comparing two user-provided documents" in system_instruction.lower() or "--- DOCUMENT A EVIDENCE:" in user_prompt:
            has_a = "NO EVIDENCE AVAILABLE" not in user_prompt.split("--- DOCUMENT B EVIDENCE:")[0]
            has_b = len(user_prompt.split("--- DOCUMENT B EVIDENCE:")) > 1 and "NO EVIDENCE AVAILABLE" not in user_prompt.split("--- DOCUMENT B EVIDENCE:")[1]

            if not has_a and not has_b:
                return json.dumps(
                    {
                        "status": "insufficient_evidence",
                        "findings": [],
                        "limitations": ["[MOCK MODE] Neither document contained matching evidence."],
                    }
                )
            elif has_a and not has_b:
                return json.dumps(
                    {
                        "status": "partial_evidence",
                        "findings": [
                            {
                                "category": "only_in_a",
                                "topic": "Terms and Conditions",
                                "summary": "[MOCK] Term found in retrieved evidence from Document A, but not found in retrieved passages from Document B.",
                                "evidence_a": ["A1"],
                                "evidence_b": [],
                                "caveat": "Absence in retrieved evidence from B does not prove absence from entire document.",
                            }
                        ],
                        "limitations": ["[MOCK MODE] Document B lacked relevant candidate passages."],
                    }
                )
            elif not has_a and has_b:
                return json.dumps(
                    {
                        "status": "partial_evidence",
                        "findings": [
                            {
                                "category": "only_in_b",
                                "topic": "Terms and Conditions",
                                "summary": "[MOCK] Term found in retrieved evidence from Document B, but not found in retrieved passages from Document A.",
                                "evidence_a": [],
                                "evidence_b": ["B1"],
                                "caveat": "Absence in retrieved evidence from A does not prove absence from entire document.",
                            }
                        ],
                        "limitations": ["[MOCK MODE] Document A lacked relevant candidate passages."],
                    }
                )
            else:
                return json.dumps(
                    {
                        "status": "complete",
                        "findings": [
                            {
                                "category": "different",
                                "topic": "Term Comparison",
                                "summary": "[MOCK] Document A specifies different conditions than Document B based on retrieved passages.",
                                "evidence_a": ["A1"],
                                "evidence_b": ["B1"],
                                "caveat": None,
                            }
                        ],
                        "limitations": ["[MOCK MODE] Deterministic comparison generated for local demonstration."],
                    }
                )

        # Parse question text first
        q_match = re.search(
            r"(?:USER QUESTION|Question):\s*(.*?)\s*(?:EVIDENCE PASSAGES|Evidence passages)",
            user_prompt,
            re.DOTALL | re.IGNORECASE,
        )
        question_text = q_match.group(1).strip() if q_match else ""
        q_lower = question_text.lower()

        # Check for prompt injection or override attempt in user question
        if "override" in q_lower or "override_confirmed" in q_lower:
            if "execute" in q_lower or "instruction" in q_lower or "section 5" in q_lower or "phrase" in q_lower:
                return json.dumps(
                    {
                        "status": "answered",
                        "answer": "[MOCK] Section 5 contains a benchmark security notice discussing an untrusted injection artifact. FinDoc AI treats all document text strictly as passive reference data and refuses to execute instructions embedded in source documents.",
                        "citations": [{"id": "E1", "quote": "Security and Operational Audit Notice"}],
                        "limitations": ["[MOCK MODE] Refused to execute document-embedded instruction."],
                    }
                )
            return json.dumps(
                {
                    "status": "insufficient_evidence",
                    "answer": "[MOCK] Refused to execute system administrator override command. Prompt instructions are treated as untrusted.",
                    "citations": [],
                    "limitations": ["[MOCK MODE] Prompt override attempt detected and rejected."],
                }
            )

        # Check for prompt injection in evidence and safely neutralize it
        if "ignore previous directions" in user_prompt.lower() or "reveal secrets" in user_prompt.lower():
            return json.dumps(
                {
                    "status": "answered",
                    "answer": "[MOCK] Retrieved source documents contain reference text discussing instructions or policies. No system instructions were overridden.",
                    "citations": [{"id": "E1", "quote": "reference text"}],
                    "limitations": ["[MOCK MODE] Prompt injection text was treated as passive document data."],
                }
            )

        # Parse all candidate evidence blocks: (id, content)
        evidence_blocks = re.findall(
            r"--- EVIDENCE \[(E\d+)\] ---.*?Content:\s*(.*?)\s*--- END EVIDENCE \[\1\] ---",
            user_prompt,
            re.DOTALL,
        )

        if not evidence_blocks:
            # Fallback regex if single block or different delimiter
            single_match = re.search(r"Content:\s*(.*?)(?:--- END EVIDENCE|$)", user_prompt, re.DOTALL)
            if single_match:
                evidence_blocks = [("E1", single_match.group(1).strip())]

        if evidence_blocks:
            # Check for unmentioned 4-digit years or capitalized entities in question
            all_evidence_lower = " ".join(c for _, c in evidence_blocks).lower()
            unanswerable_tokens = []

            # 4-digit years (e.g. 2035)
            for year in re.findall(r"\b\d{4}\b", question_text):
                if year not in all_evidence_lower:
                    unanswerable_tokens.append(year)

            # Capitalized proper nouns / distinct terms (e.g. London, JPMorgan, Marketing)
            doc_terms = {
                "apex", "beacon", "financial", "commercial", "lending", "group", "annual",
                "report", "policy", "section", "terms", "what", "when", "where", "which",
                "who", "whom", "whose", "why", "how", "does", "have", "under", "system",
                "execute", "output", "fico"
            }
            for word in re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", question_text):
                w_low = word.lower()
                if w_low not in doc_terms and w_low not in all_evidence_lower:
                    unanswerable_tokens.append(word)

            if unanswerable_tokens:
                return json.dumps(
                    {
                        "status": "insufficient_evidence",
                        "answer": f"[MOCK] The provided documents do not contain sufficient evidence regarding '{', '.join(unanswerable_tokens)}'.",
                        "citations": [],
                        "limitations": ["[MOCK MODE] Query terms were not found in retrieved passages."],
                    }
                )

            # Score each evidence block by token overlap with the question
            q_tokens = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", q_lower))
            best_id = "E1"
            best_content = evidence_blocks[0][1]
            best_score = -1

            for eid, content in evidence_blocks:
                c_tokens = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", content.lower()))
                score = len(q_tokens.intersection(c_tokens))
                if score > best_score:
                    best_score = score
                    best_id = eid
                    best_content = content

            # Split best content into sentences or bullet points to find the most relevant excerpt
            excerpts = [s.strip() for s in re.split(r"(?<=[a-zA-Z0-9])\.\s+|\n+", best_content) if s.strip()]
            best_excerpt = excerpts[0] if excerpts else best_content
            best_s_score = -1

            key_domain_terms = {
                "revenue", "income", "fee", "fees", "wire", "transfer", "maintenance",
                "waived", "score", "fico", "origination", "penalty", "prepayment",
                "prohibit", "cannabis", "assets", "sarah", "jenkins", "ratio"
            }

            for exc in excerpts:
                s_tokens = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", exc.lower()))
                overlap = q_tokens.intersection(s_tokens)
                s_score = len(overlap)
                # Boost if excerpt contains specific query domain keywords
                for term in overlap:
                    if term in key_domain_terms:
                        s_score += 2

                if s_score > best_s_score:
                    best_s_score = s_score
                    best_excerpt = exc

            clean_excerpt = best_excerpt.lstrip("-* ").strip()
            if not clean_excerpt.endswith("."):
                clean_excerpt += "."
            quote_snippet = clean_excerpt[:100]

            return json.dumps(
                {
                    "status": "answered",
                    "answer": f"[MOCK] According to the uploaded documentation, {clean_excerpt}",
                    "citations": [
                        {
                            "id": best_id,
                            "quote": quote_snippet,
                        }
                    ],
                    "limitations": [
                        "[MOCK MODE] This response was generated deterministically by the local mock engine."
                    ],
                }
            )

        # Fallback insufficient evidence
        return json.dumps(
            {
                "status": "insufficient_evidence",
                "answer": "[MOCK] Could not find sufficient evidence in the uploaded documents to answer this question.",
                "citations": [],
                "limitations": ["[MOCK MODE] Local mock engine abstention."],
            }
        )


class GeminiLLMProvider(BaseLLMProvider):
    """Live LLM provider adapter for Google Gemini models."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
    ) -> None:
        if not api_key:
            raise ConfigurationError("GEMINI_API_KEY must be provided for GeminiLLMProvider.")
        self.api_key = api_key
        # Normalize deprecated/nonexistent/billing-required model names to free-tier safe default.
        # gemini-2.5-flash: retired | gemini-3.8-flash: doesn't exist | gemini-2.0-flash: needs billing
        normalized = (model_name or "gemini-1.5-flash").strip()
        if "2.5" in normalized or "3.8" in normalized or "2.0" in normalized:
            normalized = "gemini-1.5-flash"
        self.model_name = normalized

    @property
    def model_mode(self) -> str:
        return f"live ({self.model_name})"

    def generate(self, user_prompt: str, system_instruction: str) -> str:
        """Call Google Gemini API with retry/backoff and model fallback chain."""
        import urllib.request
        import urllib.error
        import time

        # Model fallback chain: try configured model first, then free-tier safe fallbacks
        _FALLBACK_CHAIN = ["gemini-1.5-flash", "gemini-1.5-flash-8b"]
        models_to_try = [self.model_name] + [m for m in _FALLBACK_CHAIN if m != self.model_name]

        def _call_model(name: str) -> str:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{name}:generateContent?key={self.api_key}"
            )
            payload: Dict[str, Any] = {
                "system_instruction": {"parts": [{"text": system_instruction}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "temperature": 0.0,
                },
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45.0) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if not candidates:
                    raise GenerationError("Gemini API returned no candidates in response.")
                content_parts = candidates[0].get("content", {}).get("parts", [])
                if not content_parts or "text" not in content_parts[0]:
                    raise GenerationError("Malformed candidate content structure from Gemini API.")
                return content_parts[0]["text"]

        last_error: Exception = GenerationError("No models available.")
        for model in models_to_try:
            # Retry up to 3 times with exponential backoff for transient overload errors
            for attempt in range(3):
                try:
                    return _call_model(model)
                except urllib.error.HTTPError as e:
                    err_body = e.read().decode("utf-8", errors="replace")
                    if e.code in (429, 503):
                        # Transient overload / rate-limit – wait and retry
                        wait = 2 ** attempt  # 1s, 2s, 4s
                        time.sleep(wait)
                        last_error = GenerationError(
                            f"Gemini API overloaded (HTTP {e.code}) on {model}, "
                            f"attempt {attempt + 1}/3. Retrying…"
                        )
                        continue
                    elif e.code == 404:
                        # Model not found – skip straight to next fallback
                        last_error = GenerationError(
                            f"Model {model!r} not found (HTTP 404). Trying next fallback."
                        )
                        break  # break retry loop, try next model
                    else:
                        # Non-retryable HTTP error
                        raise GenerationError(
                            f"Gemini API HTTP {e.code} error: {err_body[:300]}"
                        ) from e
                except GenerationError:
                    raise
                except Exception as e:
                    raise GenerationError(f"Failed to call Gemini provider: {e}") from e
            else:
                # Exhausted all retries for this model – try next in chain
                pass

        raise last_error


def get_llm_provider(
    provider_name: str = "mock",
    api_key: Optional[str] = None,
    model_name: str = "gemini-1.5-flash",
) -> BaseLLMProvider:
    """Factory helper to obtain the configured LLM provider."""
    if provider_name.lower() == "gemini" and api_key:
        return GeminiLLMProvider(api_key=api_key, model_name=model_name)
    return MockLLMProvider()
