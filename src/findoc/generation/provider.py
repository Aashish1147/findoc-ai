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

        # Extract first evidence passage content
        content_match = re.search(r"Content:\s*(.*?)\s*--- END EVIDENCE", user_prompt, re.DOTALL)
        if content_match:
            first_content = content_match.group(1).strip()
            # Take the first sentence (split on sentence terminator, avoiding decimals in numbers)
            sentences = re.split(r"(?<=[a-zA-Z0-9])\.\s+|\n+", first_content)
            first_sentence = sentences[0].strip()
            if not first_sentence.endswith("."):
                first_sentence += "."
            quote_snippet = first_sentence[:100]

            return json.dumps(
                {
                    "status": "answered",
                    "answer": f"[MOCK] According to the uploaded documentation, {first_sentence}",
                    "citations": [
                        {
                            "id": "E1",
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
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        if not api_key:
            raise ConfigurationError("GEMINI_API_KEY must be provided for GeminiLLMProvider.")
        self.api_key = api_key
        self.model_name = model_name

    @property
    def model_mode(self) -> str:
        return f"live ({self.model_name})"

    def generate(self, user_prompt: str, system_instruction: str) -> str:
        """Call Google Gemini API via official HTTP REST endpoint using json mode."""
        import urllib.request
        import urllib.error

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

        payload: Dict[str, Any] = {
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
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

        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if not candidates:
                    raise GenerationError("Gemini API returned no candidates in response.")
                content_parts = candidates[0].get("content", {}).get("parts", [])
                if not content_parts or "text" not in content_parts[0]:
                    raise GenerationError("Malformed candidate content structure from Gemini API.")
                return content_parts[0]["text"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            # Do NOT log or expose the API key in the error message
            raise GenerationError(f"Gemini API HTTP {e.code} error: {err_body[:200]}") from e
        except Exception as e:
            raise GenerationError(f"Failed to call Gemini provider: {e}") from e


def get_llm_provider(
    provider_name: str = "mock",
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
) -> BaseLLMProvider:
    """Factory helper to obtain the configured LLM provider."""
    if provider_name.lower() == "gemini" and api_key:
        return GeminiLLMProvider(api_key=api_key, model_name=model_name)
    return MockLLMProvider()
