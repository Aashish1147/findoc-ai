"""Unit tests for Q&A prompt templates and schema instructions."""

from findoc.generation.prompts import QA_SYSTEM_PROMPT, build_qa_user_prompt


def test_system_prompt_rules_and_schema():
    """Verify that system prompt emphasizes untrusted data and schema requirements."""
    assert "UNTRUSTED REFERENCE DATA" in QA_SYSTEM_PROMPT
    assert "insufficient_evidence" in QA_SYSTEM_PROMPT
    assert "status" in QA_SYSTEM_PROMPT
    assert "citations" in QA_SYSTEM_PROMPT


def test_build_qa_user_prompt_formatting():
    """Verify user prompt embeds question and evidence passages cleanly."""
    prompt = build_qa_user_prompt(
        question="What were the total operating expenses?",
        formatted_evidence="--- EVIDENCE [E1] ---\nOperating expenses were $5.4M.\n--- END EVIDENCE [E1] ---",
    )
    assert "USER QUESTION:" in prompt
    assert "What were the total operating expenses?" in prompt
    assert "Operating expenses were $5.4M." in prompt
