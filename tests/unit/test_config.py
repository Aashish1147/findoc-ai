"""Unit tests for FinDoc AI configuration and settings."""

from pathlib import Path
import pytest
from pydantic import ValidationError

from findoc.config import Settings, get_settings
from findoc.errors import ConfigurationError


def test_default_settings_no_api_key_required():
    """Verify that default settings initialize in mock mode without requiring an API key."""
    settings = Settings(
        llm_provider="mock",
        _env_file=None,  # ignore any external .env during test
    )
    assert settings.llm_provider == "mock"
    assert settings.findoc_chunk_size == 1000
    assert settings.findoc_chunk_overlap == 150
    assert settings.findoc_top_k == 5
    assert settings.findoc_max_file_mb == 20
    assert settings.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    # Should not raise ConfigurationError
    settings.validate_provider_credentials()


def test_derived_directory_paths():
    """Verify derived directories point to expected paths under data root."""
    test_data_dir = Path("./test_data_dir")
    settings = Settings(findoc_data_dir=test_data_dir, _env_file=None)
    assert settings.chroma_dir == test_data_dir / "chroma"
    assert settings.metadata_db_path == test_data_dir / "app_metadata.sqlite"
    assert settings.uploads_dir == test_data_dir / "uploads"


def test_chunk_overlap_must_be_less_than_chunk_size():
    """Verify that overlap equal to or greater than chunk size raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            findoc_chunk_size=500,
            findoc_chunk_overlap=500,
            _env_file=None,
        )
    assert "findoc_chunk_overlap (500) must be strictly less than findoc_chunk_size (500)" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        Settings(
            findoc_chunk_size=500,
            findoc_chunk_overlap=600,
            _env_file=None,
        )
    assert "must be strictly less" in str(exc_info.value)


def test_negative_chunk_overlap_rejected():
    """Verify that negative overlap values are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            findoc_chunk_size=500,
            findoc_chunk_overlap=-10,
            _env_file=None,
        )
    assert "findoc_chunk_overlap must be non-negative" in str(exc_info.value)


def test_positive_integers_enforced():
    """Verify positive integer constraints on file size, max files, and top_k."""
    with pytest.raises(ValidationError):
        Settings(findoc_max_file_mb=0, _env_file=None)

    with pytest.raises(ValidationError):
        Settings(findoc_top_k=-1, _env_file=None)

    with pytest.raises(ValidationError):
        Settings(findoc_max_files_per_upload=0, _env_file=None)


def test_gemini_provider_requires_key():
    """Verify that selecting Gemini without an API key raises ConfigurationError on validation."""
    settings = Settings(
        llm_provider="gemini",
        gemini_api_key=None,
        _env_file=None,
    )
    with pytest.raises(ConfigurationError) as exc_info:
        settings.validate_provider_credentials()
    assert "GEMINI_API_KEY environment variable is required" in str(exc_info.value)


def test_get_settings_helper_error_handling():
    """Verify that get_settings wraps Pydantic validation errors into ConfigurationError."""
    with pytest.raises(ConfigurationError):
        get_settings(findoc_chunk_size=50)  # below 100 character minimum
