"""Configuration management for FinDoc AI.

Loads environment variables, validates runtime settings, and provides safe defaults.
"""

from pathlib import Path
from typing import Literal, Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from findoc.errors import ConfigurationError


class Settings(BaseSettings):
    """FinDoc AI application settings."""

    # LLM Provider Configuration
    llm_provider: Literal["mock", "gemini"] = Field(
        default="mock",
        description="LLM provider: 'mock' (no API key needed) or 'gemini'",
    )
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key. Required only when llm_provider='gemini'",
    )
    llm_model: str = Field(
        default="gemini-2.0-flash",
        description="Configured LLM model identifier (e.g. gemini-2.0-flash, gemini-1.5-flash)",
    )

    # Local Storage Paths
    findoc_data_dir: Path = Field(
        default=Path("./data"),
        description="Root directory for local persistent data (chroma, metadata, uploads)",
    )

    # Ingestion Constraints
    findoc_max_file_mb: int = Field(
        default=20,
        description="Maximum allowed size per uploaded file in megabytes",
    )
    findoc_max_files_per_upload: int = Field(
        default=10,
        description="Maximum number of files allowed in a single upload batch",
    )

    # Text Chunking Settings
    findoc_chunk_size: int = Field(
        default=1000,
        description="Target chunk size in characters",
    )
    findoc_chunk_overlap: int = Field(
        default=150,
        description="Overlap between adjacent chunks in characters",
    )

    # Retrieval Settings
    findoc_top_k: int = Field(
        default=5,
        description="Default number of chunks to retrieve per query",
    )

    # Embedding Model
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace local embedding model name or path",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("findoc_max_file_mb", "findoc_max_files_per_upload", "findoc_top_k")
    @classmethod
    def validate_positive_integers(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer, got {v}")
        return v

    @field_validator("findoc_chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v < 100:
            raise ValueError(f"findoc_chunk_size must be at least 100 characters, got {v}")
        return v

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.findoc_chunk_overlap >= self.findoc_chunk_size:
            raise ValueError(
                f"findoc_chunk_overlap ({self.findoc_chunk_overlap}) must be strictly less "
                f"than findoc_chunk_size ({self.findoc_chunk_size})"
            )
        if self.findoc_chunk_overlap < 0:
            raise ValueError(
                f"findoc_chunk_overlap must be non-negative, got {self.findoc_chunk_overlap}"
            )
        return self

    @property
    def chroma_dir(self) -> Path:
        """Directory for persistent Chroma vector store."""
        return self.findoc_data_dir / "chroma"

    @property
    def metadata_db_path(self) -> Path:
        """Path for local SQLite metadata database."""
        return self.findoc_data_dir / "app_metadata.sqlite"

    @property
    def uploads_dir(self) -> Path:
        """Directory for secure local file staging."""
        return self.findoc_data_dir / "uploads"

    def ensure_directories(self) -> None:
        """Create necessary application data directories if they do not exist."""
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def validate_provider_credentials(self) -> None:
        """Verify that provider credentials exist if live mode is requested."""
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY environment variable is required when LLM_PROVIDER is set to 'gemini'."
            )


def get_settings(**kwargs) -> Settings:
    """Helper to instantiate validated application settings."""
    try:
        return Settings(**kwargs)
    except ValueError as e:
        raise ConfigurationError(f"Configuration validation failed: {e}") from e
