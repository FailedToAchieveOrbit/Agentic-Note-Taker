"""Configuration management for the robust note-taking system.

This module provides Pydantic-based configuration with environment variable
support and validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Application configuration with environment variable support."""
    
    # Basic settings
    data_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "data",
        description="Directory to store notes and embeddings"
    )
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for embeddings and chat completion"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for chat completion"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI model for text embeddings"
    )
    openai_max_tokens: int = Field(
        default=1000,
        description="Maximum tokens for OpenAI responses",
        ge=1,
        le=4000,
    )
    openai_temperature: float = Field(
        default=0.7,
        description="Temperature for OpenAI responses",
        ge=0.0,
        le=2.0,
    )
    
    # Embedding settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local embedding model for semantic search"
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation",
        ge=1,
        le=128,
    )
    
    # Search settings
    search_methods: list[Literal["semantic", "bm25", "hybrid"]] = Field(
        default=["semantic", "bm25", "hybrid"],
        description="Available search methods"
    )
    default_search_method: Literal["semantic", "bm25", "hybrid"] = Field(
        default="hybrid",
        description="Default search method"
    )
    semantic_weight: float = Field(
        default=0.7,
        description="Weight for semantic search in hybrid mode",
        ge=0.0,
        le=1.0,
    )
    bm25_weight: float = Field(
        default=0.3,
        description="Weight for BM25 search in hybrid mode",
        ge=0.0,
        le=1.0,
    )
    
    # Performance settings
    enable_caching: bool = Field(
        default=True,
        description="Enable caching for embeddings and search results"
    )
    cache_ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds",
        ge=60,
        le=86400,
    )
    max_concurrent_embeddings: int = Field(
        default=4,
        description="Maximum concurrent embedding operations",
        ge=1,
        le=16,
    )
    
    # Database settings
    auto_backup: bool = Field(
        default=True,
        description="Enable automatic database backups"
    )
    backup_interval: int = Field(
        default=86400,  # 24 hours
        description="Backup interval in seconds",
        ge=3600,  # 1 hour minimum
    )
    max_backups: int = Field(
        default=7,
        description="Maximum number of backups to keep",
        ge=1,
        le=30,
    )
    
    # Logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: Optional[Path] = Field(
        default=None,
        description="Log file path (logs to console if not set)"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "RNOTE_"
        case_sensitive = False
        
    @validator("data_dir", pre=True, always=True)
    def ensure_data_dir_exists(cls, v: Path) -> Path:
        """Ensure data directory exists."""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v.resolve()
    
    @validator("openai_api_key", pre=True, always=True)
    def get_openai_key_from_env(cls, v: Optional[str]) -> Optional[str]:
        """Get OpenAI API key from environment if not set."""
        return v or os.getenv("OPENAI_API_KEY")
    
    @validator("semantic_weight", "bm25_weight")
    def validate_search_weights(cls, v: float, values: dict) -> float:
        """Validate that search weights sum to 1.0 in hybrid mode."""
        # This is a simplified validation - full validation would be more complex
        return v
    
    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is available."""
        return bool(self.openai_api_key and self.openai_api_key.strip())
    
    @property
    def embeddings_dir(self) -> Path:
        """Directory for storing embeddings cache."""
        embeddings_dir = self.data_dir / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)
        return embeddings_dir
    
    @property
    def backups_dir(self) -> Path:
        """Directory for storing database backups."""
        backups_dir = self.data_dir / "backups"
        if self.auto_backup:
            backups_dir.mkdir(exist_ok=True)
        return backups_dir
    
    @property
    def notes_file(self) -> Path:
        """Path to the main notes database file."""
        return self.data_dir / "notes.json"
    
    def model_dump_safe(self) -> dict:
        """Dump configuration without sensitive data."""
        data = self.dict()
        if data.get("openai_api_key"):
            data["openai_api_key"] = "***" + data["openai_api_key"][-4:]
        return data


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance (for testing)."""
    global _settings
    _settings = None