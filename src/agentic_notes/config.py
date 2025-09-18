"""🔧 Configuration management for the agentic note system.

Modern configuration using Pydantic Settings with environment variable support,
validation, and structured configuration for all system components.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    """Available log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class VectorDBProvider(str, Enum):
    """Supported vector database providers."""
    QDRANT = "qdrant"
    CHROMA = "chroma"
    MILVUS = "milvus"
    INMEMORY = "inmemory"  # For testing


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LITELLM = "litellm"
    LOCAL = "local"


class AgentFramework(str, Enum):
    """Supported agent frameworks."""
    LANGGRAPH = "langgraph"
    CREWAI = "crewai"
    SWARM = "swarm"


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    model_config = SettingsConfigDict(env_prefix="DB_")
    
    data_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "data",
        description="Directory for storing application data"
    )
    backup_enabled: bool = Field(
        default=True,
        description="Enable automatic database backups"
    )
    backup_interval_hours: int = Field(
        default=6,
        ge=1,
        le=168,
        description="Backup interval in hours (1-168)"
    )
    max_backups: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of backups to keep"
    )
    
    @field_validator("data_dir")
    @classmethod
    def ensure_data_dir_exists(cls, v: Path) -> Path:
        """Ensure data directory exists."""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v.resolve()


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""
    
    model_config = SettingsConfigDict(env_prefix="VECTOR_")
    
    provider: VectorDBProvider = Field(
        default=VectorDBProvider.QDRANT,
        description="Vector database provider"
    )
    url: str = Field(
        default="http://localhost:6333",
        description="Vector database URL"
    )
    collection_name: str = Field(
        default="agentic_notes",
        description="Collection name for notes"
    )
    dimension: int = Field(
        default=384,
        ge=128,
        le=4096,
        description="Embedding dimension"
    )
    distance_metric: Literal["cosine", "euclidean", "dot"] = Field(
        default="cosine",
        description="Distance metric for similarity search"
    )
    enable_indexing: bool = Field(
        default=True,
        description="Enable vector indexing for faster search"
    )
    

class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""
    
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")
    
    provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.HUGGINGFACE,
        description="Embedding provider"
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for embedding generation"
    )
    max_concurrent: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum concurrent embedding requests"
    )
    cache_embeddings: bool = Field(
        default=True,
        description="Enable embedding caching"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="Embedding cache TTL in seconds"
    )


class LLMSettings(BaseSettings):
    """Large Language Model configuration."""
    
    model_config = SettingsConfigDict(env_prefix="LLM_")
    
    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="LLM provider"
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="LLM model name"
    )
    max_tokens: int = Field(
        default=2048,
        ge=100,
        le=32000,
        description="Maximum tokens per response"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation"
    )
    timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of retries"
    )


class AgentSettings(BaseSettings):
    """Agent system configuration."""
    
    model_config = SettingsConfigDict(env_prefix="AGENT_")
    
    framework: AgentFramework = Field(
        default=AgentFramework.LANGGRAPH,
        description="Agent framework to use"
    )
    enable_memory: bool = Field(
        default=True,
        description="Enable agent memory between interactions"
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum agent iterations per task"
    )
    enable_reflection: bool = Field(
        default=True,
        description="Enable agent self-reflection and improvement"
    )
    enable_collaboration: bool = Field(
        default=True,
        description="Enable multi-agent collaboration"
    )
    execution_timeout_seconds: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Agent execution timeout in seconds"
    )


class SearchSettings(BaseSettings):
    """Search configuration."""
    
    model_config = SettingsConfigDict(env_prefix="SEARCH_")
    
    default_method: Literal["semantic", "bm25", "hybrid"] = Field(
        default="hybrid",
        description="Default search method"
    )
    semantic_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for semantic search in hybrid mode"
    )
    bm25_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 search in hybrid mode"
    )
    min_similarity_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for results"
    )
    max_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum search results to return"
    )
    enable_reranking: bool = Field(
        default=True,
        description="Enable result reranking"
    )


class APISettings(BaseSettings):
    """API configuration."""
    
    model_config = SettingsConfigDict(env_prefix="API_")
    
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    huggingface_token: Optional[str] = Field(
        default=None,
        description="Hugging Face token"
    )
    langfuse_public_key: Optional[str] = Field(
        default=None,
        description="Langfuse public key for observability"
    )
    langfuse_secret_key: Optional[str] = Field(
        default=None,
        description="Langfuse secret key"
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse host URL"
    )
    
    @field_validator("openai_api_key", mode="before")
    @classmethod
    def get_openai_key_from_env(cls, v: Optional[str]) -> Optional[str]:
        """Get OpenAI API key from environment if not set."""
        return v or os.getenv("OPENAI_API_KEY")
    
    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def get_anthropic_key_from_env(cls, v: Optional[str]) -> Optional[str]:
        """Get Anthropic API key from environment if not set."""
        return v or os.getenv("ANTHROPIC_API_KEY")


class ServerSettings(BaseSettings):
    """Server configuration."""
    
    model_config = SettingsConfigDict(env_prefix="SERVER_")
    
    host: str = Field(
        default="127.0.0.1",
        description="Server host"
    )
    port: int = Field(
        default=8000,
        ge=1000,
        le=65535,
        description="Server port"
    )
    workers: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of worker processes"
    )
    reload: bool = Field(
        default=True,
        description="Enable auto-reload in development"
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Server log level"
    )


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""
    
    model_config = SettingsConfigDict(env_prefix="MONITORING_")
    
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    metrics_port: int = Field(
        default=9090,
        ge=1000,
        le=65535,
        description="Metrics server port"
    )
    enable_tracing: bool = Field(
        default=True,
        description="Enable distributed tracing"
    )
    enable_langfuse: bool = Field(
        default=True,
        description="Enable Langfuse observability"
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Application log level"
    )
    structured_logging: bool = Field(
        default=True,
        description="Enable structured JSON logging"
    )


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application metadata
    app_name: str = Field(
        default="Agentic Note Taker",
        description="Application name"
    )
    app_version: str = Field(
        default="3.0.0",
        description="Application version"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    environment: Literal["development", "testing", "production"] = Field(
        default="development",
        description="Application environment"
    )
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    api: APISettings = Field(default_factory=APISettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    @model_validator(mode="after")
    def validate_search_weights(self) -> Settings:
        """Ensure search weights sum to 1.0 in hybrid mode."""
        total_weight = self.search.semantic_weight + self.search.bm25_weight
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(
                f"Search weights must sum to 1.0, got {total_weight}"
            )
        return self
    
    @computed_field
    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is available."""
        return bool(self.api.openai_api_key and self.api.openai_api_key.strip())
    
    @computed_field
    @property
    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is available."""
        return bool(self.api.anthropic_api_key and self.api.anthropic_api_key.strip())
    
    @computed_field
    @property
    def has_langfuse_config(self) -> bool:
        """Check if Langfuse configuration is complete."""
        return bool(
            self.api.langfuse_public_key
            and self.api.langfuse_secret_key
            and self.api.langfuse_host
        )
    
    def model_dump_safe(self) -> Dict[str, Any]:
        """Dump configuration without sensitive data."""
        data = self.model_dump()
        
        # Redact API keys
        if data.get("api", {}).get("openai_api_key"):
            data["api"]["openai_api_key"] = "***" + data["api"]["openai_api_key"][-4:]
        if data.get("api", {}).get("anthropic_api_key"):
            data["api"]["anthropic_api_key"] = "***" + data["api"]["anthropic_api_key"][-4:]
        if data.get("api", {}).get("langfuse_secret_key"):
            data["api"]["langfuse_secret_key"] = "***" + data["api"]["langfuse_secret_key"][-4:]
        
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
    """Reset the global settings instance (mainly for testing)."""
    global _settings
    _settings = None