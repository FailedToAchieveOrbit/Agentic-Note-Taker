"""Pydantic models for the note-taking system.

This module defines the data models used throughout the application
with proper validation and serialization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class NoteCreate(BaseModel):
    """Model for creating a new note."""
    
    content: str = Field(
        ...,
        min_length=1,
        max_length=100_000,
        description="The content of the note"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags for the note"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the note"
    )
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate and clean note content."""
        return v.strip()
    
    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Validate and clean tags."""
        # Remove duplicates and clean tags
        cleaned_tags = []
        seen = set()
        
        for tag in v:
            clean_tag = tag.strip().lower()
            if clean_tag and clean_tag not in seen and len(clean_tag) <= 50:
                cleaned_tags.append(clean_tag)
                seen.add(clean_tag)
        
        return cleaned_tags[:10]  # Limit to 10 tags


class Note(BaseModel):
    """Model representing a note in the database."""
    
    id: str = Field(
        ...,
        description="Unique identifier for the note"
    )
    content: str = Field(
        ...,
        min_length=1,
        description="The content of the note"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags for the note"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the note"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the note was created"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the note was last updated"
    )
    
    class Config:
        # Allow datetime serialization
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override to ensure proper datetime serialization."""
        data = super().model_dump(**kwargs)
        
        # Ensure datetime fields are serialized as ISO strings
        if isinstance(data.get("created_at"), datetime):
            data["created_at"] = data["created_at"].isoformat()
        if isinstance(data.get("updated_at"), datetime):
            data["updated_at"] = data["updated_at"].isoformat()
        
        return data
    
    @property
    def word_count(self) -> int:
        """Count of words in the note content."""
        return len(self.content.split())
    
    @property
    def char_count(self) -> int:
        """Count of characters in the note content."""
        return len(self.content)
    
    def has_tag(self, tag: str) -> bool:
        """Check if note has a specific tag."""
        return tag.lower() in [t.lower() for t in self.tags]


class SearchResult(BaseModel):
    """Model representing a search result."""
    
    note: Note = Field(
        ...,
        description="The note that matched the search"
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score (0.0 to 1.0)"
    )
    method: Literal["semantic", "bm25", "hybrid"] = Field(
        ...,
        description="Search method used to find this result"
    )
    highlights: Optional[List[str]] = Field(
        default=None,
        description="Text snippets that matched the query"
    )


class SearchQuery(BaseModel):
    """Model for search queries."""
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The search query"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return"
    )
    method: Literal["semantic", "bm25", "hybrid"] = Field(
        default="hybrid",
        description="Search method to use"
    )
    threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )
    tag_filter: Optional[str] = Field(
        default=None,
        description="Optional tag to filter by"
    )
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and clean search query."""
        return v.strip()


class DatabaseStats(BaseModel):
    """Model for database statistics."""
    
    total_notes: int = Field(
        ...,
        ge=0,
        description="Total number of notes"
    )
    total_tags: int = Field(
        ...,
        ge=0,
        description="Total number of unique tags"
    )
    avg_length: float = Field(
        ...,
        ge=0.0,
        description="Average note length in characters"
    )
    oldest_note: str = Field(
        ...,
        description="Date of oldest note"
    )
    newest_note: str = Field(
        ...,
        description="Date of newest note"
    )
    db_size: str = Field(
        ...,
        description="Database file size (human readable)"
    )
    top_tags: List[tuple[str, int]] = Field(
        default_factory=list,
        description="Most frequently used tags"
    )


class EmbeddingInfo(BaseModel):
    """Model for embedding service information."""
    
    backend: Optional[str] = Field(
        default=None,
        description="Active embedding backend"
    )
    model: str = Field(
        ...,
        description="Model name being used"
    )
    dimension: Optional[int] = Field(
        default=None,
        ge=1,
        description="Embedding dimension"
    )
    cache_files: int = Field(
        default=0,
        ge=0,
        description="Number of cached embedding files"
    )
    cache_size_mb: float = Field(
        default=0.0,
        ge=0.0,
        description="Cache size in megabytes"
    )
    is_available: bool = Field(
        ...,
        description="Whether embedding service is available"
    )


class SystemInfo(BaseModel):
    """Model for system information."""
    
    version: str = Field(
        ...,
        description="Application version"
    )
    python_version: str = Field(
        ...,
        description="Python version"
    )
    platform: str = Field(
        ...,
        description="Operating system platform"
    )
    data_dir: str = Field(
        ...,
        description="Data directory path"
    )
    config_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration summary (without sensitive data)"
    )
    embedding_info: EmbeddingInfo = Field(
        ...,
        description="Embedding service information"
    )


class APIResponse(BaseModel):
    """Generic API response model."""
    
    success: bool = Field(
        ...,
        description="Whether the operation was successful"
    )
    message: Optional[str] = Field(
        default=None,
        description="Success or error message"
    )
    data: Optional[Any] = Field(
        default=None,
        description="Response data"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    
    success: bool = Field(
        default=False,
        description="Always false for error responses"
    )
    error: str = Field(
        ...,
        description="Error type or category"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp"
    )