"""Robust Note Taker - AI-powered semantic search for your notes.

A modern Python application for storing and searching through notes using
semantic search, BM25, and hybrid search methods. Built with async/await,
type hints, and modern Python best practices.
"""

__version__ = "2.0.0"
__author__ = "FailedToAchieveOrbit"
__description__ = "AI-powered semantic search for your notes"

# Public API
from .config import Settings, get_settings, reset_settings
from .database import AsyncNoteDatabase
from .embeddings import EmbeddingService
from .models import (
    Note,
    NoteCreate,
    SearchResult,
    SearchQuery,
    DatabaseStats,
    EmbeddingInfo,
    SystemInfo,
    APIResponse,
    ErrorResponse,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__description__",
    
    # Configuration
    "Settings",
    "get_settings",
    "reset_settings",
    
    # Core classes
    "AsyncNoteDatabase",
    "EmbeddingService",
    
    # Models
    "Note",
    "NoteCreate", 
    "SearchResult",
    "SearchQuery",
    "DatabaseStats",
    "EmbeddingInfo",
    "SystemInfo",
    "APIResponse",
    "ErrorResponse",
]