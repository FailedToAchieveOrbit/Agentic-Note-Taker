"""📋 Data models for the agentic note system.

Modern Pydantic v2 models with strict typing, validation, and serialization.
"""

from .agent import AgentMessage, AgentRole, AgentTask, Conversation
from .note import Note, NoteCreate, NoteUpdate, NoteMetadata
from .search import SearchQuery, SearchResult, SearchMethod, RetrievalResult
from .system import SystemHealth, SystemMetrics, SystemStatus

__all__ = [
    # Agent models
    "AgentMessage",
    "AgentRole", 
    "AgentTask",
    "Conversation",
    
    # Note models
    "Note",
    "NoteCreate",
    "NoteUpdate",
    "NoteMetadata",
    
    # Search models
    "SearchQuery",
    "SearchResult", 
    "SearchMethod",
    "RetrievalResult",
    
    # System models
    "SystemHealth",
    "SystemMetrics",
    "SystemStatus",
]