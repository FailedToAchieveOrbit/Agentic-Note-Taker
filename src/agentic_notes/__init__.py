"""🤖 Agentic Note Taker - AI-powered collaborative note management.

A modern agentic system that uses multiple AI agents to intelligently manage,
search, and interact with your notes using the latest RAG techniques.

## Features
- 🗣️ Multi-agent collaboration with CrewAI
- 🌐 Agentic workflows with LangGraph
- 🔍 Hybrid semantic + BM25 search
- ⚡ FastAPI async backend
- 💾 Vector database integration
- 📊 Real-time monitoring and observability
"""

__version__ = "3.0.0"
__author__ = "FailedToAchieveOrbit"
__description__ = "Agentic RAG-powered note-taking system"

# Public API exports
from .config import Settings, get_settings
from .core.database import AgenticDatabase
from .core.vector_store import VectorStore
from .models.note import Note, NoteCreate, NoteUpdate
from .models.search import SearchQuery, SearchResult

__all__ = [
    "__version__",
    "__author__", 
    "__description__",
    "Settings",
    "get_settings",
    "AgenticDatabase",
    "VectorStore",
    "Note",
    "NoteCreate",
    "NoteUpdate",
    "SearchQuery",
    "SearchResult",
]