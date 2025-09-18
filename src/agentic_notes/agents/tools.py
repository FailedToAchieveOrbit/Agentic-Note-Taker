"""🛠️ Agent tools for the agentic note system.

Implements LangChain-compatible tools for search, note management,
analysis, and web search capabilities.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..core.database import AgenticDatabase
from ..core.vector_store import VectorStore
from ..models.note import Note, NoteCreate, NoteUpdate
from ..models.search import SearchQuery


class SearchInput(BaseModel):
    """Input schema for search tool."""
    query: str = Field(description="Search query")
    method: str = Field(default="hybrid", description="Search method: semantic, bm25, or hybrid")
    top_k: int = Field(default=5, description="Number of results to return")
    threshold: float = Field(default=0.1, description="Minimum similarity threshold")


class SearchTool(BaseTool):
    """Tool for searching through notes."""
    
    name: str = "search"
    description: str = "Search through existing notes using semantic and keyword search"
    args_schema: Type[BaseModel] = SearchInput
    
    def __init__(self, vector_store: VectorStore, database: AgenticDatabase, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store
        self.database = database
    
    async def _arun(self, query: str, method: str = "hybrid", top_k: int = 5, threshold: float = 0.1) -> str:
        """Async implementation of the search tool."""
        try:
            search_query = SearchQuery(
                query=query,
                method=method,
                top_k=top_k,
                threshold=threshold
            )
            
            results = await self.vector_store.search(search_query)
            
            if not results:
                return f"No results found for query: {query}"
            
            # Format results
            formatted_results = []
            for result in results:
                note = result.note
                formatted_results.append(
                    f"Title: {note.title}\n"
                    f"Score: {result.score:.3f}\n"
                    f"Content: {note.content[:200]}...\n"
                    f"Tags: {', '.join(note.metadata.tags)}\n"
                    f"---"
                )
            
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    def _run(self, query: str, method: str = "hybrid", top_k: int = 5, threshold: float = 0.1) -> str:
        """Sync wrapper for async implementation."""
        return asyncio.run(self._arun(query, method, top_k, threshold))


class NoteInput(BaseModel):
    """Input schema for note tool."""
    action: str = Field(description="Action: create, update, delete, or get")
    note_id: Optional[str] = Field(default=None, description="Note ID for update/delete/get")
    title: Optional[str] = Field(default=None, description="Note title")
    content: Optional[str] = Field(default=None, description="Note content")
    tags: Optional[List[str]] = Field(default=None, description="Note tags")


class NoteTool(BaseTool):
    """Tool for managing notes."""
    
    name: str = "note"
    description: str = "Create, update, delete, or retrieve notes"
    args_schema: Type[BaseModel] = NoteInput
    
    def __init__(self, database: AgenticDatabase, **kwargs):
        super().__init__(**kwargs)
        self.database = database
    
    async def _arun(self, action: str, note_id: Optional[str] = None, 
                   title: Optional[str] = None, content: Optional[str] = None,
                   tags: Optional[List[str]] = None) -> str:
        """Async implementation of the note tool."""
        try:
            if action == "create":
                if not title or not content:
                    return "Error: Title and content are required for creating a note"
                
                note_create = NoteCreate(
                    title=title,
                    content=content
                )
                if tags:
                    note_create.metadata.tags = tags
                
                note = await self.database.create_note(note_create)
                return f"Created note '{note.title}' with ID: {note.id}"
            
            elif action == "update":
                if not note_id:
                    return "Error: Note ID is required for updating"
                
                update_data = NoteUpdate()
                if title:
                    update_data.title = title
                if content:
                    update_data.content = content
                if tags:
                    update_data.metadata = {"tags": tags}
                
                note = await self.database.update_note(note_id, update_data)
                if note:
                    return f"Updated note '{note.title}'"
                else:
                    return f"Note with ID {note_id} not found"
            
            elif action == "delete":
                if not note_id:
                    return "Error: Note ID is required for deletion"
                
                success = await self.database.delete_note(note_id)
                if success:
                    return f"Deleted note with ID: {note_id}"
                else:
                    return f"Note with ID {note_id} not found"
            
            elif action == "get":
                if not note_id:
                    return "Error: Note ID is required for retrieval"
                
                note = await self.database.get_note(note_id)
                if note:
                    return (
                        f"Title: {note.title}\n"
                        f"Content: {note.content}\n"
                        f"Tags: {', '.join(note.metadata.tags)}\n"
                        f"Created: {note.created_at}\n"
                        f"Updated: {note.updated_at}"
                    )
                else:
                    return f"Note with ID {note_id} not found"
            
            else:
                return f"Unknown action: {action}. Use create, update, delete, or get"
                
        except Exception as e:
            return f"Note operation failed: {str(e)}"
    
    def _run(self, action: str, note_id: Optional[str] = None, 
            title: Optional[str] = None, content: Optional[str] = None,
            tags: Optional[List[str]] = None) -> str:
        """Sync wrapper for async implementation."""
        return asyncio.run(self._arun(action, note_id, title, content, tags))


class AnalysisInput(BaseModel):
    """Input schema for analysis tool."""
    text: str = Field(description="Text to analyze")
    analysis_type: str = Field(default="all", description="Type of analysis: topics, entities, sentiment, or all")


class AnalysisTool(BaseTool):
    """Tool for analyzing text content."""
    
    name: str = "analysis"
    description: str = "Analyze text for topics, entities, sentiment, and other insights"
    args_schema: Type[BaseModel] = AnalysisInput
    
    async def _arun(self, text: str, analysis_type: str = "all") -> str:
        """Async implementation of the analysis tool."""
        try:
            results = {}
            
            if analysis_type in ["all", "topics"]:
                # Simple keyword extraction (in production, use NLP libraries)
                words = text.lower().split()
                word_freq = {}
                for word in words:
                    if len(word) > 3:
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                top_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                results["topics"] = [topic for topic, _ in top_topics]
            
            if analysis_type in ["all", "entities"]:
                # Simple entity extraction (in production, use NER)
                import re
                entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
                results["entities"] = list(set(entities))[:10]
            
            if analysis_type in ["all", "sentiment"]:
                # Simple sentiment analysis (in production, use sentiment models)
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
                
                text_lower = text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                if pos_count > neg_count:
                    sentiment = "positive"
                    score = 0.6
                elif neg_count > pos_count:
                    sentiment = "negative"
                    score = -0.6
                else:
                    sentiment = "neutral"
                    score = 0.0
                
                results["sentiment"] = {"label": sentiment, "score": score}
            
            # Format results
            formatted_results = []
            for key, value in results.items():
                if key == "sentiment":
                    formatted_results.append(f"{key.title()}: {value['label']} (score: {value['score']:.2f})")
                else:
                    formatted_results.append(f"{key.title()}: {', '.join(map(str, value))}")
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    def _run(self, text: str, analysis_type: str = "all") -> str:
        """Sync wrapper for async implementation."""
        return asyncio.run(self._arun(text, analysis_type))


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    query: str = Field(description="Web search query")
    num_results: int = Field(default=5, description="Number of results to return")


class WebSearchTool(BaseTool):
    """Tool for searching the web."""
    
    name: str = "web_search"
    description: str = "Search the web for additional information"
    args_schema: Type[BaseModel] = WebSearchInput
    
    async def _arun(self, query: str, num_results: int = 5) -> str:
        """Async implementation of the web search tool."""
        try:
            # In production, integrate with actual web search API
            # For now, return a placeholder
            return f"Web search results for '{query}':\n\n[This is a placeholder. In production, this would integrate with a web search API like Tavily, SerpAPI, or Google Custom Search to fetch real-time web results.]\n\nSuggested integration: Use Tavily Search API for real-time web search capabilities."
            
        except Exception as e:
            return f"Web search failed: {str(e)}"
    
    def _run(self, query: str, num_results: int = 5) -> str:
        """Sync wrapper for async implementation."""
        return asyncio.run(self._arun(query, num_results))