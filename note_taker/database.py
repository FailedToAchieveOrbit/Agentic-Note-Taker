"""Modern async database with multiple search methods and caching.

This module provides an efficient async database layer with support for
semantic search, BM25, and hybrid search methods with caching and
performance optimizations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple, Union

import aiofiles
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Settings
from .embeddings import EmbeddingService
from .models import Note, NoteCreate, SearchResult

logger = logging.getLogger(__name__)

SearchMethod = Literal["semantic", "bm25", "hybrid"]


class AsyncNoteDatabase:
    """Modern async note database with multiple search methods.
    
    Features:
    - Async I/O operations
    - Multiple search methods (semantic, BM25, hybrid)
    - Embedding caching
    - Background indexing
    - Automatic backups
    """
    
    def __init__(self, data_dir: Union[str, Path], settings: Optional[Settings] = None):
        """Initialize the async database.
        
        Args:
            data_dir: Directory to store database files
            settings: Configuration settings
        """
        if settings is None:
            from .config import get_settings
            settings = get_settings()
        
        self.settings = settings
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.notes_file = self.data_dir / "notes.json"
        self.embeddings_file = self.data_dir / "embeddings.npz"
        self.index_file = self.data_dir / "search_index.json"
        
        # Runtime state
        self._notes: List[Note] = []
        self._embeddings: Optional[np.ndarray] = None
        self._bm25_index: Optional[BM25Okapi] = None
        self._last_modified: Optional[float] = None
        self._lock = asyncio.Lock()
        
        # Services
        self.embedding_service = EmbeddingService(settings)
        
        # Cache for search results
        self._search_cache: Dict[str, Tuple[List[SearchResult], float]] = {}
    
    async def __aenter__(self) -> AsyncNoteDatabase:
        """Async context manager entry."""
        await self._load_data()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self.settings.auto_backup and exc_type is None:
            await self._create_backup()
    
    async def _load_data(self) -> None:
        """Load notes and indexes from disk."""
        async with self._lock:
            try:
                if self.notes_file.exists():
                    async with aiofiles.open(self.notes_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        data = json.loads(content)
                        self._notes = [Note(**note_data) for note_data in data.get("notes", [])]
                    
                    self._last_modified = self.notes_file.stat().st_mtime
                else:
                    self._notes = []
                
                # Load embeddings if available
                if self.embeddings_file.exists() and self._notes:
                    try:
                        embeddings_data = np.load(self.embeddings_file)
                        if 'embeddings' in embeddings_data and len(embeddings_data['embeddings']) == len(self._notes):
                            self._embeddings = embeddings_data['embeddings']
                    except Exception as e:
                        logger.warning(f"Failed to load embeddings: {e}")
                        self._embeddings = None
                
                # Build BM25 index
                await self._rebuild_bm25_index()
                
                logger.info(f"Loaded {len(self._notes)} notes")
                
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                self._notes = []
    
    async def _save_data(self) -> None:
        """Save notes to disk."""
        data = {
            "notes": [note.model_dump() for note in self._notes],
            "version": "2.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Atomic write
        temp_file = self.notes_file.with_suffix(".tmp")
        async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=2, ensure_ascii=False))
        
        # Atomic move
        temp_file.replace(self.notes_file)
        self._last_modified = time.time()
    
    async def _save_embeddings(self) -> None:
        """Save embeddings to disk."""
        if self._embeddings is not None:
            np.savez_compressed(
                self.embeddings_file,
                embeddings=self._embeddings,
                version="2.0",
                updated_at=time.time(),
            )
    
    async def _rebuild_bm25_index(self) -> None:
        """Rebuild the BM25 search index."""
        if not self._notes:
            self._bm25_index = None
            return
        
        # Tokenize documents for BM25
        documents = []
        for note in self._notes:
            # Simple tokenization (in production, use proper tokenizer)
            tokens = note.content.lower().split()
            documents.append(tokens)
        
        self._bm25_index = BM25Okapi(documents)
        logger.debug(f"Built BM25 index for {len(documents)} documents")
    
    async def _ensure_embeddings(self, force_rebuild: bool = False) -> None:
        """Ensure embeddings are available for all notes."""
        if not self._notes:
            self._embeddings = None
            return
        
        if self._embeddings is not None and len(self._embeddings) == len(self._notes) and not force_rebuild:
            return
        
        logger.info("Generating embeddings for notes...")
        
        # Extract content for embedding
        texts = [note.content for note in self._notes]
        
        # Generate embeddings
        self._embeddings = await self.embedding_service.embed_batch(texts)
        
        # Save to disk
        await self._save_embeddings()
        
        logger.info(f"Generated embeddings for {len(texts)} notes")
    
    async def add_note(
        self, 
        content: str, 
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new note to the database.
        
        Args:
            content: Note content
            tags: Optional list of tags
            metadata: Optional metadata dictionary
            
        Returns:
            Note ID
        """
        async with self._lock:
            note = Note(
                id=str(uuid.uuid4()),
                content=content.strip(),
                tags=tags or [],
                metadata=metadata or {},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            
            self._notes.append(note)
            
            # Generate embedding for the new note
            if self.embedding_service.is_available():
                embedding = await self.embedding_service.embed_text(content)
                if self._embeddings is None:
                    self._embeddings = embedding.reshape(1, -1)
                else:
                    self._embeddings = np.vstack([self._embeddings, embedding])
            
            # Rebuild BM25 index
            await self._rebuild_bm25_index()
            
            # Save to disk
            await self._save_data()
            await self._save_embeddings()
            
            # Clear search cache
            self._search_cache.clear()
            
            logger.info(f"Added note {note.id[:8]}... with {len(content)} characters")
            
            return note.id
    
    async def list_notes(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        tag_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List notes with optional filtering.
        
        Args:
            limit: Maximum number of notes to return
            offset: Number of notes to skip
            tag_filter: Filter by tag
            
        Returns:
            List of note dictionaries
        """
        notes = self._notes.copy()
        
        # Filter by tag
        if tag_filter:
            notes = [note for note in notes if tag_filter.lower() in [tag.lower() for tag in note.tags]]
        
        # Sort by creation date (newest first)
        notes.sort(key=lambda n: n.created_at, reverse=True)
        
        # Apply offset and limit
        if offset:
            notes = notes[offset:]
        if limit:
            notes = notes[:limit]
        
        return [note.model_dump() for note in notes]
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        method: SearchMethod = "hybrid",
        threshold: float = 0.1
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search notes using specified method.
        
        Args:
            query: Search query
            top_k: Number of results to return
            method: Search method to use
            threshold: Minimum similarity threshold
            
        Returns:
            List of (note_dict, score) tuples
        """
        if not self._notes:
            return []
        
        # Check cache
        cache_key = f"{method}:{query}:{top_k}:{threshold}"
        if cache_key in self._search_cache:
            cached_results, cache_time = self._search_cache[cache_key]
            if time.time() - cache_time < self.settings.cache_ttl:
                return [(result.note.model_dump(), result.score) for result in cached_results]
        
        # Perform search based on method
        if method == "semantic":
            results = await self._semantic_search(query, top_k, threshold)
        elif method == "bm25":
            results = await self._bm25_search(query, top_k, threshold)
        elif method == "hybrid":
            results = await self._hybrid_search(query, top_k, threshold)
        else:
            raise ValueError(f"Unknown search method: {method}")
        
        # Cache results
        if self.settings.enable_caching:
            self._search_cache[cache_key] = (results, time.time())
        
        return [(result.note.model_dump(), result.score) for result in results]
    
    async def _semantic_search(
        self, query: str, top_k: int, threshold: float
    ) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        await self._ensure_embeddings()
        
        if self._embeddings is None:
            logger.warning("Embeddings not available for semantic search")
            return []
        
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self._embeddings).flatten()
        
        # Get top results above threshold
        results = []
        for idx, score in enumerate(similarities):
            if score >= threshold:
                results.append(SearchResult(
                    note=self._notes[idx],
                    score=float(score),
                    method="semantic"
                ))
        
        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    
    async def _bm25_search(
        self, query: str, top_k: int, threshold: float
    ) -> List[SearchResult]:
        """Perform BM25 search."""
        if self._bm25_index is None:
            logger.warning("BM25 index not available")
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self._bm25_index.get_scores(query_tokens)
        
        # Normalize scores to [0, 1] range
        if len(scores) > 0:
            max_score = max(scores)
            if max_score > 0:
                scores = scores / max_score
        
        # Get top results above threshold
        results = []
        for idx, score in enumerate(scores):
            if score >= threshold:
                results.append(SearchResult(
                    note=self._notes[idx],
                    score=float(score),
                    method="bm25"
                ))
        
        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    
    async def _hybrid_search(
        self, query: str, top_k: int, threshold: float
    ) -> List[SearchResult]:
        """Perform hybrid search combining semantic and BM25."""
        # Get results from both methods
        semantic_results = await self._semantic_search(query, top_k * 2, 0.0)
        bm25_results = await self._bm25_search(query, top_k * 2, 0.0)
        
        # Create score maps
        semantic_scores = {r.note.id: r.score for r in semantic_results}
        bm25_scores = {r.note.id: r.score for r in bm25_results}
        
        # Combine scores
        all_note_ids = set(semantic_scores.keys()) | set(bm25_scores.keys())
        results = []
        
        for note_id in all_note_ids:
            semantic_score = semantic_scores.get(note_id, 0.0)
            bm25_score = bm25_scores.get(note_id, 0.0)
            
            # Weighted combination
            combined_score = (
                self.settings.semantic_weight * semantic_score +
                self.settings.bm25_weight * bm25_score
            )
            
            if combined_score >= threshold:
                # Find the note
                note = next(n for n in self._notes if n.id == note_id)
                results.append(SearchResult(
                    note=note,
                    score=float(combined_score),
                    method="hybrid"
                ))
        
        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    
    async def generate_answer(
        self, query: str, search_results: List[Tuple[Dict[str, Any], float]]
    ) -> Optional[str]:
        """Generate an AI-powered answer from search results."""
        if not self.settings.has_openai_key or not search_results:
            return None
        
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            
            # Prepare context from search results
            context_parts = []
            for i, (note, score) in enumerate(search_results[:3], 1):
                context_parts.append(f"Note {i} (relevance: {score:.2f}): {note['content']}")
            
            context = "\n\n".join(context_parts)
            
            # Create prompt
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers questions based on provided notes. "
                        "Use only the information contained in the notes to answer the question. "
                        "If the notes don't contain relevant information, say you don't know. "
                        "Keep your answer concise and directly address the question."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Notes:\n{context}\n\nQuestion: {query}\n\nAnswer:",
                },
            ]
            
            # Get response
            response = await client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                max_tokens=self.settings.openai_max_tokens,
                temperature=self.settings.openai_temperature,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating AI answer: {e}")
            return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self._notes:
            return {
                "total_notes": 0,
                "total_tags": 0,
                "avg_length": 0,
                "oldest_note": "N/A",
                "newest_note": "N/A",
                "db_size": "0 B",
                "top_tags": [],
            }
        
        # Calculate stats
        total_notes = len(self._notes)
        all_tags = []
        total_length = 0
        
        for note in self._notes:
            all_tags.extend(note.tags)
            total_length += len(note.content)
        
        # Tag frequency
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Date range
        dates = [note.created_at for note in self._notes]
        oldest = min(dates).strftime("%Y-%m-%d %H:%M") if dates else "N/A"
        newest = max(dates).strftime("%Y-%m-%d %H:%M") if dates else "N/A"
        
        # File size
        db_size = "N/A"
        if self.notes_file.exists():
            size_bytes = self.notes_file.stat().st_size
            if size_bytes < 1024:
                db_size = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                db_size = f"{size_bytes / 1024:.1f} KB"
            else:
                db_size = f"{size_bytes / (1024 * 1024):.1f} MB"
        
        return {
            "total_notes": total_notes,
            "total_tags": len(tag_counts),
            "avg_length": total_length / total_notes if total_notes > 0 else 0,
            "oldest_note": oldest,
            "newest_note": newest,
            "db_size": db_size,
            "top_tags": top_tags,
        }
    
    async def _create_backup(self) -> None:
        """Create a backup of the database."""
        if not self.settings.auto_backup:
            return
        
        try:
            backup_dir = self.settings.backups_dir
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"notes_backup_{timestamp}.json"
            
            # Copy current database
            if self.notes_file.exists():
                async with aiofiles.open(self.notes_file, 'r') as src:
                    content = await src.read()
                    async with aiofiles.open(backup_file, 'w') as dst:
                        await dst.write(content)
                
                logger.info(f"Created backup: {backup_file.name}")
                
                # Clean old backups
                await self._cleanup_old_backups()
        
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
    
    async def _cleanup_old_backups(self) -> None:
        """Remove old backup files."""
        backup_dir = self.settings.backups_dir
        
        if not backup_dir.exists():
            return
        
        # Get all backup files
        backup_files = list(backup_dir.glob("notes_backup_*.json"))
        backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Keep only the most recent backups
        files_to_remove = backup_files[self.settings.max_backups:]
        
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                logger.debug(f"Removed old backup: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to remove backup {file_path.name}: {e}")