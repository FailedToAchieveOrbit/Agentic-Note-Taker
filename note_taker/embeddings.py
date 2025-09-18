"""Modern embedding service with multiple backends and caching.

This module provides a unified interface for text embeddings with support
for OpenAI, local sentence-transformers, and caching for performance.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .config import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Modern embedding service with multiple backends and caching.
    
    Features:
    - Multiple embedding backends (OpenAI, local models)
    - Async batch processing
    - Disk caching for embeddings
    - Automatic fallback between backends
    """
    
    def __init__(self, settings: Settings):
        """Initialize the embedding service.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.cache_dir = settings.embeddings_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Runtime state
        self._openai_client: Optional[object] = None
        self._local_model: Optional[object] = None
        self._backend: Optional[str] = None
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_embeddings)
        
        # Initialize backend
        asyncio.create_task(self._initialize_backend())
    
    async def _initialize_backend(self) -> None:
        """Initialize the embedding backend."""
        # Try OpenAI first if available
        if self.settings.has_openai_key:
            try:
                from openai import AsyncOpenAI
                self._openai_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
                self._backend = "openai"
                logger.info(f"Initialized OpenAI embeddings with model {self.settings.openai_embedding_model}")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        # Fallback to local model
        try:
            from sentence_transformers import SentenceTransformer
            # Load in thread pool to avoid blocking
            self._local_model = await asyncio.to_thread(
                SentenceTransformer, self.settings.embedding_model
            )
            self._backend = "local"
            logger.info(f"Initialized local embeddings with model {self.settings.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding backend: {e}")
            self._backend = None
    
    def is_available(self) -> bool:
        """Check if embedding service is available."""
        return self._backend is not None
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"{self._backend}_{self.settings.embedding_model}_{text_hash}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for cache key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    async def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from cache."""
        if not self.settings.enable_caching:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                embedding = pickle.load(f)
                return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to load embedding from cache: {e}")
            # Remove corrupted cache file
            try:
                cache_path.unlink()
            except Exception:
                pass
            return None
    
    async def _save_to_cache(self, cache_key: str, embedding: np.ndarray) -> None:
        """Save embedding to cache."""
        if not self.settings.enable_caching:
            return
        
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding.astype(np.float32), f)
        except Exception as e:
            logger.warning(f"Failed to save embedding to cache: {e}")
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not self.is_available():
            raise RuntimeError("No embedding backend available")
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        cached_embedding = await self._load_from_cache(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate embedding
        async with self._semaphore:
            if self._backend == "openai":
                embedding = await self._embed_openai([text])
                result = embedding[0]
            elif self._backend == "local":
                embedding = await self._embed_local([text])
                result = embedding[0]
            else:
                raise RuntimeError(f"Unknown backend: {self._backend}")
        
        # Cache result
        await self._save_to_cache(cache_key, result)
        
        return result
    
    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Embedding matrix as numpy array (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, -1)
        
        if not self.is_available():
            raise RuntimeError("No embedding backend available")
        
        # Check cache for each text
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached_embedding = await self._load_from_cache(cache_key)
            if cached_embedding is not None:
                embeddings.append((i, cached_embedding))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts in batches
        if uncached_texts:
            batch_size = self.settings.embedding_batch_size
            new_embeddings = []
            
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i + batch_size]
                
                async with self._semaphore:
                    if self._backend == "openai":
                        batch_embeddings = await self._embed_openai(batch)
                    elif self._backend == "local":
                        batch_embeddings = await self._embed_local(batch)
                    else:
                        raise RuntimeError(f"Unknown backend: {self._backend}")
                
                new_embeddings.extend(batch_embeddings)
            
            # Cache new embeddings and add to results
            for text, embedding, orig_idx in zip(uncached_texts, new_embeddings, uncached_indices):
                cache_key = self._get_cache_key(text)
                await self._save_to_cache(cache_key, embedding)
                embeddings.append((orig_idx, embedding))
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings], dtype=np.float32)
        
        return result
    
    async def _embed_openai(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using OpenAI API."""
        try:
            response = await self._openai_client.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=texts,
            )
            
            # Extract embeddings
            embeddings = []
            for item in response.data:
                embedding = np.array(item.embedding, dtype=np.float32)
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise RuntimeError(f"Failed to generate OpenAI embeddings: {e}")
    
    async def _embed_local(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using local model."""
        try:
            # Run in thread pool to avoid blocking
            embeddings = await asyncio.to_thread(
                self._local_model.encode,
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            
            # Convert to list of arrays
            if len(texts) == 1:
                embeddings = [embeddings]
            else:
                embeddings = [embeddings[i] for i in range(len(embeddings))]
            
            return [np.array(emb, dtype=np.float32) for emb in embeddings]
            
        except Exception as e:
            logger.error(f"Local embedding error: {e}")
            raise RuntimeError(f"Failed to generate local embeddings: {e}")
    
    async def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        if not self.is_available():
            raise RuntimeError("No embedding backend available")
        
        # Generate a test embedding to determine dimension
        test_embedding = await self.embed_text("test")
        return test_embedding.shape[0]
    
    async def clear_cache(self) -> int:
        """Clear all cached embeddings.
        
        Returns:
            Number of cache files removed
        """
        if not self.cache_dir.exists():
            return 0
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        removed_count = 0
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {removed_count} embedding cache files")
        return removed_count
    
    def get_cache_info(self) -> dict:
        """Get information about the embedding cache."""
        if not self.cache_dir.exists():
            return {
                "cache_files": 0,
                "cache_size": 0,
                "cache_size_mb": 0.0,
            }
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cache_files": len(cache_files),
            "cache_size": total_size,
            "cache_size_mb": total_size / (1024 * 1024),
        }