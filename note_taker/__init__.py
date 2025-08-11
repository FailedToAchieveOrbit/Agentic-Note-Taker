"""Top-level package for the note_taker module.

This package exposes the two main classes used by the CLI:

- :class:`note_taker.database.NoteDatabase` manages the storage of
  notes and their embeddings.
- :class:`note_taker.embedding_model.EmbeddingModel` wraps
  embedding backends (OpenAI or Sentence‑Transformers).
"""

from .database import NoteDatabase  # noqa: F401
from .embedding_model import EmbeddingModel  # noqa: F401

__all__ = ["NoteDatabase", "EmbeddingModel"]