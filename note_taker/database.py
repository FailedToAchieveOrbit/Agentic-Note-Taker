"""Database layer for storing notes and their embeddings.

Notes are persisted as JSON in ``data/notes.json``.  Embeddings are
stored alongside the notes in the same file.  This simple storage
format is adequate for small to medium‑sized collections.  Should you
need to scale beyond a few thousand notes, consider replacing this
with a proper vector store (e.g. FAISS, Weaviate, etc.).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import List, Tuple

import numpy as np


class NoteDatabase:
    """A minimal persistent store for notes.

    Notes are saved as a list of dictionaries with unique IDs and textual
    content.  For retrieval, the database computes TF‑IDF vectors on
    the fly rather than persisting embeddings.  This design avoids
    external dependencies such as OpenAI or sentence‑transformers while
    still providing meaningful similarity measurements.
    """

    def __init__(self, data_dir: str = "data") -> None:
        """Initialise the database.

        Parameters
        ----------
        data_dir: str
            Directory in which to store the ``notes.json`` file.  Will
            be created if it does not exist.
        """
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.data_file = os.path.join(self.data_dir, "notes.json")
        self.notes: List[dict] = []  # list of dicts with 'id' and 'content'
        # embeddings are no longer persisted; search uses TF‑IDF on the fly
        self._load()

    def _load(self) -> None:
        """Load notes from disk if present."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.notes = data.get("notes", [])
            except Exception:
                # on any error, fall back to empty state
                self.notes = []

    def _save(self) -> None:
        """Persist the notes to disk."""
        data = {
            "notes": self.notes,
        }
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def add_note(self, content: str) -> None:
        """Append a new note to the database.

        Parameters
        ----------
        content: str
            The free‑form text content of the note.
        """
        note_id = str(uuid.uuid4())
        self.notes.append({"id": note_id, "content": content})
        self._save()

    def search(self, query: str, top_k: int = 3) -> List[Tuple[dict, float]]:
        """Return the top ``top_k`` notes most similar to a query string.

        This method uses TF‑IDF vectorisation and cosine similarity to
        compute relevance scores between notes and the query.  It
        dynamically fits a new vectoriser on the corpus of notes each
        time it is called.

        Parameters
        ----------
        query: str
            The question or search term.
        top_k: int
            The number of top results to return.

        Returns
        -------
        List[Tuple[dict, float]]
            A list of (note, similarity_score) pairs sorted by
            descending similarity.
        """
        if not self.notes:
            return []
        docs: List[str] = [n["content"] for n in self.notes]
        # Lazy import to avoid heavy dependency if unused
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

        vectorizer = TfidfVectorizer()
        note_matrix = vectorizer.fit_transform(docs)
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(note_matrix, query_vec).flatten()
        idx_sorted = sims.argsort()[::-1]
        results: List[Tuple[dict, float]] = []
        for i in idx_sorted[: max(top_k, 0)]:
            results.append((self.notes[int(i)], float(sims[int(i)])))
        return results