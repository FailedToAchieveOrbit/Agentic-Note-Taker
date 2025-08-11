"""Wrapper around different embedding backends.

An instance of :class:`EmbeddingModel` can generate vector
representations for lists of texts.  By default it will use OpenAI’s
`text-embedding-ada-002` model when an API key is available via the
``OPENAI_API_KEY`` environment variable.  If no key is found it will
attempt to fall back to the open‑source `sentence-transformers`
package.  Should both be unavailable, an informative exception is
raised.
"""

from __future__ import annotations

import os
from typing import Iterable, List

import numpy as np


class EmbeddingModel:
    """Abstraction over OpenAI and SentenceTransformer embedding models."""

    def __init__(self) -> None:
        self.mode: str | None = None
        # Check for an OpenAI API key
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            import openai  # type: ignore

            openai.api_key = openai_key
            self.mode = "openai"
        else:
            # Try to load sentence‑transformers as a fallback
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.mode = "sbert"
            except Exception as exc:
                raise RuntimeError(
                    "Neither an OpenAI API key nor a compatible sentence-transformer model is available. "
                    "Please set the OPENAI_API_KEY environment variable or install sentence-transformers."
                ) from exc

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Compute embeddings for a batch of texts.

        Parameters
        ----------
        texts: Iterable[str]
            A list or iterable of strings to embed.

        Returns
        -------
        np.ndarray
            An array of shape (len(texts), dim) containing the embeddings.
        """
        # Convert to list to allow multiple passes
        list_texts: List[str] = list(texts)
        if self.mode == "openai":
            import openai  # type: ignore

            embeddings: List[List[float]] = []
            # The OpenAI API accepts up to ~2048 tokens per request; batch by 100 for
            # simplicity and to stay well within limits.  The ordering of responses
            # matches the input order if sorted by the `index` field.
            batch_size = 100
            for start in range(0, len(list_texts), batch_size):
                batch = list_texts[start : start + batch_size]
                resp = openai.Embedding.create(input=batch, model="text-embedding-ada-002")
                # `resp['data']` is a list of dicts with an `index` and `embedding`
                sorted_data = sorted(resp["data"], key=lambda x: x["index"])
                embeddings.extend([item["embedding"] for item in sorted_data])
            return np.array(embeddings, dtype=float)
        else:
            # Use the sentence-transformer; normalization optional (for cosine similarity)
            embs = self.model.encode(list_texts, normalize_embeddings=True)
            return embs.astype(float)