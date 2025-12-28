"""Search agent - retrieves relevant notes from knowledge base."""

import logging
from typing import Optional

from .base import BaseAgent, AgentState

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent):
    """Intelligently searches the knowledge base for relevant notes.

    Uses hybrid search combining semantic and BM25 methods.
    """

    def __init__(self, name: str = "searcher", db: Optional[object] = None) -> None:
        """Initialize search agent.

        Args:
            name: Agent identifier
            db: Database instance (injected dependency)
        """
        super().__init__(name)
        self.db = db

    async def execute(self, state: AgentState) -> AgentState:
        """Search for relevant notes.

        Args:
            state: Current workflow state

        Returns:
            Updated state with search results
        """
        if not self.db:
            logger.warning("Database not initialized for search agent")
            state.context["search_error"] = "Database not initialized"
            state.search_results = []
            state.context["num_results"] = 0
            return state

        try:
            # Perform hybrid search
            results = await self.db.search(
                query=state.query, top_k=5, method="hybrid", threshold=0.1
            )

            state.search_results = results
            state.context["search_method"] = "hybrid"
            state.context["num_results"] = len(results)

            # Calculate average confidence from search scores
            if results:
                scores = [score for _, score in results]
                state.confidence = sum(scores) / len(scores)
                logger.info(
                    f"Search agent found {len(results)} results, "
                    f"avg confidence: {state.confidence:.3f}"
                )
            else:
                state.confidence = 0.0
                logger.warning(f"Search agent found no results for query: {state.query}")

            state.context["search_complete"] = True
            return state

        except Exception as e:
            logger.error(f"Search agent error: {e}")
            state.context["search_error"] = str(e)
            state.search_results = []
            state.confidence = 0.0
            return state
