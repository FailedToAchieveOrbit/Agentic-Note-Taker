"""Analysis agent - extracts insights from content."""

import logging
from typing import Any, Dict

from .base import BaseAgent, AgentState

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseAgent):
    """Analyzes content for topics, entities, and sentiment.

    Extracts key information from search results for synthesis.
    """

    def __init__(self, name: str = "analyzer") -> None:
        """Initialize analysis agent.

        Args:
            name: Agent identifier
        """
        super().__init__(name)

    async def execute(self, state: AgentState) -> AgentState:
        """Analyze search results.

        Args:
            state: Current workflow state

        Returns:
            Updated state with analysis results
        """
        if not state.search_results:
            logger.warning("No search results to analyze")
            state.analysis = {"error": "No search results"}
            state.context["analysis_error"] = "No search results available"
            return state

        try:
            # Combine all search result content
            combined_content = "\n".join(
                [note.get("content", "") for note, _ in state.search_results]
            )

            # Perform basic analysis
            analysis = self._analyze_content(combined_content, state.search_results)

            state.analysis = analysis
            state.context["analysis_complete"] = True

            logger.info(
                f"Analysis agent completed: "
                f"found {len(state.search_results)} notes, "
                f"total {analysis['word_count']} words"
            )

            return state

        except Exception as e:
            logger.error(f"Analysis agent error: {e}")
            state.context["analysis_error"] = str(e)
            state.analysis = {"error": str(e)}
            return state

    @staticmethod
    def _analyze_content(
        content: str, search_results: list
    ) -> Dict[str, Any]:
        """Perform content analysis.

        Args:
            content: Combined content from search results
            search_results: List of (note, score) tuples

        Returns:
            Analysis results dictionary
        """
        words = content.split()
        word_count = len(words)
        unique_words = len(set(w.lower() for w in words))

        # Calculate statistics
        avg_score = (
            sum(score for _, score in search_results) / len(search_results)
            if search_results
            else 0.0
        )

        # Extract key phrases (simple: 2-3 word phrases)
        phrases = []
        words_lower = content.lower().split()
        for i in range(len(words_lower) - 1):
            phrase = f"{words_lower[i]} {words_lower[i + 1]}"
            if len(phrase) > 5:  # Minimum phrase length
                phrases.append(phrase)

        # Get most common phrases
        phrase_counts = {}
        for phrase in phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        top_phrases = sorted(
            phrase_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "word_count": word_count,
            "unique_words": unique_words,
            "note_count": len(search_results),
            "avg_relevance_score": avg_score,
            "lexical_diversity": unique_words / word_count if word_count > 0 else 0,
            "top_phrases": [phrase for phrase, _ in top_phrases],
            "entities": [],  # Would be populated by spacy in full implementation
            "sentiment": "neutral",  # Would be populated by textblob in full implementation
            "topics": [],  # Would be populated by LDA in full implementation
        }
