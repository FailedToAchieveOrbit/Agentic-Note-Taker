"""Synthesis agent - combines information into coherent responses."""

import logging
from typing import Optional

from .base import BaseAgent, AgentState

logger = logging.getLogger(__name__)


class SynthesisAgent(BaseAgent):
    """Combines search results and analysis into coherent responses.

    Uses LLM to generate comprehensive answers based on retrieved information.
    """

    def __init__(self, name: str = "synthesizer") -> None:
        """Initialize synthesis agent.

        Args:
            name: Agent identifier
        """
        super().__init__(name)

    async def execute(self, state: AgentState) -> AgentState:
        """Generate synthesized answer.

        Args:
            state: Current workflow state

        Returns:
            Updated state with synthesized answer
        """
        if not state.search_results:
            state.answer = "No relevant information found in knowledge base."
            state.sources = []
            return state

        try:
            state.answer = await self._generate_answer(
                state.query, state.search_results
            )
            state.sources = [note.get("id") for note, _ in state.search_results]

            logger.info(
                f"Synthesis agent generated answer of "
                f"{len(state.answer.split())} words"
            )
            state.context["synthesis_complete"] = True

            return state

        except Exception as e:
            logger.error(f"Synthesis agent error: {e}")
            state.answer = f"Error generating answer: {str(e)}"
            state.sources = []
            state.context["synthesis_error"] = str(e)
            return state

    async def _generate_answer(
        self, query: str, search_results: list
    ) -> str:
        """Generate answer using LLM.

        Args:
            query: User query
            search_results: Search result tuples

        Returns:
            Generated answer string
        """
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback to simple answer
            return self._simple_answer(query, search_results)

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key)

            # Build context from search results
            context_parts = []
            for i, (note, score) in enumerate(search_results[:3], 1):
                content_preview = note.get("content", "")[:200]
                context_parts.append(
                    f"Note {i} (relevance: {score:.2f}): {content_preview}..."
                )

            context = "\n\n".join(context_parts)

            prompt = f"""
Based on these sources, answer the user's question concisely and clearly.

Question: {query}

Sources:
{context}

Instructions:
- Use only information from the sources
- Keep the answer clear and well-structured
- If sources don't contain relevant info, say so
- Reference source numbers when relevant

Answer:
"""

            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"LLM synthesis failed, using simple answer: {e}")
            return self._simple_answer(query, search_results)

    @staticmethod
    def _simple_answer(query: str, search_results: list) -> str:
        """Generate simple answer without LLM.

        Args:
            query: User query
            search_results: Search results

        Returns:
            Simple synthesized answer
        """
        if not search_results:
            return "No information found."

        answer_parts = [
            f"Based on {len(search_results)} relevant notes:\n",
        ]

        for i, (note, score) in enumerate(search_results[:3], 1):
            content = note.get("content", "")
            preview = content[:150] + "..." if len(content) > 150 else content
            answer_parts.append(f"\n{i}. {preview} (relevance: {score:.2f})")

        return "".join(answer_parts)
