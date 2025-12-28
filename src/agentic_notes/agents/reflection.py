"""Reflection agent - evaluates response quality."""

import json
import logging
import re
from typing import Any, Dict

from .base import BaseAgent, AgentState

logger = logging.getLogger(__name__)


class ReflectionAgent(BaseAgent):
    """Evaluates and improves response quality.

    Reflects on the generated answer and provides quality metrics.
    """

    def __init__(self, name: str = "reflector") -> None:
        """Initialize reflection agent.

        Args:
            name: Agent identifier
        """
        super().__init__(name)

    async def execute(self, state: AgentState) -> AgentState:
        """Reflect on answer quality.

        Args:
            state: Current workflow state

        Returns:
            Updated state with reflection results
        """
        if not state.answer:
            state.context["reflection"] = {
                "quality_score": 0.0,
                "issues": ["No answer generated"],
                "suggestions": ["Generate an answer first"],
            }
            return state

        try:
            reflection = await self._evaluate_answer(
                state.query, state.answer, state.search_results, state.analysis
            )
            state.context["reflection"] = reflection
            state.confidence = reflection.get("quality_score", state.confidence)

            logger.info(
                f"Reflection agent evaluated answer quality: "
                f"{reflection['quality_score']:.2f}/1.0"
            )
            state.context["reflection_complete"] = True

            return state

        except Exception as e:
            logger.error(f"Reflection agent error: {e}")
            state.context["reflection_error"] = str(e)
            return state

    async def _evaluate_answer(
        self,
        query: str,
        answer: str,
        search_results: list,
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate answer quality using LLM.

        Args:
            query: Original user query
            answer: Generated answer
            search_results: Search results used
            analysis: Content analysis results

        Returns:
            Reflection evaluation
        """
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._simple_evaluation(query, answer, search_results, analysis)

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key)

            prompt = f"""
Evaluate the quality of this answer to the user's question.

Question: {query}
Answer: {answer}
Search Results: {len(search_results or [])} found
Average Relevance: {analysis.get('avg_relevance_score', 0):.2f}

Provide a JSON response with:
- quality (0-1): overall quality score
- completeness (0-1): how well does it answer the question
- accuracy (0-1): based on source relevance
- clarity (0-1): how clear is the answer
- issues: list of any problems or gaps
- suggestions: list of improvements

Example:
{{
    "quality": 0.85,
    "completeness": 0.8,
    "accuracy": 0.9,
    "clarity": 0.85,
    "issues": [],
    "suggestions": []
}}
"""

            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )

            response_text = response.choices[0].message.content
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)

            if json_match:
                parsed = json.loads(json_match.group())
                # Calculate average quality
                quality = (
                    parsed.get("quality", 0.5)
                    + parsed.get("completeness", 0.5)
                    + parsed.get("accuracy", 0.5)
                    + parsed.get("clarity", 0.5)
                ) / 4
                parsed["quality_score"] = quality
                return parsed

            return self._simple_evaluation(query, answer, search_results, analysis)

        except Exception as e:
            logger.warning(f"LLM reflection failed, using simple evaluation: {e}")
            return self._simple_evaluation(query, answer, search_results, analysis)

    @staticmethod
    def _simple_evaluation(
        query: str, answer: str, search_results: list, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simple evaluation without LLM.

        Args:
            query: Original query
            answer: Generated answer
            search_results: Search results
            analysis: Analysis results

        Returns:
            Simple evaluation
        """
        issues = []
        suggestions = []

        # Evaluate based on simple heuristics
        answer_length = len(answer.split())
        if answer_length < 10:
            issues.append("Answer is too short")
            suggestions.append("Provide more detailed response")
        elif answer_length > 500:
            issues.append("Answer is very long")
            suggestions.append("Summarize key points")

        if not search_results:
            issues.append("No sources found")
            suggestions.append("Try different search terms")

        avg_relevance = analysis.get("avg_relevance_score", 0)
        quality_score = min(avg_relevance, 0.95) if avg_relevance > 0 else 0.3

        return {
            "quality_score": quality_score,
            "completeness": 0.7 if len(search_results) > 2 else 0.5,
            "accuracy": avg_relevance,
            "clarity": 0.7,
            "issues": issues,
            "suggestions": suggestions,
        }
