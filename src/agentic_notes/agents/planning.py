"""Planning agent - decomposes queries into actionable steps."""

import json
import logging
import re
from typing import Any, Dict

from .base import BaseAgent, AgentState

logger = logging.getLogger(__name__)


class PlanningAgent(BaseAgent):
    """Decomposes complex queries into actionable steps.

    This agent analyzes user requests and breaks them down into
    clear, ordered steps for execution.
    """

    def __init__(self, name: str = "planner") -> None:
        """Initialize planning agent.

        Args:
            name: Agent identifier
        """
        super().__init__(name)

    async def execute(self, state: AgentState) -> AgentState:
        """Create an execution plan for the query.

        Args:
            state: Current workflow state

        Returns:
            Updated state with plan in context
        """
        try:
            # Import here to avoid circular imports
            from openai import AsyncOpenAI
            import os

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set, using simple planning")
                plan = self._simple_plan(state.query)
            else:
                client = AsyncOpenAI(api_key=api_key)
                plan = await self._llm_plan(client, state.query)

            state.context["plan"] = plan
            state.context["planning_complete"] = True

            logger.info(f"Planning agent created {len(plan.get('steps', []))} steps")
            return state

        except Exception as e:
            logger.error(f"Planning agent error: {e}")
            state.context["planning_error"] = str(e)
            # Fallback to simple plan
            state.context["plan"] = self._simple_plan(state.query)
            return state

    async def _llm_plan(self, client: Any, query: str) -> Dict[str, Any]:
        """Generate plan using LLM.

        Args:
            client: OpenAI async client
            query: User query to plan for

        Returns:
            Structured plan
        """
        prompt = f"""
        Analyze this user request and break it down into clear, actionable steps.
        
        Request: {query}
        
        Respond with a JSON object containing:
        - steps: list of ordered steps (each step is a short string)
        - dependencies: array of step indices that have dependencies
        - complexity: "low", "medium", or "high"
        - estimated_duration_seconds: rough estimate
        
        Example format:
        {{
            "steps": ["Search for relevant notes", "Analyze content", "Synthesize answer"],
            "dependencies": [],
            "complexity": "medium",
            "estimated_duration_seconds": 30
        }}
        """

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        response_text = response.choices[0].message.content
        return self._parse_plan(response_text)

    @staticmethod
    def _parse_plan(response: str) -> Dict[str, Any]:
        """Parse LLM response into structured plan.

        Args:
            response: Raw LLM response

        Returns:
            Parsed plan dictionary
        """
        # Extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse plan JSON: {e}")
                return {"steps": [], "error": "Failed to parse plan"}

        return {"steps": [], "error": "No JSON found in response"}

    @staticmethod
    def _simple_plan(query: str) -> Dict[str, Any]:
        """Create simple plan without LLM.

        Args:
            query: User query

        Returns:
            Basic plan structure
        """
        return {
            "steps": [
                "Search for relevant notes",
                "Analyze search results",
                "Extract key information",
                "Synthesize comprehensive answer",
                "Evaluate quality and confidence",
            ],
            "dependencies": [],
            "complexity": "medium",
            "estimated_duration_seconds": 30,
            "method": "simple",
        }
