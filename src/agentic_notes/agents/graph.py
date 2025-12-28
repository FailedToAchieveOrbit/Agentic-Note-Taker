"""LangGraph orchestrator for multi-agent workflow."""

import logging
from typing import Optional, Callable, Any

from langgraph.graph import StateGraph, START, END

from .base import AgentState
from .planning import PlanningAgent
from .search import SearchAgent
from .analysis import AnalysisAgent
from .synthesis import SynthesisAgent
from .reflection import ReflectionAgent

logger = logging.getLogger(__name__)


class NoteAgentGraph:
    """LangGraph-based orchestrator for multi-agent workflow.

    Coordinates planning, search, analysis, synthesis, and reflection agents.
    """

    def __init__(self, db: Optional[object] = None) -> None:
        """Initialize agent graph.

        Args:
            db: Database instance for search agent
        """
        self.db = db
        self.graph = self._build_graph()
        self.agents = {
            "planner": PlanningAgent(),
            "searcher": SearchAgent(db=db),
            "analyzer": AnalysisAgent(),
            "synthesizer": SynthesisAgent(),
            "reflector": ReflectionAgent(),
        }

    def _build_graph(self) -> StateGraph:
        """Build the agent workflow graph.

        Returns:
            Compiled LangGraph StateGraph
        """
        # Create graph with AgentState
        graph = StateGraph(AgentState)

        # Add nodes for each agent
        graph.add_node("planner", self._node_planner)
        graph.add_node("searcher", self._node_searcher)
        graph.add_node("analyzer", self._node_analyzer)
        graph.add_node("synthesizer", self._node_synthesizer)
        graph.add_node("reflector", self._node_reflector)

        # Define workflow: START -> planner -> searcher -> analyzer -> synthesizer -> reflector -> END
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "searcher")
        graph.add_edge("searcher", "analyzer")
        graph.add_edge("analyzer", "synthesizer")
        graph.add_edge("synthesizer", "reflector")
        graph.add_edge("reflector", END)

        # Compile graph
        return graph.compile()

    async def _node_planner(self, state: AgentState) -> AgentState:
        """Execute planning agent.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        logger.info(f"Executing planner agent for query: {state.query[:50]}...")
        return await self.agents["planner"].execute(state)

    async def _node_searcher(self, state: AgentState) -> AgentState:
        """Execute search agent.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        logger.info("Executing search agent")
        return await self.agents["searcher"].execute(state)

    async def _node_analyzer(self, state: AgentState) -> AgentState:
        """Execute analysis agent.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        logger.info("Executing analyzer agent")
        return await self.agents["analyzer"].execute(state)

    async def _node_synthesizer(self, state: AgentState) -> AgentState:
        """Execute synthesis agent.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        logger.info("Executing synthesizer agent")
        return await self.agents["synthesizer"].execute(state)

    async def _node_reflector(self, state: AgentState) -> AgentState:
        """Execute reflection agent.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        logger.info("Executing reflector agent")
        return await self.agents["reflector"].execute(state)

    async def execute(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> AgentState:
        """Execute the agent workflow.

        Args:
            query: User query
            context: Optional additional context

        Returns:
            Final agent state with answer and metadata
        """
        # Initialize state
        state = AgentState(
            query=query,
            context=context or {},
        )

        logger.info(f"Starting agent workflow for query: {query[:50]}...")

        try:
            # Execute the graph
            # Note: LangGraph expects synchronous execution by default
            # In a real implementation, you'd use invoke() method
            # For now, we'll execute agents sequentially
            state = await self.agents["planner"].execute(state)
            state = await self.agents["searcher"].execute(state)
            state = await self.agents["analyzer"].execute(state)
            state = await self.agents["synthesizer"].execute(state)
            state = await self.agents["reflector"].execute(state)

            logger.info(
                f"Agent workflow completed. Answer: {state.answer[:100]}..., "
                f"Confidence: {state.confidence:.3f}"
            )

            return state

        except Exception as e:
            logger.error(f"Error executing agent workflow: {e}")
            state.answer = f"Error in agent workflow: {str(e)}"
            return state

    def get_graph_diagram(self) -> str:
        """Get ASCII diagram of agent workflow.

        Returns:
            ASCII workflow diagram
        """
        return """
        START
          ↓
     [Planner]    - Decompose query into steps
          ↓
     [Searcher]   - Search knowledge base (hybrid search)
          ↓
    [Analyzer]    - Extract entities, topics, sentiment
          ↓
  [Synthesizer]   - Generate answer from sources
          ↓
  [Reflector]     - Evaluate quality and confidence
          ↓
        END
        """
