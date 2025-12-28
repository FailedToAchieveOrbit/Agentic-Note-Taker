"""Tests for multi-agent system."""

import pytest
from agentic_notes.agents.base import AgentState, BaseAgent
from agentic_notes.agents.planning import PlanningAgent
from agentic_notes.agents.search import SearchAgent
from agentic_notes.agents.analysis import AnalysisAgent
from agentic_notes.agents.synthesis import SynthesisAgent
from agentic_notes.agents.reflection import ReflectionAgent
from agentic_notes.agents.graph import NoteAgentGraph


class TestAgentState:
    """Test AgentState dataclass."""

    def test_agent_state_creation(self):
        """Test creating an agent state."""
        state = AgentState(query="test query")
        assert state.query == "test query"
        assert state.context == {}
        assert state.search_results is None
        assert state.answer is None
        assert state.confidence == 0.0

    def test_agent_state_with_context(self):
        """Test agent state with context."""
        context = {"user_id": "123", "session_id": "abc"}
        state = AgentState(query="test", context=context)
        assert state.context == context

    def test_agent_state_repr(self):
        """Test agent state string representation."""
        state = AgentState(query="test query", confidence=0.75)
        repr_str = repr(state)
        assert "test query" in repr_str
        assert "0.75" in repr_str


class TestPlanningAgent:
    """Test Planning Agent."""

    @pytest.mark.asyncio
    async def test_planning_agent_execution(self):
        """Test planning agent creates a plan."""
        agent = PlanningAgent()
        state = AgentState(query="What about machine learning?")
        result = await agent.execute(state)

        assert "planning_complete" in result.context
        assert "plan" in result.context
        plan = result.context["plan"]
        assert "steps" in plan
        assert len(plan["steps"]) > 0

    def test_planning_agent_simple_plan(self):
        """Test simple plan generation."""
        plan = PlanningAgent._simple_plan("test query")
        assert "steps" in plan
        assert "complexity" in plan
        assert "estimated_duration_seconds" in plan
        assert len(plan["steps"]) >= 3


class TestSearchAgent:
    """Test Search Agent."""

    @pytest.mark.asyncio
    async def test_search_agent_no_db(self):
        """Test search agent without database."""
        agent = SearchAgent(db=None)
        state = AgentState(query="test query")
        result = await agent.execute(state)

        assert "search_error" in result.context
        assert result.search_results == []

    @pytest.mark.asyncio
    async def test_search_agent_empty_results(self):
        """Test search agent with no results."""
        # Create mock database that returns no results
        class MockDB:
            async def search(self, query, top_k, method, threshold):
                return []

        agent = SearchAgent(db=MockDB())
        state = AgentState(query="nonexistent query")
        result = await agent.execute(state)

        assert result.search_results == []
        assert result.confidence == 0.0


class TestAnalysisAgent:
    """Test Analysis Agent."""

    @pytest.mark.asyncio
    async def test_analysis_agent_no_results(self):
        """Test analysis with no search results."""
        agent = AnalysisAgent()
        state = AgentState(query="test", search_results=[])
        result = await agent.execute(state)

        assert "analysis_error" in result.context

    @pytest.mark.asyncio
    async def test_analysis_agent_with_results(self):
        """Test analysis with search results."""
        agent = AnalysisAgent()
        mock_results = [
            ({"content": "machine learning is powerful"}, 0.8),
            ({"content": "deep learning advances AI"}, 0.75),
        ]
        state = AgentState(query="test", search_results=mock_results)
        result = await agent.execute(state)

        assert result.analysis is not None
        assert "word_count" in result.analysis
        assert "unique_words" in result.analysis
        assert result.analysis["note_count"] == 2

    def test_content_analysis(self):
        """Test content analysis function."""
        content = "machine learning deep learning neural networks"
        results = [("note1", 0.8), ("note2", 0.7)]
        analysis = AnalysisAgent._analyze_content(content, results)

        assert analysis["word_count"] > 0
        assert analysis["unique_words"] > 0
        assert analysis["avg_relevance_score"] == 0.75
        assert "top_phrases" in analysis


class TestSynthesisAgent:
    """Test Synthesis Agent."""

    @pytest.mark.asyncio
    async def test_synthesis_no_results(self):
        """Test synthesis with no search results."""
        agent = SynthesisAgent()
        state = AgentState(query="test", search_results=[])
        result = await agent.execute(state)

        assert "No relevant information" in result.answer
        assert result.sources == []

    @pytest.mark.asyncio
    async def test_synthesis_simple_answer(self):
        """Test simple answer generation."""
        agent = SynthesisAgent()
        mock_results = [
            ({"id": "note1", "content": "machine learning is important"}, 0.8),
            ({"id": "note2", "content": "deep learning is powerful"}, 0.75),
        ]
        state = AgentState(query="test", search_results=mock_results)
        result = await agent.execute(state)

        assert result.answer is not None
        assert len(result.answer) > 0
        assert "note1" in result.sources
        assert "note2" in result.sources


class TestReflectionAgent:
    """Test Reflection Agent."""

    @pytest.mark.asyncio
    async def test_reflection_no_answer(self):
        """Test reflection without answer."""
        agent = ReflectionAgent()
        state = AgentState(query="test", answer=None)
        result = await agent.execute(state)

        reflection = result.context["reflection"]
        assert reflection["quality_score"] == 0.0
        assert len(reflection["issues"]) > 0

    @pytest.mark.asyncio
    async def test_reflection_with_answer(self):
        """Test reflection with answer."""
        agent = ReflectionAgent()
        mock_results = [({"id": "note1", "content": "test"}, 0.8)]
        state = AgentState(
            query="test",
            answer="This is a test answer about the topic.",
            search_results=mock_results,
            analysis={"avg_relevance_score": 0.8},
        )
        result = await agent.execute(state)

        reflection = result.context["reflection"]
        assert "quality_score" in reflection
        assert "issues" in reflection
        assert "suggestions" in reflection

    def test_simple_evaluation(self):
        """Test simple evaluation function."""
        results = [({"id": "note1"}, 0.8)]
        analysis = {"avg_relevance_score": 0.8}
        evaluation = ReflectionAgent._simple_evaluation(
            "test query", "This is a detailed test answer.", results, analysis
        )

        assert "quality_score" in evaluation
        assert "completeness" in evaluation
        assert "accuracy" in evaluation
        assert "clarity" in evaluation
        assert "issues" in evaluation
        assert "suggestions" in evaluation


class TestNoteAgentGraph:
    """Test NoteAgentGraph orchestrator."""

    def test_graph_initialization(self):
        """Test initializing agent graph."""
        graph = NoteAgentGraph(db=None)
        assert graph.db is None
        assert "planner" in graph.agents
        assert "searcher" in graph.agents
        assert "analyzer" in graph.agents
        assert "synthesizer" in graph.agents
        assert "reflector" in graph.agents

    def test_graph_diagram(self):
        """Test workflow diagram generation."""
        graph = NoteAgentGraph()
        diagram = graph.get_graph_diagram()
        assert "Planner" in diagram
        assert "Searcher" in diagram
        assert "Analyzer" in diagram
        assert "Synthesizer" in diagram
        assert "Reflector" in diagram

    @pytest.mark.asyncio
    async def test_graph_execution(self):
        """Test full agent graph execution."""
        graph = NoteAgentGraph(db=None)  # No DB for testing
        state = await graph.execute("What about AI?")

        assert state.query == "What about AI?"
        assert "planning_complete" in state.context
        assert "search_error" in state.context  # Expected since no DB
        assert state.answer is not None  # Should have fallback answer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
