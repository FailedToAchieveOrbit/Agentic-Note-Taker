"""🤖 Agentic system for intelligent note management.

Multi-agent system using LangGraph, CrewAI, and OpenAI Swarm for collaborative
note processing, search, and user interaction.
"""

from .base import AgenticAgent, AgentState
from .crew import NoteCrew
from .graph import create_note_agent_graph
from .tools import (
    SearchTool,
    NoteTool, 
    AnalysisTool,
    WebSearchTool,
)

__all__ = [
    "AgenticAgent",
    "AgentState", 
    "NoteCrew",
    "create_note_agent_graph",
    "SearchTool",
    "NoteTool",
    "AnalysisTool",
    "WebSearchTool",
]