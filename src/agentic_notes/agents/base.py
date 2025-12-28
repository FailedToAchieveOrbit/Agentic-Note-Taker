"""Base agent interface and state management."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class AgentState:
    """Shared state across agents in the workflow."""

    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    search_results: Optional[List[tuple]] = None
    analysis: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None
    confidence: float = 0.0
    sources: Optional[List[str]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        return (
            f"AgentState(query='{self.query[:50]}...', "
            f"confidence={self.confidence:.2f}, "
            f"has_answer={self.answer is not None})"
        )


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name: str) -> None:
        """Initialize agent with name.

        Args:
            name: Agent identifier
        """
        self.name = name
        self.execution_count = 0
        self.total_duration = 0.0

    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """Execute agent task and return updated state.

        Args:
            state: Current workflow state

        Returns:
            Updated state after agent execution
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    def __str__(self) -> str:
        return self.name
