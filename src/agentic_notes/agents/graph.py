"""🕸️ LangGraph-based agentic workflow for note management.

Implements a sophisticated agentic RAG system using LangGraph for orchestrating
multiple AI agents with planning, reflection, and tool use capabilities.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from ..config import get_settings
from ..core.database import AgenticDatabase
from ..core.vector_store import VectorStore
from .tools import SearchTool, NoteTool, AnalysisTool, WebSearchTool


class AgentState(TypedDict):
    """State maintained across the agentic workflow."""
    
    # Core state
    messages: List[BaseMessage]
    user_query: str
    task_type: str
    
    # Intermediate results
    search_results: Optional[List[Dict[str, Any]]]
    analysis_results: Optional[Dict[str, Any]]
    generated_content: Optional[str]
    
    # Planning and reflection
    plan: Optional[List[str]]
    current_step: int
    reflection: Optional[str]
    
    # Tool usage tracking
    tools_used: List[str]
    iteration_count: int
    max_iterations: int
    
    # Final outputs
    final_answer: Optional[str]
    confidence_score: Optional[float]
    sources: List[str]


class NoteAgentGraph:
    """LangGraph-based agentic system for note management."""
    
    def __init__(self, database: AgenticDatabase, vector_store: VectorStore):
        """Initialize the agentic graph."""
        self.settings = get_settings()
        self.database = database
        self.vector_store = vector_store
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.settings.llm.model_name,
            temperature=self.settings.llm.temperature,
            max_tokens=self.settings.llm.max_tokens,
            api_key=self.settings.api.openai_api_key,
        )
        
        # Initialize tools
        self.tools = {
            "search": SearchTool(vector_store, database),
            "note": NoteTool(database),
            "analysis": AnalysisTool(),
            "web_search": WebSearchTool(),
        }
        
        # Create tool node
        self.tool_node = ToolNode(list(self.tools.values()))
        
        # Build the graph
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("reflector", self._reflector_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Add edges
        workflow.add_edge("planner", "executor")
        workflow.add_conditional_edges(
            "executor",
            self._should_continue,
            {
                "tools": "tools",
                "reflect": "reflector",
                "synthesize": "synthesizer",
            }
        )
        workflow.add_edge("tools", "executor")
        workflow.add_edge("reflector", "executor")
        workflow.add_edge("synthesizer", END)
        
        return workflow
    
    async def _planner_node(self, state: AgentState) -> AgentState:
        """Planning node - analyzes the query and creates an execution plan."""
        
        system_prompt = """
You are an expert AI planning agent for a note management system.
Your job is to analyze user queries and create detailed execution plans.

Available tools:
- search: Search through existing notes using semantic and keyword search
- note: Create, update, or manage notes
- analysis: Analyze note content for topics, entities, sentiment
- web_search: Search the web for additional information

Based on the user query, create a step-by-step plan to accomplish their goal.
Respond with a JSON object containing:
- task_type: The type of task (search, create, analyze, etc.)
- plan: List of specific steps to execute
- estimated_steps: Number of steps expected

User Query: {query}
"""
        
        messages = [
            SystemMessage(content=system_prompt.format(query=state["user_query"])),
            HumanMessage(content=state["user_query"])
        ]
        
        response = await self.llm.ainvoke(messages)
        
        # Parse response (simplified - in production would use structured output)
        try:
            import json
            plan_data = json.loads(response.content)
            state["task_type"] = plan_data.get("task_type", "general")
            state["plan"] = plan_data.get("plan", ["Process user query"])
            state["max_iterations"] = len(state["plan"]) * 2
        except Exception:
            # Fallback if JSON parsing fails
            state["task_type"] = "general"
            state["plan"] = ["Analyze query", "Search for relevant information", "Generate response"]
            state["max_iterations"] = 6
        
        state["current_step"] = 0
        state["iteration_count"] = 0
        state["tools_used"] = []
        state["sources"] = []
        
        state["messages"] = add_messages(state.get("messages", []), [response])
        
        return state
    
    async def _executor_node(self, state: AgentState) -> AgentState:
        """Executor node - determines next action and executes tools."""
        
        system_prompt = """
You are an expert AI executor agent. Based on the current state and plan,
determine the next action to take.

Current Plan: {plan}
Current Step: {current_step}
Iteration: {iteration_count}/{max_iterations}
Tools Used: {tools_used}

Available actions:
1. Use a tool (specify which tool and parameters)
2. Request reflection on current progress
3. Generate final synthesis

Respond with your next action and reasoning.
"""
        
        messages = [
            SystemMessage(content=system_prompt.format(
                plan=state.get("plan", []),
                current_step=state.get("current_step", 0),
                iteration_count=state.get("iteration_count", 0),
                max_iterations=state.get("max_iterations", 5),
                tools_used=state.get("tools_used", [])
            )),
            HumanMessage(content=f"User query: {state['user_query']}")
        ]
        
        # Add recent context
        if state.get("messages"):
            messages.extend(state["messages"][-3:])  # Last 3 messages for context
        
        response = await self.llm.ainvoke(messages)
        
        state["iteration_count"] += 1
        state["messages"] = add_messages(state.get("messages", []), [response])
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine the next node based on current state."""
        
        last_message = state["messages"][-1]
        content = last_message.content.lower()
        
        # Check iteration limit
        if state.get("iteration_count", 0) >= state.get("max_iterations", 5):
            return "synthesize"
        
        # Determine action based on content
        if any(tool in content for tool in ["search", "note", "analysis", "web_search"]):
            return "tools"
        elif "reflect" in content or "review" in content:
            return "reflect"
        elif "final" in content or "synthesize" in content or "complete" in content:
            return "synthesize"
        else:
            # Default to tools if uncertain
            return "tools"
    
    async def _reflector_node(self, state: AgentState) -> AgentState:
        """Reflection node - evaluates progress and adjusts strategy."""
        
        system_prompt = """
You are an expert AI reflection agent. Analyze the current progress and provide
guidance for improvement.

Original Query: {query}
Plan: {plan}
Current Step: {current_step}
Tools Used: {tools_used}
Iteration: {iteration_count}

Evaluate:
1. Are we making progress toward the goal?
2. What has worked well so far?
3. What should be adjusted?
4. Should we continue or synthesize results?

Provide specific recommendations for next steps.
"""
        
        messages = [
            SystemMessage(content=system_prompt.format(
                query=state["user_query"],
                plan=state.get("plan", []),
                current_step=state.get("current_step", 0),
                tools_used=state.get("tools_used", []),
                iteration_count=state.get("iteration_count", 0)
            ))
        ]
        
        response = await self.llm.ainvoke(messages)
        
        state["reflection"] = response.content
        state["messages"] = add_messages(state.get("messages", []), [response])
        
        return state
    
    async def _synthesizer_node(self, state: AgentState) -> AgentState:
        """Synthesis node - generates final response based on all gathered information."""
        
        system_prompt = """
You are an expert AI synthesis agent. Generate a comprehensive final response
based on all the information gathered.

Original Query: {query}
Search Results: {search_results}
Analysis Results: {analysis_results}
Generated Content: {generated_content}
Sources: {sources}

Provide a helpful, accurate, and well-structured response that directly
addresses the user's query. Include relevant sources and confidence level.
"""
        
        messages = [
            SystemMessage(content=system_prompt.format(
                query=state["user_query"],
                search_results=state.get("search_results", []),
                analysis_results=state.get("analysis_results", {}),
                generated_content=state.get("generated_content", ""),
                sources=state.get("sources", [])
            ))
        ]
        
        response = await self.llm.ainvoke(messages)
        
        state["final_answer"] = response.content
        state["confidence_score"] = 0.8  # TODO: Implement actual confidence scoring
        state["messages"] = add_messages(state.get("messages", []), [response])
        
        return state
    
    async def process_query(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Process a user query through the agentic workflow."""
        
        # Initialize state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "task_type": "",
            "search_results": None,
            "analysis_results": None,
            "generated_content": None,
            "plan": None,
            "current_step": 0,
            "reflection": None,
            "tools_used": [],
            "iteration_count": 0,
            "max_iterations": 5,
            "final_answer": None,
            "confidence_score": None,
            "sources": [],
        }
        
        # Configure session
        config = {"configurable": {"thread_id": session_id or "default"}}
        
        try:
            # Run the workflow
            result = await self.app.ainvoke(initial_state, config=config)
            
            return {
                "answer": result.get("final_answer", "I couldn't generate a response."),
                "confidence": result.get("confidence_score", 0.0),
                "sources": result.get("sources", []),
                "task_type": result.get("task_type", "general"),
                "iterations": result.get("iteration_count", 0),
                "tools_used": result.get("tools_used", []),
            }
            
        except Exception as e:
            return {
                "answer": f"An error occurred: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "task_type": "error",
                "iterations": 0,
                "tools_used": [],
                "error": str(e)
            }


def create_note_agent_graph(
    database: AgenticDatabase, 
    vector_store: VectorStore
) -> NoteAgentGraph:
    """Factory function to create a configured note agent graph."""
    return NoteAgentGraph(database, vector_store)