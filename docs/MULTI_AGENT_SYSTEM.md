# Multi-Agent System Documentation

## Overview

The Agentic Note Taker uses a sophisticated multi-agent system powered by LangGraph to process user queries intelligently. Five specialized agents collaborate to understand queries, search knowledge bases, analyze content, synthesize answers, and evaluate quality.

## Architecture

```
┌─────────┐
│ START   │
└────┬────┘
     │
     ▼
┌──────────────────────────────────────┐
│ PLANNER AGENT                        │
│ • Decomposes complex queries        │
│ • Creates execution plan            │
│ • Identifies dependencies           │
└────┬─────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────┐
│ SEARCHER AGENT                       │
│ • Performs hybrid search             │
│ • Combines semantic + BM25           │
│ • Retrieves relevant notes           │
└────┬─────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────┐
│ ANALYZER AGENT                       │
│ • Extracts key information           │
│ • Identifies entities & topics       │
│ • Computes relevance metrics         │
└────┬─────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────┐
│ SYNTHESIZER AGENT                    │
│ • Generates coherent answer          │
│ • Uses LLM for synthesis             │
│ • Includes source citations          │
└────┬─────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────┐
│ REFLECTOR AGENT                      │
│ • Evaluates answer quality           │
│ • Computes confidence scores         │
│ • Suggests improvements              │
└────┬─────────────────────────────────┘
     │
     ▼
┌─────────┐
│ END     │
└─────────┘
```

## Agent Specifications

### 1. Planning Agent

**Purpose:** Decompose complex user queries into actionable steps.

**Key Methods:**
- `execute(state)` - Main execution method
- `_llm_plan(client, query)` - Generate plan using OpenAI
- `_simple_plan(query)` - Fallback simple plan
- `_parse_plan(response)` - Extract JSON from LLM response

**Input State:**
```python
AgentState(
    query="What about machine learning?"
)
```

**Output State Updates:**
```python
state.context["plan"] = {
    "steps": ["Search", "Analyze", "Synthesize"],
    "complexity": "medium",
    "estimated_duration_seconds": 30
}
state.context["planning_complete"] = True
```

**Environment Variables:**
- `OPENAI_API_KEY` - For LLM-based planning (optional)

---

### 2. Search Agent

**Purpose:** Intelligently search the knowledge base for relevant notes.

**Key Methods:**
- `execute(state)` - Main execution method
- Uses hybrid search (semantic + BM25)

**Input State:**
```python
AgentState(
    query="What about machine learning?"
)
```

**Output State Updates:**
```python
state.search_results = [
    ({"id": "note1", "content": "...", ...}, 0.85),
    ({"id": "note2", "content": "...", ...}, 0.75),
]
state.context["search_method"] = "hybrid"
state.context["num_results"] = 2
state.confidence = 0.8  # Average of search scores
```

**Dependencies:**
- Database with `search()` method
- Must support `method="hybrid"`

---

### 3. Analysis Agent

**Purpose:** Extract insights and structure information from search results.

**Key Methods:**
- `execute(state)` - Main execution method
- `_analyze_content(content, search_results)` - Analyze combined content

**Input State:**
```python
AgentState(
    query="...",
    search_results=[...]
)
```

**Output State Updates:**
```python
state.analysis = {
    "word_count": 5000,
    "unique_words": 800,
    "note_count": 5,
    "avg_relevance_score": 0.80,
    "lexical_diversity": 0.16,
    "top_phrases": ["machine learning", "deep learning", ...],
    "entities": [],  # Populated by spacy in advanced version
    "sentiment": "neutral",  # Populated by textblob
    "topics": []  # Populated by LDA
}
state.context["analysis_complete"] = True
```

**Future Enhancements:**
- Named Entity Recognition (spacy)
- Sentiment Analysis (textblob)
- Topic Modeling (scikit-learn LDA)

---

### 4. Synthesis Agent

**Purpose:** Generate coherent, well-sourced answers from retrieved information.

**Key Methods:**
- `execute(state)` - Main execution method
- `_generate_answer(query, search_results)` - Generate using LLM
- `_simple_answer(query, search_results)` - Fallback answer

**Input State:**
```python
AgentState(
    query="What about machine learning?",
    search_results=[...],
    analysis={...}
)
```

**Output State Updates:**
```python
state.answer = "Based on {N} relevant notes:\n[Synthesized answer here]"
state.sources = ["note1_id", "note2_id", ...]
state.context["synthesis_complete"] = True
```

**LLM Prompt Template:**
```
Based on these sources, answer the user's question concisely.

Question: {query}

Sources:
[Note previews with relevance scores]

Instructions:
- Use only information from sources
- Keep answer clear and well-structured
- Reference source numbers when relevant

Answer:
```

---

### 5. Reflection Agent

**Purpose:** Evaluate response quality and provide confidence metrics.

**Key Methods:**
- `execute(state)` - Main execution method
- `_evaluate_answer(query, answer, search_results, analysis)` - LLM evaluation
- `_simple_evaluation(...)` - Fallback evaluation

**Input State:**
```python
AgentState(
    query="...",
    answer="...",
    search_results=[...],
    analysis={...}
)
```

**Output State Updates:**
```python
state.context["reflection"] = {
    "quality_score": 0.82,  # 0-1
    "completeness": 0.80,
    "accuracy": 0.85,
    "clarity": 0.80,
    "issues": ["Could provide more detail"],
    "suggestions": ["Add examples"]
}
state.confidence = 0.82  # Updates confidence based on quality
state.context["reflection_complete"] = True
```

**Evaluation Criteria:**
- **Quality Score:** Overall quality (0-1)
- **Completeness:** How well query is answered
- **Accuracy:** Based on source relevance
- **Clarity:** Answer comprehensibility
- **Issues:** Identified problems
- **Suggestions:** Improvements

---

## Data Flow

### AgentState Object

The `AgentState` dataclass flows through all agents:

```python
@dataclass
class AgentState:
    query: str                      # User question
    context: Dict[str, Any]         # Shared context between agents
    search_results: Optional[List]  # (note, score) tuples
    analysis: Optional[Dict]        # Content analysis results
    answer: Optional[str]           # Generated answer
    confidence: float               # Confidence score (0-1)
    sources: Optional[List[str]]    # Source note IDs
    timestamp: datetime             # When state was created
```

### State Evolution

```
Initial State:
{
    query: "What about machine learning?",
    context: {},
    search_results: None,
    analysis: None,
    answer: None,
    confidence: 0.0,
    sources: None
}

After Planner:
{
    ... (all above) ...
    context: {
        "plan": {...},
        "planning_complete": True
    }
}

After Searcher:
{
    ... (all above) ...
    search_results: [...],
    confidence: 0.80,
    context: {
        ... (previous context) ...
        "search_method": "hybrid",
        "num_results": 5,
        "search_complete": True
    }
}

After Analyzer:
{
    ... (all above) ...
    analysis: {...},
    context: {
        ... (previous context) ...
        "analysis_complete": True
    }
}

After Synthesizer:
{
    ... (all above) ...
    answer: "...",
    sources: ["note1", ...],
    context: {
        ... (previous context) ...
        "synthesis_complete": True
    }
}

After Reflector:
{
    ... (all above) ...
    confidence: 0.82,  # Updated from reflection
    context: {
        ... (previous context) ...
        "reflection": {...},
        "reflection_complete": True
    }
}
```

## NoteAgentGraph Orchestrator

### Usage

```python
from agentic_notes.agents.graph import NoteAgentGraph
from agentic_notes.database import NoteDatabase

# Initialize
db = NoteDatabase()
graph = NoteAgentGraph(db=db)

# Execute
final_state = await graph.execute(
    query="What about machine learning?",
    context={"user_id": "123"}
)

# Access results
print(final_state.answer)
print(final_state.confidence)
print(final_state.sources)
print(final_state.context["reflection"])
```

### Workflow Properties

- **Sequential Execution:** Agents run in order (Planner → ... → Reflector)
- **State Passing:** Each agent receives full state, returns updated state
- **Error Handling:** Each agent has try-catch, includes error in context
- **Async/Await:** All agents are async for concurrent operations
- **LangGraph Integration:** Leverages LangGraph's StateGraph for orchestration

## Integration with FastAPI

### POST /chat Endpoint

```python
from fastapi import FastAPI
from agentic_notes.agents.graph import NoteAgentGraph

app = FastAPI()
graph = NoteAgentGraph(db=db)

@app.post("/chat")
async def chat(request: ChatRequest):
    state = await graph.execute(
        query=request.query,
        context=request.context or {}
    )
    
    return {
        "answer": state.answer,
        "confidence": state.confidence,
        "sources": state.sources,
        "analysis": state.analysis,
        "reflection": state.context.get("reflection"),
        "planning": state.context.get("plan")
    }
```

## Testing

Comprehensive test suite in `tests/test_agents.py`:

```bash
# Run all agent tests
pytest tests/test_agents.py -v

# Run specific agent tests
pytest tests/test_agents.py::TestPlanningAgent -v
pytest tests/test_agents.py::TestSearchAgent -v
pytest tests/test_agents.py::TestAnalysisAgent -v
pytest tests/test_agents.py::TestSynthesisAgent -v
pytest tests/test_agents.py::TestReflectionAgent -v

# Run orchestrator tests
pytest tests/test_agents.py::TestNoteAgentGraph -v

# Run with coverage
pytest tests/test_agents.py --cov=src/agentic_notes/agents
```

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional (for advanced features)
SPACY_MODEL=en_core_web_sm  # For NER
LDAFORUM_PATH=/path/to/lda  # For topic modeling
```

## Performance Metrics

### Expected Times

- **Planning:** 100-500ms
- **Search:** 100-300ms
- **Analysis:** 50-200ms
- **Synthesis:** 500-2000ms (includes LLM)
- **Reflection:** 500-1500ms (includes LLM)
- **Total:** ~1.5-4 seconds

### Optimization Strategies

1. **Caching:** Cache embeddings and LLM responses
2. **Parallel Execution:** Run independent agents concurrently
3. **LLM Selection:** Use faster models (gpt-4o-mini) where appropriate
4. **Batching:** Process multiple queries together

## Future Enhancements

### Phase 2: Advanced NLP
- Named Entity Recognition (spacy)
- Sentiment Analysis (textblob)
- Topic Modeling (LDA)

### Phase 3: External Integration
- Web search enrichment
- Fact-checking agent
- Knowledge graph integration

### Phase 4: Production Features
- LangFuse observability
- Rate limiting
- Authentication
- Caching strategies

## Troubleshooting

### No Results from Search
- Check database has notes
- Verify search method is "hybrid"
- Check threshold value (default: 0.1)

### Low Confidence Scores
- Ensure search results are relevant
- Check if analysis is extracting correct metrics
- Verify LLM is generating good answers

### LLM Timeouts
- Reduce max_tokens in prompts
- Use faster model (gpt-4o-mini)
- Add request timeouts

## See Also

- [Architecture Documentation](./ARCHITECTURE.md)
- [API Reference](./API.md)
- [Configuration Guide](../note_taker/config.py)

---

**Last Updated:** 2025-12-28  
**Status:** Production Ready  
**Completion:** 100% of multi-agent feature
