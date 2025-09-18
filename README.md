# 🤖 Agentic Note Taker - Next-Gen AI-Powered Note Management

> **A revolutionary note-taking system powered by multiple AI agents working collaboratively to help you capture, organize, and interact with your knowledge.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-green.svg)](https://github.com/langchain-ai/langgraph)
[![CrewAI](https://img.shields.io/badge/CrewAI-Latest-orange.svg)](https://github.com/joaomdmoura/crewAI)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-red.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎆 What Makes This Special

This isn't just another note-taking app. It's a **fully agentic system** where AI agents collaborate to:

- 🗣️ **Plan and Execute** complex note management tasks
- 🔍 **Search Intelligently** using hybrid semantic + BM25 search
- 🧠 **Analyze Content** for topics, entities, and sentiment
- 🌐 **Web Integration** to enrich your notes with real-time information
- 📊 **Self-Reflect** and improve responses over time

## 🎨 Architecture Highlights

### 🤖 Multi-Agent Collaboration
- **LangGraph** for sophisticated agentic workflows
- **CrewAI** for role-based agent teams
- **OpenAI Swarm** patterns for lightweight coordination
- **Planning, Execution, Reflection** cycle

### ⚡ Modern Tech Stack
- **Python 3.12+** with strict typing
- **FastAPI** async backend
- **Pydantic v2** for robust data validation
- **Qdrant/Chroma** vector databases
- **LangFuse** observability

### 🔍 Advanced Search
- **Semantic Search** using latest embedding models
- **BM25** for keyword matching
- **Hybrid Search** combining both approaches
- **Agentic Retrieval** with query planning

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/FailedToAchieveOrbit/robust-note-taker.git
cd robust-note-taker

# Install with uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or with pip
pip install -e .
```

### 2. Configuration

```bash
# Create your .env file
cp .env.example .env

# Edit .env with your API keys
OPENAI_API_KEY=your_openai_key_here
VECTOR_PROVIDER=qdrant  # or chroma
VECTOR_URL=http://localhost:6333
```

### 3. Start the Vector Database

```bash
# Start Qdrant with Docker
docker run -p 6333:6333 qdrant/qdrant

# Or start Chroma
chroma run --host localhost --port 8000
```

### 4. Launch the System

```bash
# Start the FastAPI server
agentic-server

# Or use the CLI
agentic-notes "Create a note about quantum computing"

# Or run the agent crew
agentic-agents
```

## 📚 Usage Examples

### 📝 Creating Notes

```bash
# Simple note creation
agentic-notes "Create a note titled 'Meeting Notes' with content about our Q4 planning session"

# AI-enhanced note with analysis
agentic-notes "Analyze this text and create a note: [your text here]"
```

### 🔎 Intelligent Search

```bash
# Semantic search
agentic-notes "Find notes about machine learning algorithms"

# Agentic search with reasoning
agentic-notes "I'm working on a presentation about AI trends. What relevant notes do I have?"
```

### 🤖 Agent Interactions

```python
from agentic_notes import create_note_agent_graph
from agentic_notes.core import AgenticDatabase, VectorStore

# Initialize components
db = AgenticDatabase()
vector_store = VectorStore()

# Create the agentic system
agent_graph = create_note_agent_graph(db, vector_store)

# Process complex queries
result = await agent_graph.process_query(
    "I need to prepare a comprehensive summary of all my AI research notes from the last 3 months, highlighting key trends and breakthrough papers."
)

print(result['answer'])
print(f"Confidence: {result['confidence']}")
print(f"Sources: {result['sources']}")
```

## 🎨 System Architecture

```
┌──────────────────────────┐
│         User Interface         │
│    (CLI, API, Web UI)         │
├──────────────────────────┤
│      Agentic Layer            │
│  ┌──────────────────────┐  │
│  │    LangGraph         │  │
│  │  (Planning &        │  │
│  │   Orchestration)    │  │
│  ├──────────────────────┤  │
│  │      CrewAI         │  │
│  │  (Specialized       │  │
│  │     Agents)        │  │
│  └──────────────────────┘  │
├──────────────────────────┤
│       Core Services           │
│  ┌──────────────────────┐  │
│  │   Vector Store      │  │
│  │ (Qdrant/Chroma)   │  │
│  ├──────────────────────┤  │
│  │    Database        │  │
│  │ (AsyncIO + JSON)  │  │
│  ├──────────────────────┤  │
│  │   Embeddings      │  │
│  │(OpenAI/HuggingF) │  │
│  └──────────────────────┘  │
└──────────────────────────┘
```

## 📈 Key Features

### 🤖 Agentic Capabilities
- **Planning Agent**: Breaks down complex queries into actionable steps
- **Search Agent**: Intelligently searches through your knowledge base
- **Analysis Agent**: Extracts topics, entities, and insights
- **Synthesis Agent**: Combines information into coherent responses
- **Reflection Agent**: Evaluates and improves agent performance

### 🔍 Advanced Search
- **Semantic Search**: Understanding meaning beyond keywords
- **BM25 Ranking**: Statistical relevance scoring
- **Hybrid Approaches**: Best of both worlds
- **Query Expansion**: AI-powered query enhancement

### 📊 Rich Metadata
- **Auto-tagging**: AI-generated tags and categories
- **Entity Extraction**: People, places, organizations
- **Sentiment Analysis**: Emotional tone detection
- **Topic Modeling**: Automatic theme identification

### ⚡ Performance & Scalability
- **Async Architecture**: Non-blocking I/O operations
- **Vector Caching**: Fast embedding retrieval
- **Connection Pooling**: Efficient resource usage
- **Horizontal Scaling**: Multi-worker support

## 🔧 Configuration

The system is highly configurable through environment variables and `.env` files:

```bash
# Core Settings
APP_NAME="Agentic Note Taker"
ENVIRONMENT=development
DEBUG=true

# AI Configuration
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=your_key_here

# Vector Database
VECTOR_PROVIDER=qdrant
VECTOR_URL=http://localhost:6333
VECTOR_COLLECTION_NAME=agentic_notes

# Embedding Settings
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Agent Configuration
AGENT_FRAMEWORK=langgraph
AGENT_MAX_ITERATIONS=10
AGENT_ENABLE_REFLECTION=true

# Observability
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
MONITORING_ENABLE_METRICS=true
```

## 🔬 API Documentation

The system exposes a comprehensive REST API:

```bash
# Start the server
agentic-server

# Visit the interactive docs
open http://localhost:8000/docs
```

### Key Endpoints:

- `POST /notes/` - Create a new note
- `GET /notes/{note_id}` - Retrieve a note
- `PUT /notes/{note_id}` - Update a note
- `DELETE /notes/{note_id}` - Delete a note
- `POST /search/` - Search notes
- `POST /chat/` - Chat with your notes
- `GET /health/` - System health check

## 🧑‍💻 Development

### Setup Development Environment

```bash
# Install development dependencies
uv sync --group development

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest

# Run type checking
mypy src/

# Run linting
ruff check src/

# Format code
black src/
```

### Project Structure

```
src/agentic_notes/
├── __init__.py
├── config.py              # Configuration management
├── agents/                # Agentic system
│   ├── __init__.py
│   ├── graph.py           # LangGraph workflows
│   ├── crew.py            # CrewAI agents
│   └── tools.py           # Agent tools
├── core/                  # Core services
│   ├── database.py        # Data persistence
│   ├── vector_store.py    # Vector operations
│   └── embeddings.py      # Embedding models
├── models/                # Data models
│   ├── note.py            # Note models
│   ├── search.py          # Search models
│   └── agent.py           # Agent models
├── api/                   # FastAPI routes
└── cli.py                 # Command line interface
```

## 📊 Observability & Monitoring

Built-in observability with LangFuse integration:

- 🔍 **Trace Requests**: Follow agent execution paths
- 📊 **Performance Metrics**: Response times, success rates
- 🚫 **Error Tracking**: Automatic error capture and alerts
- 💰 **Cost Monitoring**: Track AI API usage and costs

## 🎆 Advanced Features

### 🌐 Web Integration
- Real-time web search through agent tools
- Automatic fact-checking and source verification
- Content enrichment from external sources

### 📊 Analytics Dashboard
- Note creation trends
- Search pattern analysis
- Agent performance metrics
- Usage insights

### 🔒 Security & Privacy
- Local-first architecture
- Encrypted embeddings storage
- API key management
- Audit logging


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🚀 Roadmap

- [ ] 🧠 **Advanced NLP**: Integration with latest language models
- [ ] 🌐 **Multi-modal**: Support for images, audio, video
- [ ] 🔗 **Integrations**: Obsidian, Notion, Roam Research sync
- [ ] 📱 **Mobile Apps**: iOS and Android applications
- [ ] 🌍 **Collaboration**: Real-time collaborative editing
- [ ] 🚫 **Enterprise**: SSO, advanced security, compliance

---

<div align="center">

**Built with ❤️**

</div>
