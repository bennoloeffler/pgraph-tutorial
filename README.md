# pgraph tutorial

A PostgreSQL/Python application demonstrating intelligent data search and retrieval with LLM-powered knowledge graph extraction.

## Features

### 1. Fast Searching Index
Multiple search strategies:
- Keyword search (PostgreSQL full-text)
- Semantic search (embeddings via pgvector)
- SQL queries
- Knowledge graph traversal
- Abstracted source URLs

### 2. Ingestion Pipeline
Automated data processing:
- Queue-based ingestion from multiple sources
- **LLM-powered extraction** (GPT-4o-mini):
  - Entity extraction (nodes)
  - Relationship extraction (edges)
  - Summary generation
  - Keyword extraction
- Embedding generation (text-embedding-3-small)
- PostgreSQL storage with pgvector

### 3. Query Agent (Planned)
AI-powered query interpretation:
- Query translation and interpretation
- User clarification for ambiguous queries
- Strategy formulation
- ReAct pattern implementation
- Dynamic Python script generation

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL with pgvector extension
- OpenAI API key (for LLM extraction)

### Setup

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your database credentials and OpenAI API key

# Initialize database
uv run python -m pgt.p_01_init_db

# Fill queue with sample data
uv run python -m pgt.p_02_fill_queue

# Run ingestion with LLM extraction
uv run python -m pgt.p_03_ingest_from_queue
```

### Environment Variables

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=pgraph_tutorial

# OpenAI (for LLM extraction)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## Architecture

### Data Flow

```
Source Files → Queue (JSON) → Ingestion Pipeline
                                    ↓
                            LLM Extraction (GPT-4o-mini)
                            ├── Nodes (entities)
                            ├── Edges (relationships)
                            ├── Summaries
                            └── Keywords
                                    ↓
                            Embedding Generation
                                    ↓
                            PostgreSQL Storage
                            ├── kg_node
                            ├── kg_edge
                            └── kg_node_embedding
```

### Database Schema

- `kg_label` - Node type labels
- `kg_edge_type` - Relationship type labels
- `kg_node` - Entities with properties (including keywords)
- `kg_edge` - Relationships between nodes
- `kg_node_embedding` - Vector embeddings (1536 dimensions)

## Documentation

- [Knowledge Graph Extraction](docs/knowledge_graph_extraction.md)
- [Source Abstraction](docs/source_abstraction.md)

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Format code
uv run black pgt/ tests/

# Lint
uv run ruff check pgt/ tests/ --fix

# Type check
uv run mypy pgt/
```



