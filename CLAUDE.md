# CLAUDE.md


This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pgraph-tutorial is a PostgreSQL/Python application for intelligent data search and retrieval. The system combines multiple search strategies (keyword, embeddings, SQL, knowledge graph) with an ingestion pipeline and an AI agent for query interpretation.

## Blackbox design
Whenever you plan/design/architect/implemnt:
Use CLAUDE_BBS.md
In order to create modules that can be used like black boxes.

## STEAL FROM OTHER PROJECT
Whenever the user talks of stealing, then have a deep look to folder
../pgraph
There is almost all you need to know do build this tutorial. 

## Architecture (Planned)

### Three Core Modules

1. **Search Index** - Multi-strategy search system
   - Keyword search
   - Vector embeddings for semantic search
   - SQL queries
   - Knowledge graph (nodes/edges)
   - Abstracted URLs pointing to data sources

2. **Ingestion Pipeline** - Data processing queue
   - Multiple source inputs
   - Keyword extraction
   - Node/edge extraction for knowledge graph
   - Document/email summarization
   - Embedding generation
   - PostgreSQL storage

3. **Query Agent** - AI-powered query interpreter
   - Query translation and interpretation
   - User clarification for ambiguous queries
   - Query strategy formulation
   - ReAct pattern implementation
   - Dynamic Python script generation for complex queries

## Project Structure

- **`pgt/`** - Python source code
- **`tests/`** - Test suite

## Development Commands

```bash
# Package management (ALWAYS use uv, NEVER pip)
uv add <package>         # Install new dependency
uv sync                  # Sync dependencies
uv remove <package>      # Remove package

# Environment
source .env              # Load environment variables

# Testing
uv run pytest            # Run all tests
uv run pytest tests/test_file.py -v  # Run specific test file
uv run pytest -v -k "test_name"      # Run specific test

# Quality checks
uv run black .           # Format code
uv run ruff check --fix . # Lint
uv run mypy pgt/         # Type checking
```

## Configuration

- **`.env`** - Database credentials and API keys (never commit)
- **`pyproject.toml`** - Dependencies and tool configuration (to be created)

## Key Design Considerations

- PostgreSQL with pgvector extension for embeddings
- Async Python for concurrent processing
- Clean separation between search strategies
- Queue-based ingestion for reliability
- Agent uses multiple search strategies based on query intent
