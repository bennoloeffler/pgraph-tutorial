# Knowledge Graph Extraction

This document describes the LLM-based knowledge graph extraction system that processes documents during ingestion to extract nodes, edges, summaries, keywords, and embeddings.

## Overview

When a document is ingested into the system, the extraction pipeline:

1. **Reads document content** (text from PDFs, JSON, Markdown, etc.)
2. **Extracts entities (nodes)** - People, organizations, concepts, topics, etc.
3. **Extracts relationships (edges)** - How entities relate to each other
4. **Creates summaries** - Document-level and entity-level summaries
5. **Extracts keywords** - Important terms for keyword search
6. **Generates embeddings** - Vector representations for semantic search

## Architecture

```
Queue File (JSON)
       ↓
  Content Extraction
       ↓
  LLM Processing (GPT-4o-mini)
       ├─→ Node Extraction
       ├─→ Edge Extraction
       ├─→ Summary Generation
       └─→ Keyword Extraction
       ↓
  Embedding Generation (text-embedding-3-small)
       ↓
  PostgreSQL Storage
       ├─→ kg_node (with keywords in properties)
       ├─→ kg_edge
       └─→ kg_node_embedding
```

## LLM Extraction Process

### Step 1: Entity Extraction

The LLM analyzes the document and extracts:

- **Document node** - Represents the whole document
- **Entity nodes** - People, organizations, concepts, topics mentioned
- **Relationships** - How entities connect to each other

### Step 2: Summary Generation

For each extracted node:

- **Short description** (1 sentence) - Used for embedding
- **Summary** (1-2 paragraphs) - Detailed context

### Step 3: Keyword Extraction

Keywords are extracted at two levels:

- **Document keywords** - Main topics/terms
- **Entity keywords** - Specific to each entity

### Step 4: Embedding Generation

Embeddings are generated using OpenAI's `text-embedding-3-small` model (1536 dimensions) for:

- Document summaries
- Entity short descriptions

## Data Model

### kg_node Properties

```json
{
  "title": "Document Title",
  "content_type": "application/pdf",
  "summary": "Document summary...",
  "short_description": "One sentence description",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "text": "Full text content (if available)",
  "extracted_by": "gpt-4o-mini"
}
```

### kg_edge Properties

```json
{
  "description": "Relationship description",
  "extracted_by": "gpt-4o-mini"
}
```

### kg_node_embedding

| Column | Type | Description |
|--------|------|-------------|
| node_id | BIGINT | FK to kg_node |
| model_name | TEXT | "text-embedding-3-small" |
| short_description | TEXT | Text that was embedded |
| embedding | vector(1536) | The embedding vector |

## Edge Types

Common relationship types extracted:

- `mentions` - Document mentions an entity
- `contains` - Document contains a section/topic
- `relates_to` - General relationship
- `authored_by` - Author relationship
- `sent_to` - Email recipient
- `part_of` - Part of a larger entity
- `similar_to` - Semantic similarity

## Node Labels

Common entity types:

- `Document` - The source document
- `Person` - Named individuals
- `Organization` - Companies, institutions
- `Topic` - Subject matters
- `Concept` - Abstract ideas
- `Section` - Document sections
- `Email` - Email messages

## Configuration

Environment variables:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## Usage

The extraction happens automatically during queue ingestion:

```bash
# Process queue with extraction
python -m pgt.p_03_ingest_from_queue
```

## Performance Considerations

- **Batch processing** - Multiple documents can be processed in parallel
- **Token limits** - Large documents are chunked before LLM processing
- **Rate limits** - Implements exponential backoff for API calls
- **Caching** - Embeddings cached to avoid redundant API calls

## Finding Existing Nodes

Before creating new nodes, the system checks for existing nodes to avoid duplicates:

1. **Exact name match** - Same label and name
2. **Semantic similarity** - High embedding similarity (>0.9)

If a match is found, the node is updated rather than duplicated.

## Search Integration

The extracted data enables multiple search strategies:

1. **Keyword search** - Uses `keywords` array in properties
2. **Semantic search** - Uses embeddings with pgvector
3. **Graph traversal** - Navigate relationships
4. **Combined search** - Weighted combination of all strategies
