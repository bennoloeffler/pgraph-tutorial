# The Data Monster: Building an Intelligent Knowledge Graph System

A comprehensive tutorial for building a hybrid search system that combines knowledge graphs, semantic search, and intelligent agents.

---

## Overview

This tutorial teaches you how to build a "Data Monster" - a powerful system that can:
- **Store** complex relationships between entities
- **Search** using multiple strategies (keyword, semantic, graph traversal)
- **Answer** natural language questions intelligently
- **Ingest** data from various sources continuously

The system has three main components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  SEARCH INDEX   │ ←── │  AGENT SYSTEM   │ ←── │    INGESTION    │
│                 │     │                 │     │    PIPELINE     │
│ • Knowledge     │     │ • Decomposes    │     │                 │
│   Graph         │     │   Questions     │     │ • Message Queue │
│ • Semantic      │     │ • Routes to     │     │ • Buffering     │
│   Search        │     │   Strategies    │     │ • Background    │
│ • Keyword       │     │ • Reranks       │     │   Workers       │
│   Search        │     │   Results       │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Part 1: The Search Index

The Search Index is the foundation of your Data Monster. It stores data in three complementary ways.

### 1.1 The Knowledge Graph

A knowledge graph stores entities (nodes) and their relationships (edges) with flexible properties.

#### Why PostgreSQL?

You don't need a specialized graph database! PostgreSQL with JSONB gives you:
- **Flexibility**: Add properties without migrations
- **Performance**: GIN indexes on JSONB are fast
- **Familiarity**: Standard SQL queries
- **Scale**: Handles millions of nodes/edges easily

#### Database Schema

```sql
-- Enable vector extension for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- Labels (types of nodes: Company, Product, Location, etc.)
CREATE TABLE kg_label (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

-- Edge types (relationship types: PRODUCES, LOCATED_IN, etc.)
CREATE TABLE kg_edge_type (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

-- Nodes - the entities in your graph
CREATE TABLE kg_node (
    id BIGSERIAL PRIMARY KEY,
    label_id INT NOT NULL REFERENCES kg_label(id),
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    source JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_created_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Edges - relationships between nodes
CREATE TABLE kg_edge (
    id BIGSERIAL PRIMARY KEY,
    src_id BIGINT NOT NULL REFERENCES kg_node(id) ON DELETE CASCADE,
    dst_id BIGINT NOT NULL REFERENCES kg_node(id) ON DELETE CASCADE,
    type_id INT NOT NULL REFERENCES kg_edge_type(id),
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(src_id, dst_id, type_id)
);

-- Performance indexes
CREATE INDEX idx_kg_node_label ON kg_node(label_id);
CREATE INDEX idx_kg_node_props_gin ON kg_node USING GIN (properties);
CREATE INDEX idx_kg_edge_src ON kg_edge(src_id);
CREATE INDEX idx_kg_edge_dst ON kg_edge(dst_id);
CREATE INDEX idx_kg_edge_type ON kg_edge(type_id);
```

Each node stores both flexible `properties` for graph attributes and a `source`
payload that records where the fact came from (file path, queue message, API
metadata, etc.) plus a mandatory `source_created_at` timestamp so you can
distinguish the original event time from when it was ingested.

#### Creating Nodes and Edges

```python
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import json

def get_or_create_label(cur, label_name):
    """Get label ID, creating if needed."""
    cur.execute(
        "INSERT INTO kg_label (name) VALUES (%s) "
        "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name "
        "RETURNING id",
        (label_name,)
    )
    return cur.fetchone()['id']

def create_node(label_name, properties, source, source_created_at):
    """Create a node in the knowledge graph."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Get or create the label
    label_id = get_or_create_label(cur, label_name)

    # Insert the node
    cur.execute(
        "INSERT INTO kg_node (label_id, properties, source, source_created_at) "
        "VALUES (%s, %s, %s, %s) RETURNING id",
        (
            label_id,
            json.dumps(properties),
            json.dumps(source),
            source_created_at,
        )
    )
    node_id = cur.fetchone()['id']

    conn.commit()
    cur.close()
    conn.close()

    return node_id

def create_edge(src_id, dst_id, edge_type, properties=None):
    """Create an edge between two nodes."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Get or create edge type
    cur.execute(
        "INSERT INTO kg_edge_type (name) VALUES (%s) "
        "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name "
        "RETURNING id",
        (edge_type,)
    )
    type_id = cur.fetchone()[0]

    # Insert the edge
    cur.execute(
        "INSERT INTO kg_edge (src_id, dst_id, type_id, properties) "
        "VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (src_id, dst_id, type_id) DO NOTHING",
        (src_id, dst_id, type_id, json.dumps(properties or {}))
    )

    conn.commit()
    cur.close()
    conn.close()

# Example usage
company_id = create_node(
    "Company",
    {
        "name": "Acme Corp",
        "industry": "Manufacturing",
        "employees": 500
    },
    {"ingest_path": "/docs/acme_overview.pdf"},
    datetime.utcnow(),
)

product_id = create_node(
    "Product",
    {
        "name": "Widget Pro",
        "category": "Industrial Equipment"
    },
    {"ingest_path": "/docs/products/widget_pro.pdf"},
    datetime.utcnow(),
)

create_edge(company_id, product_id, "PRODUCES")
```

### 1.2 Semantic Search with pgvector

Semantic search finds results by meaning, not just keywords. We use embeddings (vector representations of text) stored with pgvector.

#### Embeddings Table

```sql
-- Node embeddings for semantic search
CREATE TABLE kg_node_embedding (
    node_id BIGINT PRIMARY KEY REFERENCES kg_node(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL DEFAULT 'text-embedding-3-small',
    short_description TEXT NOT NULL,  -- Human-readable description
    embedding vector(1536) NOT NULL,   -- The vector (1536 dims for OpenAI)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- HNSW index for fast similarity search
CREATE INDEX idx_node_embedding_hnsw
ON kg_node_embedding
USING hnsw (embedding vector_cosine_ops);
```

#### Generating and Storing Embeddings

```python
from openai import OpenAI

def generate_embedding(text):
    """Generate an embedding for text using OpenAI."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def store_embedding(node_id, description, embedding):
    """Store an embedding for a node."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Convert to PostgreSQL vector format
    embedding_str = '[' + ','.join(map(str, embedding)) + ']'

    cur.execute("""
        INSERT INTO kg_node_embedding (node_id, short_description, embedding)
        VALUES (%s, %s, %s::vector)
        ON CONFLICT (node_id) DO UPDATE
        SET short_description = EXCLUDED.short_description,
            embedding = EXCLUDED.embedding
    """, (node_id, description, embedding_str))

    conn.commit()
    cur.close()
    conn.close()

# Example: Create node and embedding together
company_id = create_node("Company", {"name": "Acme Corp", "industry": "Manufacturing"})
description = "Acme Corp - Manufacturing company producing industrial widgets"
embedding = generate_embedding(description)
store_embedding(company_id, description, embedding)
```

#### Semantic Search Query

```python
def semantic_search(query_text, limit=10):
    """Find nodes semantically similar to the query."""
    # Generate embedding for the query
    query_embedding = generate_embedding(query_text)
    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT
            e.node_id,
            e.short_description,
            e.embedding <=> %s::vector AS distance,
            n.properties->>'name' AS name,
            l.name AS label
        FROM kg_node_embedding e
        JOIN kg_node n ON e.node_id = n.id
        JOIN kg_label l ON n.label_id = l.id
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """, (embedding_str, embedding_str, limit))

    results = cur.fetchall()
    cur.close()
    conn.close()

    return [dict(r) for r in results]

# Usage
results = semantic_search("companies that make automation equipment")
for r in results:
    print(f"{r['name']} ({r['label']}) - distance: {r['distance']:.3f}")
```

### 1.3 Keyword Search with SQL

For exact matches and filtering, use standard SQL with JSONB operators.

```python
def keyword_search(label_type, field, value):
    """Search nodes by exact field match."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT n.id, n.properties
        FROM kg_node n
        JOIN kg_label l ON n.label_id = l.id
        WHERE l.name = %s
          AND n.properties->>%s ILIKE %s
    """, (label_type, field, f"%{value}%"))

    results = cur.fetchall()
    cur.close()
    conn.close()

    return [dict(r) for r in results]

# Usage
results = keyword_search("Company", "name", "Acme")
```

### 1.4 Graph Traversal with Recursive CTEs

Find connected nodes up to N hops away:

```python
def traverse_from_node(start_id, max_depth=2):
    """Traverse graph from a starting node."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        WITH RECURSIVE traversal AS (
            -- Start node
            SELECT
                0 AS depth,
                n.id AS node_id,
                n.properties,
                l.name AS label
            FROM kg_node n
            JOIN kg_label l ON n.label_id = l.id
            WHERE n.id = %s

            UNION ALL

            -- Connected nodes
            SELECT
                t.depth + 1,
                n2.id,
                n2.properties,
                l2.name
            FROM traversal t
            JOIN kg_edge e ON (e.src_id = t.node_id OR e.dst_id = t.node_id)
            JOIN kg_node n2 ON n2.id = CASE
                WHEN e.src_id = t.node_id THEN e.dst_id
                ELSE e.src_id
            END
            JOIN kg_label l2 ON n2.label_id = l2.id
            WHERE t.depth < %s
        )
        SELECT DISTINCT depth, node_id, properties, label
        FROM traversal
        ORDER BY depth, label
    """, (start_id, max_depth))

    results = cur.fetchall()
    cur.close()
    conn.close()

    return [dict(r) for r in results]
```

### 1.5 Hybrid Search Strategy

The real power comes from combining all three search types:

```python
def hybrid_search(query, use_semantic=True, use_graph=True):
    """Combine semantic search with graph traversal."""
    results = []

    if use_semantic:
        # Step 1: Find entry points via semantic search
        entry_points = semantic_search(query, limit=5)
        entry_node_ids = [ep['node_id'] for ep in entry_points]

        if use_graph and entry_node_ids:
            # Step 2: Traverse graph from entry points
            for node_id in entry_node_ids:
                connected = traverse_from_node(node_id, max_depth=2)
                results.extend(connected)

    # Deduplicate by node_id
    seen = set()
    unique_results = []
    for r in results:
        if r['node_id'] not in seen:
            seen.add(r['node_id'])
            unique_results.append(r)

    return unique_results
```

---

## Part 2: The Agent System

The Agent System is the "brain" that interprets natural language questions and orchestrates the search strategies.

### 2.1 Query Classification

Not all queries are the same. The agent first classifies what type of query it's dealing with:

```python
def classify_query_type(user_query):
    """
    Classify query to determine best search strategy.

    Returns:
        'selection' - Exact filtering (COUNT, WHERE, exact matches)
        'semantic'  - Fuzzy conceptual (related to, similar to)
        'hybrid'    - Ambiguous - try both
    """
    query_lower = user_query.lower()

    # Selection patterns (sharp, exact)
    selection_patterns = [
        r'\b(all|every|each)\s+(companies?|products?)',
        r'\b(how many|count|number of)',
        r'\b(which|what)\s+(types?|kinds?)\s+exist',
        r'\bwhere\s+.*\s+(=|>|<)',
    ]

    # Semantic patterns (fuzzy, conceptual)
    semantic_patterns = [
        r'\b(related to|similar to|like|about)',
        r'\b(find|search|discover)\s+\w+',
        r'\bthat\s+(do|work|focus|specialize)',
    ]

    import re

    selection_score = sum(1 for p in selection_patterns if re.search(p, query_lower))
    semantic_score = sum(1 for p in semantic_patterns if re.search(p, query_lower))

    if selection_score > semantic_score:
        return 'selection'
    elif semantic_score > selection_score:
        return 'semantic'
    else:
        return 'hybrid'

# Examples
classify_query_type("Show me all companies")          # → 'selection'
classify_query_type("Find companies related to AI")   # → 'semantic'
classify_query_type("Companies in Munich")            # → 'hybrid'
```

### 2.2 Query Planning

For complex queries, the agent creates an execution plan:

```python
def plan_query(user_query, schema_info):
    """
    Create execution plan before generating code.

    Returns a plan with:
    - target_entities: What to find (Company, Product, etc.)
    - required_relationships: Edges to traverse
    - filters: Conditions to apply
    - aggregations: COUNT, AVG, etc.
    - execution_strategy: direct_sql, semantic_search, traversal, hybrid
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a query planner for a knowledge graph.
                Return JSON with: target_entities, required_relationships,
                filters, aggregations, execution_strategy, feasibility."""
            },
            {
                "role": "user",
                "content": f"Schema: {schema_info}\n\nQuery: {user_query}"
            }
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

# Example plan for "Companies in Munich that produce automation equipment"
# {
#   "target_entities": ["Company"],
#   "required_relationships": ["LOCATED_IN", "PRODUCES"],
#   "filters": [{"field": "location", "value": "Munich"}],
#   "aggregations": null,
#   "execution_strategy": "traversal",
#   "feasibility": "feasible"
# }
```

### 2.3 Query Decomposition

Complex queries are broken into simpler sub-queries:

```python
def decompose_query(user_query):
    """
    Break complex queries into simpler sub-queries.

    Only decompose if query has multiple conditions.
    """
    query_lower = user_query.lower()

    # Check complexity
    complexity_indicators = [' and ', ' with ', ' that ', ' where ', ' having ']
    complexity = sum(1 for i in complexity_indicators if i in query_lower)

    if complexity < 2:
        return None  # Simple query, no decomposition needed

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Break complex queries into sub-queries.
                Return JSON with: should_decompose (bool), sub_queries (list),
                combination_strategy (intersect, union, filter, aggregate)."""
            },
            {
                "role": "user",
                "content": f"Query: {user_query}"
            }
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

# Example decomposition:
# Query: "Companies in Munich with more than 500 employees that produce automation equipment"
# {
#   "should_decompose": true,
#   "sub_queries": [
#     "Find companies in Munich",
#     "Filter by employees > 500",
#     "Filter by produces automation equipment"
#   ],
#   "combination_strategy": "intersect"
# }
```

### 2.4 Code Generation

The agent generates Python/SQL code to answer queries:

```python
def generate_query_code(user_query, entry_points=None, previous_error=None):
    """Generate Python code to answer the user's query."""

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Build context for the LLM
    schema_info = get_schema_info(entry_points)

    system_prompt = """You are a Python code generator for a knowledge graph.

    Available:
    - cur: Database cursor (RealDictCursor)
    - generate_embedding(text): Generate embeddings

    Rules:
    1. Return ONLY valid Python code
    2. Use cur.execute() then cur.fetchall()
    3. Convert results: results = [dict(r) for r in results]
    4. Set 'results' variable with your answer
    5. For vector search: embedding <=> %s::vector
    6. Join kg_label to filter by node type
    7. Use properties->>'field' for JSONB access
    """

    user_prompt = f"""Schema: {schema_info}

    Query: {user_query}

    Generate Python code to answer this query:"""

    if previous_error:
        user_prompt += f"\n\nPrevious error: {previous_error}\nFix the code."

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    code = response.choices[0].message.content

    # Clean up the code (remove markdown formatting)
    code = code.replace("```python", "").replace("```", "").strip()

    return code
```

### 2.5 Code Execution with Retry

Execute generated code safely with automatic retry on errors:

```python
def execute_query(user_query, max_retries=3):
    """Execute a natural language query with retry on errors."""

    # Step 1: Classify query type
    query_type = classify_query_type(user_query)

    # Step 2: Find entry points if semantic
    entry_points = None
    if query_type in ['semantic', 'hybrid']:
        entry_points = find_entry_points(user_query)

    # Step 3: Generate and execute code
    code = None
    error = None

    for attempt in range(max_retries):
        code = generate_query_code(
            user_query,
            entry_points=entry_points,
            previous_error=error
        )

        try:
            # Execute in safe environment
            results = execute_code_safely(code)
            return results
        except Exception as e:
            error = str(e)
            print(f"Attempt {attempt + 1} failed: {error}")

    raise Exception(f"Failed after {max_retries} attempts: {error}")

def execute_code_safely(code):
    """Execute generated code in a controlled environment."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Create execution environment
    local_vars = {
        'cur': cur,
        'conn': conn,
        'generate_embedding': generate_embedding,
        'results': None
    }

    try:
        exec(code, {}, local_vars)
        results = local_vars.get('results', [])
    finally:
        cur.close()
        conn.close()

    return results
```

### 2.6 Result Reranking

After getting results, rerank them by relevance to the original query:

```python
def rerank_results(results, query, top_k=10):
    """Rerank results by semantic similarity to the query."""
    if not results:
        return results

    # Generate query embedding
    query_embedding = generate_embedding(query)

    # Generate embeddings for each result (use name/description)
    scored_results = []
    for result in results:
        description = result.get('name', '') or result.get('short_description', '')
        if description:
            result_embedding = generate_embedding(description)
            # Calculate cosine similarity
            similarity = cosine_similarity(query_embedding, result_embedding)
            scored_results.append((similarity, result))

    # Sort by similarity (descending)
    scored_results.sort(key=lambda x: x[0], reverse=True)

    return [r for _, r in scored_results[:top_k]]

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    import numpy as np
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## Part 3: The Ingestion Pipeline

The Ingestion Pipeline handles getting data into your system efficiently and reliably.

### 3.1 The Embedding Queue

Embeddings are expensive (API calls), so we queue them for batch processing:

```sql
-- Embedding queue table
CREATE TABLE kg_embedding_queue (
    id BIGSERIAL PRIMARY KEY,
    entity_type TEXT NOT NULL CHECK (entity_type IN ('node', 'edge')),
    entity_id BIGINT NOT NULL,
    entry_point_text TEXT NOT NULL,  -- Text to embed
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT,
    retry_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    UNIQUE(entity_type, entity_id)  -- Prevent duplicates
);

-- Index for efficient queue processing
CREATE INDEX idx_embedding_queue_status
ON kg_embedding_queue(status, created_at);
```

### 3.2 Queue Management

```python
class EmbeddingQueue:
    """Manage embedding generation queue."""

    def enqueue(self, entity_type, entity_id, entry_point_text):
        """
        Add item to queue. Non-blocking, returns immediately.
        """
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO kg_embedding_queue (entity_type, entity_id, entry_point_text)
            VALUES (%s, %s, %s)
            ON CONFLICT (entity_type, entity_id) DO NOTHING
        """, (entity_type, entity_id, entry_point_text))

        conn.commit()
        cur.close()
        conn.close()

    def get_pending_batch(self, batch_size=100):
        """
        Get next batch and mark as processing.
        Uses FOR UPDATE SKIP LOCKED for safe concurrent processing.
        """
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Atomically claim items
        cur.execute("""
            UPDATE kg_embedding_queue
            SET status = 'processing', processed_at = NOW()
            WHERE id IN (
                SELECT id FROM kg_embedding_queue
                WHERE status = 'pending'
                ORDER BY created_at
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, entity_type, entity_id, entry_point_text
        """, (batch_size,))

        results = cur.fetchall()
        conn.commit()
        cur.close()
        conn.close()

        return [
            {
                'id': r[0],
                'entity_type': r[1],
                'entity_id': r[2],
                'entry_point_text': r[3]
            }
            for r in results
        ]

    def mark_completed(self, queue_id):
        """Mark item as completed."""
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute(
            "UPDATE kg_embedding_queue SET status = 'completed' WHERE id = %s",
            (queue_id,)
        )

        conn.commit()
        cur.close()
        conn.close()

    def mark_failed(self, queue_id, error_message):
        """Mark item as failed with error."""
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            UPDATE kg_embedding_queue
            SET status = 'failed',
                error_message = %s,
                retry_count = retry_count + 1
            WHERE id = %s
        """, (error_message[:500], queue_id))

        conn.commit()
        cur.close()
        conn.close()
```

### 3.3 Background Worker

Process the queue in batches (more efficient than one-by-one):

```python
class EmbeddingWorker:
    """Background worker that processes embedding queue."""

    def __init__(self, batch_size=100, poll_interval=5):
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.queue = EmbeddingQueue()
        self.running = False

    def process_batch(self):
        """Process one batch of pending items."""
        items = self.queue.get_pending_batch(self.batch_size)

        if not items:
            return 0

        # Extract texts for batch embedding
        texts = [item['entry_point_text'] for item in items]

        try:
            # Generate all embeddings in one API call
            embeddings = generate_embeddings_batch(texts)
        except Exception as e:
            # Mark all as failed
            for item in items:
                self.queue.mark_failed(item['id'], str(e))
            raise

        # Separate by entity type
        node_data = []
        edge_data = []

        for i, item in enumerate(items):
            if item['entity_type'] == 'node':
                node_data.append((
                    item['entity_id'],
                    item['entry_point_text'],
                    embeddings[i]
                ))
            else:
                edge_data.append((
                    item['entity_id'],
                    item['entry_point_text'],
                    embeddings[i]
                ))

        # Store embeddings in bulk
        if node_data:
            store_embeddings_batch(node_data)
        if edge_data:
            store_edge_embeddings_batch(edge_data)

        # Mark all as completed
        for item in items:
            self.queue.mark_completed(item['id'])

        return len(items)

    def run(self):
        """Run worker continuously."""
        self.running = True
        print("Embedding worker started")

        while self.running:
            try:
                processed = self.process_batch()
                if processed > 0:
                    print(f"Processed {processed} embeddings")
                    time.sleep(2)  # Brief pause between batches
                else:
                    time.sleep(self.poll_interval)  # Wait for new items
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(self.poll_interval)

        print("Worker stopped")

def generate_embeddings_batch(texts, batch_size=100):
    """Generate embeddings for multiple texts efficiently."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
```

### 3.4 Automatic Queuing on Node Creation

Modify your `create_node` function to automatically queue embeddings:

```python
def create_node_with_embedding(label_name, properties):
    """Create node and queue its embedding automatically."""
    # Create the node
    node_id = create_node(label_name, properties)

    # Generate entry point text
    name = properties.get('name', 'Unknown')
    description = properties.get('description', '')
    entry_point_text = f"{label_name}: {name}"
    if description:
        entry_point_text += f" - {description}"

    # Queue for embedding generation
    queue = EmbeddingQueue()
    queue.enqueue('node', node_id, entry_point_text)

    return node_id
```

### 3.5 Web Research and Ingestion

For ingesting data from external sources like web research:

```python
def research_and_ingest(company_name, iterations=5):
    """Research a company and add to knowledge graph."""

    # Step 1: Search the web
    search_results = web_search(company_name, iterations)

    # Step 2: Extract structured data using LLM
    company_info = extract_company_info(search_results, company_name)

    if not company_info:
        return None

    # Step 3: Create nodes
    # Company node
    company_id = create_node_with_embedding("Company", {
        "name": company_info.get('name'),
        "industry": company_info.get('industry'),
        "employees": company_info.get('employees'),
        "location": company_info.get('location'),
        "description": company_info.get('description'),
        "revenue": company_info.get('revenue')
    })

    # Industry node
    industry_name = company_info.get('industry')
    if industry_name:
        industry_id = create_node_with_embedding("Industry", {
            "name": industry_name,
            "description": f"Industry: {industry_name}"
        })
        create_edge(company_id, industry_id, "OPERATES_IN")

    # Location node
    location_name = company_info.get('location')
    if location_name:
        location_id = create_node_with_embedding("Location", {
            "name": location_name
        })
        create_edge(company_id, location_id, "LOCATED_IN")

    # Product nodes
    for product in company_info.get('products', []):
        product_id = create_node_with_embedding("Product", {
            "name": product.get('name'),
            "category": product.get('category')
        })
        create_edge(company_id, product_id, "PRODUCES")

    # Document nodes
    for doc in company_info.get('documents', []):
        doc_id = create_node_with_embedding("Document", {
            "url": doc.get('url'),
            "title": doc.get('title'),
            "type": doc.get('type')
        })
        create_edge(company_id, doc_id, "HAS_DOCUMENT")

    return company_id

def extract_company_info(search_results, company_name):
    """Use LLM to extract structured company info from search results."""
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """Extract structured company information.
                Return JSON with: name, industry, employees (int), location,
                description, revenue, products (list), documents (list of {url, title, type})"""
            },
            {
                "role": "user",
                "content": f"Company: {company_name}\n\nSearch results:\n{search_results}"
            }
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
```

---

## Part 4: Putting It All Together

### 4.1 Complete System Setup

```python
# setup.py - Complete system setup

def setup_complete_system():
    """Set up the entire Data Monster system."""

    # 1. Create database schema
    print("Setting up database schema...")
    setup_schema()  # Creates all tables and indexes

    # 2. Start embedding worker (in background)
    print("Starting embedding worker...")
    worker = EmbeddingWorker()
    import threading
    worker_thread = threading.Thread(target=worker.run)
    worker_thread.daemon = True
    worker_thread.start()

    print("System ready!")
    return worker
```

### 4.2 Building the Knowledge Graph

```python
def build_knowledge_graph(data_source):
    """Build knowledge graph from data source."""

    # Example: Building from company data
    companies = fetch_company_data(data_source)

    for company in companies:
        # Create company node
        company_id = create_node_with_embedding("Company", {
            "name": company['name'],
            "industry": company['industry'],
            "employees": company.get('employees'),
            "location": company.get('location')
        })

        # Create related nodes and edges
        # Industry
        industry_id = create_node_with_embedding("Industry", {
            "name": company['industry']
        })
        create_edge(company_id, industry_id, "OPERATES_IN")

        # Location
        if company.get('location'):
            location_id = create_node_with_embedding("Location", {
                "name": company['location']
            })
            create_edge(company_id, location_id, "LOCATED_IN")

        # Products
        for product in company.get('products', []):
            product_id = create_node_with_embedding("Product", {
                "name": product
            })
            create_edge(company_id, product_id, "PRODUCES")

    print(f"Created {len(companies)} companies with relationships")
    print("Embeddings are being generated in background...")
```

### 4.3 Querying the System

```python
def query_data_monster(question):
    """
    Ask the Data Monster a question in natural language.

    Examples:
    - "Show me all companies"
    - "Find companies related to automation"
    - "What products does Acme Corp make?"
    - "Companies in Munich with more than 500 employees"
    """
    print(f"\nQuestion: {question}")
    print("-" * 50)

    # 1. Classify query
    query_type = classify_query_type(question)
    print(f"Query type: {query_type}")

    # 2. Find entry points (for semantic queries)
    entry_points = None
    if query_type in ['semantic', 'hybrid']:
        entry_points = find_entry_points(question, limit=5)
        if entry_points['nodes']:
            print(f"Found {len(entry_points['nodes'])} entry points")

    # 3. Plan the query
    plan = plan_query(question, get_schema_info())
    print(f"Strategy: {plan.get('execution_strategy')}")

    # 4. Execute
    results = execute_query(question, entry_points=entry_points)

    # 5. Rerank by relevance
    if results:
        results = rerank_results(results, question)

    # 6. Display results
    print(f"\nResults ({len(results)} found):")
    for i, result in enumerate(results[:10], 1):
        name = result.get('name') or result.get('company_name') or 'Unknown'
        print(f"  {i}. {name}")

    return results

# Example usage
if __name__ == "__main__":
    # Setup
    setup_complete_system()

    # Build graph
    build_knowledge_graph("sample_data.csv")

    # Wait for embeddings (in real system, this runs continuously)
    import time
    time.sleep(10)

    # Query
    query_data_monster("Find companies that make automation equipment")
    query_data_monster("What industries are represented in Munich?")
    query_data_monster("Show me all products produced by companies with over 1000 employees")
```

---

## Part 5: Best Practices

### 5.1 Schema Design

1. **Use lookup tables for labels and edge types** - Keeps your data normalized and efficient
2. **Keep JSONB properties flat** - Avoid deeply nested structures
3. **Index only frequently queried properties** - GIN indexes are powerful but expensive

### 5.2 Embedding Strategy

1. **Batch everything** - Generate embeddings in batches, not one-by-one
2. **Use a queue** - Don't block on embedding generation
3. **Keep descriptions concise** - 1-2 sentences is ideal for embeddings

### 5.3 Query Optimization

1. **Classify first** - Don't use semantic search for exact matches
2. **Limit traversal depth** - Stay within 3-4 hops for performance
3. **Use hybrid strategically** - Combine approaches for best results

### 5.4 Agent Design

1. **Plan before executing** - Schema-aware planning prevents errors
2. **Decompose complex queries** - Simpler sub-queries are more reliable
3. **Retry with context** - Include error messages in retry prompts

### 5.5 Production Considerations

1. **Monitor the queue** - Track pending/failed items
2. **Rate limit API calls** - Respect embedding API limits
3. **Log everything** - Debug complex query failures
4. **Test with real queries** - Build a test suite of expected questions

---

## Summary

The Data Monster architecture provides:

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Search Index** | Store and search data | Knowledge graph + Semantic search + SQL |
| **Agent System** | Interpret questions | Classification + Planning + Code generation |
| **Ingestion Pipeline** | Add data | Queue + Batch processing + Background workers |

Together, these components create a powerful system that can:
- Store complex relationships between any types of entities
- Answer natural language questions using the best search strategy
- Scale to millions of nodes while maintaining fast queries
- Continuously ingest new data without blocking

Start simple, test thoroughly, and iterate!
