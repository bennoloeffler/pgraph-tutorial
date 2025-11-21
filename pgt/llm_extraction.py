"""LLM-based extraction of nodes, edges, summaries, keywords, and embeddings."""

from typing import Any

import asyncpg
from openai import AsyncOpenAI

from pgt.p_99_utils_llm import (
    EMBEDDING_MODEL,
    EXTRACTION_MODEL,
    extract_from_content,
    generate_embedding,
)


async def find_existing_node(
    conn: asyncpg.Connection,
    label_id: int,
    name: str,
) -> int | None:
    """Find existing node by label and name."""
    result = await conn.fetchval(
        """
        SELECT id FROM kg_node
        WHERE label_id = $1 AND properties->>'title' = $2
        """,
        label_id,
        name,
    )
    return int(result) if result else None


async def get_or_create_edge_type(conn: asyncpg.Connection, name: str) -> int:
    """Get or create an edge type."""
    return await conn.fetchval(
        """
        INSERT INTO kg_edge_type (name) VALUES ($1)
        ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
        RETURNING id
        """,
        name,
    )


async def create_entity_node(
    conn: asyncpg.Connection,
    label_id: int,
    entity: dict[str, Any],
    document_node_id: int,
) -> int:
    """Create a node for an extracted entity."""
    properties = {
        "title": entity["name"],
        "short_description": entity.get("short_description", ""),
        "keywords": entity.get("keywords", []),
        "extracted_from_node": document_node_id,
        "extracted_by": EXTRACTION_MODEL,
    }

    return await conn.fetchval(
        """
        INSERT INTO kg_node (label_id, properties, source, source_created_at)
        VALUES ($1, $2, $3, NOW())
        RETURNING id
        """,
        label_id,
        properties,
        {"extracted_from": document_node_id},
    )


async def create_edge(
    conn: asyncpg.Connection,
    src_id: int,
    dst_id: int,
    type_id: int,
    description: str,
) -> int | None:
    """Create an edge between two nodes."""
    try:
        return await conn.fetchval(
            """
            INSERT INTO kg_edge (src_id, dst_id, type_id, properties)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (src_id, dst_id, type_id) DO UPDATE
            SET properties = EXCLUDED.properties
            RETURNING id
            """,
            src_id,
            dst_id,
            type_id,
            {"description": description, "extracted_by": EXTRACTION_MODEL},
        )
    except Exception:
        return None


async def store_embedding(
    conn: asyncpg.Connection,
    node_id: int,
    short_description: str,
    embedding: list[float],
) -> None:
    """Store embedding for a node."""
    await conn.execute(
        """
        INSERT INTO kg_node_embedding
            (node_id, model_name, short_description, embedding)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (node_id) DO UPDATE
        SET model_name = EXCLUDED.model_name,
            short_description = EXCLUDED.short_description,
            embedding = EXCLUDED.embedding,
            created_at = NOW()
        """,
        node_id,
        EMBEDDING_MODEL,
        short_description,
        embedding,
    )


async def process_extraction(
    conn: asyncpg.Connection,
    client: AsyncOpenAI,
    document_node_id: int,
    title: str,
    content_type: str,
    content: str,
    get_or_create_label_func,
) -> dict[str, Any]:
    """
    Process full extraction pipeline for a document.

    Returns stats about what was extracted.
    """
    # Extract information using LLM
    extraction = await extract_from_content(client, title, content_type, content)

    # Update document node with extracted info
    doc_properties = {
        "summary": extraction.get("document_summary", ""),
        "short_description": extraction.get("document_short_description", ""),
        "keywords": extraction.get("document_keywords", []),
        "extracted_by": EXTRACTION_MODEL,
    }

    await conn.execute(
        """
        UPDATE kg_node
        SET properties = properties || $1
        WHERE id = $2
        """,
        doc_properties,
        document_node_id,
    )

    # Generate and store embedding for document
    doc_description = extraction.get("document_short_description", title)
    doc_embedding = await generate_embedding(client, doc_description)
    await store_embedding(conn, document_node_id, doc_description, doc_embedding)

    # Create entity nodes and their embeddings
    entity_nodes: dict[str, int] = {"Document": document_node_id}
    entities_created = 0

    for entity in extraction.get("entities", []):
        entity_name = entity.get("name", "")
        entity_label = entity.get("label", "Entity")

        if not entity_name:
            continue

        label_id = await get_or_create_label_func(conn, entity_label)

        # Check for existing node
        existing_id = await find_existing_node(conn, label_id, entity_name)
        if existing_id:
            entity_nodes[entity_name] = existing_id
            continue

        # Create new entity node
        node_id = await create_entity_node(conn, label_id, entity, document_node_id)
        entity_nodes[entity_name] = node_id
        entities_created += 1

        # Generate embedding for entity
        entity_desc = entity.get("short_description", entity_name)
        entity_embedding = await generate_embedding(client, entity_desc)
        await store_embedding(conn, node_id, entity_desc, entity_embedding)

    # Create edges
    edges_created = 0
    for rel in extraction.get("relationships", []):
        from_entity = rel.get("from_entity", "")
        to_entity = rel.get("to_entity", "")
        rel_type = rel.get("relationship_type", "relates_to")
        description = rel.get("description", "")

        # Map "Document" to the actual document node
        src_id = entity_nodes.get(from_entity)
        dst_id = entity_nodes.get(to_entity)

        if not src_id or not dst_id:
            continue

        type_id = await get_or_create_edge_type(conn, rel_type)
        edge_id = await create_edge(conn, src_id, dst_id, type_id, description)
        if edge_id:
            edges_created += 1

    return {
        "entities_created": entities_created,
        "edges_created": edges_created,
        "keywords_count": len(extraction.get("document_keywords", [])),
        "has_summary": bool(extraction.get("document_summary")),
    }
