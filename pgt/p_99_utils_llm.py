"""LLM utility functions kept separate from extraction orchestration.

These functions are pure LLM wrappers so they can be swapped or mocked easily,
following the BBS black-box principle.
"""

import json
import os
from typing import Any

from openai import AsyncOpenAI

EXTRACTION_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Limits guard against excessive payloads
MAX_EXTRACTION_CHARS = 15000
MAX_EMBEDDING_CHARS = 8000

# noqa: E501 - Prompt strings exceed line length for readability
EXTRACTION_PROMPT = """Analyze the document and extract structured information.

Document Title: {title}
Content Type: {content_type}
Content:
{content}

---

Extract the following in JSON format:

1. **document_summary**: A 1-2 paragraph summary of the entire document
2. **document_short_description**: Single sentence (max 100 chars)
3. **document_keywords**: Array of 5-10 important keywords/terms
4. **entities**: Array of entities mentioned in the document, each with:
   - name: Entity name
   - label: Type (Person, Organization, Topic, Concept, etc.)
   - short_description: One sentence description
   - keywords: Array of 3-5 relevant keywords
5. **relationships**: Array of relationships between entities, each with:
   - from_entity: Source entity name
   - to_entity: Target entity name
   - relationship_type: Type (mentions, contains, relates_to, etc.)
   - description: Brief description of the relationship

Important:
- Include the document itself as implicit source for relationships
- Use "Document" as the from_entity for direct document relationships
- Focus on the most important entities (max 10)
- Keywords should be specific and searchable

Respond with valid JSON only, no markdown formatting."""


async def create_openai_client() -> AsyncOpenAI:
    """Create an async OpenAI client."""
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _build_extraction_prompt(title: str, content_type: str, content: str) -> str:
    return EXTRACTION_PROMPT.format(
        title=title,
        content_type=content_type,
        content=content[:MAX_EXTRACTION_CHARS],
    )


async def extract_from_content(
    client: AsyncOpenAI,
    title: str,
    content_type: str,
    content: str,
) -> dict[str, Any]:
    """Extract nodes, edges, summaries, and keywords from document content."""
    prompt = _build_extraction_prompt(title, content_type, content)

    response = await client.chat.completions.create(
        model=EXTRACTION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledge graph extraction assistant. "
                    "Extract structured information from documents for graph "
                    "databases. Always respond with valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    result_text = response.choices[0].message.content or "{}"
    return json.loads(result_text)


async def generate_embedding(
    client: AsyncOpenAI,
    text: str,
) -> list[float]:
    """Generate embedding vector for text."""
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text[:MAX_EMBEDDING_CHARS],
    )
    return response.data[0].embedding


__all__ = [
    "create_openai_client",
    "extract_from_content",
    "generate_embedding",
    "EXTRACTION_MODEL",
    "EMBEDDING_MODEL",
    "MAX_EXTRACTION_CHARS",
    "MAX_EMBEDDING_CHARS",
]
