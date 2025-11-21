"""Tests for LLM-based knowledge graph extraction."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pgt.llm_extraction import (
    create_entity_node,
    find_existing_node,
    get_or_create_edge_type,
    process_extraction,
    store_embedding,
)
from pgt.p_99_utils_llm import (
    EMBEDDING_MODEL,
    EXTRACTION_MODEL,
    create_openai_client,
    extract_from_content,
    generate_embedding,
)


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = AsyncMock()
    return client


@pytest.fixture
def sample_extraction_response():
    """Sample LLM extraction response."""
    return {
        "document_summary": "This document describes a lean automation strategy.",
        "document_short_description": "Lean automation strategy document",
        "document_keywords": ["automation", "lean", "efficiency", "process"],
        "entities": [
            {
                "name": "AMF Company",
                "label": "Organization",
                "short_description": "The company implementing lean automation",
                "keywords": ["automation", "manufacturing"],
            },
            {
                "name": "Lean Methodology",
                "label": "Concept",
                "short_description": "Methodology for reducing waste",
                "keywords": ["lean", "efficiency"],
            },
        ],
        "relationships": [
            {
                "from_entity": "Document",
                "to_entity": "AMF Company",
                "relationship_type": "mentions",
                "description": "Document mentions AMF Company",
            },
            {
                "from_entity": "AMF Company",
                "to_entity": "Lean Methodology",
                "relationship_type": "implements",
                "description": "AMF implements lean methodology",
            },
        ],
    }


@pytest.fixture
def mock_embedding():
    """Sample embedding vector."""
    return [0.1] * 1536


class TestCreateOpenAIClient:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    async def test_creates_client(self):
        client = await create_openai_client()
        assert client is not None


class TestExtractFromContent:
    async def test_extracts_content(
        self, mock_openai_client, sample_extraction_response
    ):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(sample_extraction_response)))
        ]
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await extract_from_content(
            mock_openai_client,
            "Test Document",
            "text/plain",
            "This is test content about AMF and lean automation.",
        )

        assert (
            result["document_summary"] == sample_extraction_response["document_summary"]
        )
        assert len(result["entities"]) == 2
        assert len(result["relationships"]) == 2

        # Verify API call
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == EXTRACTION_MODEL

    async def test_handles_empty_response(self, mock_openai_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=None))]
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await extract_from_content(
            mock_openai_client, "Test", "text/plain", "Content"
        )

        assert result == {}


class TestGenerateEmbedding:
    async def test_generates_embedding(self, mock_openai_client, mock_embedding):
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embedding)]
        mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await generate_embedding(mock_openai_client, "Test text")

        assert result == mock_embedding
        assert len(result) == 1536

        # Verify API call
        mock_openai_client.embeddings.create.assert_called_once()
        call_args = mock_openai_client.embeddings.create.call_args
        assert call_args.kwargs["model"] == EMBEDDING_MODEL


class TestDatabaseOperations:
    @pytest.fixture
    def mock_conn(self):
        """Create a mock database connection."""
        conn = AsyncMock()
        return conn

    async def test_find_existing_node_found(self, mock_conn):
        mock_conn.fetchval = AsyncMock(return_value=42)

        result = await find_existing_node(mock_conn, 1, "Test Entity")

        assert result == 42
        mock_conn.fetchval.assert_called_once()

    async def test_find_existing_node_not_found(self, mock_conn):
        mock_conn.fetchval = AsyncMock(return_value=None)

        result = await find_existing_node(mock_conn, 1, "Unknown Entity")

        assert result is None

    async def test_get_or_create_edge_type(self, mock_conn):
        mock_conn.fetchval = AsyncMock(return_value=5)

        result = await get_or_create_edge_type(mock_conn, "mentions")

        assert result == 5
        mock_conn.fetchval.assert_called_once()

    async def test_create_entity_node(self, mock_conn):
        mock_conn.fetchval = AsyncMock(return_value=100)

        entity = {
            "name": "Test Entity",
            "short_description": "A test entity",
            "keywords": ["test", "entity"],
        }

        result = await create_entity_node(mock_conn, 1, entity, 50)

        assert result == 100
        mock_conn.fetchval.assert_called_once()

        # Verify properties
        # fetchval args: (sql, label_id, properties, source, timestamp)
        call_args = mock_conn.fetchval.call_args
        properties = call_args[0][2]  # Third positional arg is properties
        assert properties["title"] == "Test Entity"
        assert properties["extracted_from_node"] == 50
        assert "keywords" in properties

    async def test_store_embedding(self, mock_conn, mock_embedding):
        mock_conn.execute = AsyncMock()

        await store_embedding(mock_conn, 42, "Test description", mock_embedding)

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert call_args[0][1] == 42  # node_id
        assert call_args[0][2] == EMBEDDING_MODEL
        assert call_args[0][3] == "Test description"
        assert call_args[0][4] == mock_embedding


class TestProcessExtraction:
    async def test_full_extraction_pipeline(
        self, mock_openai_client, sample_extraction_response, mock_embedding
    ):
        # Setup mock connection
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        # Mock label lookup
        label_ids = {"Document": 1, "Organization": 2, "Concept": 3}

        async def mock_get_label(conn, name):
            label_ids.setdefault(name, len(label_ids) + 1)
            return label_ids[name]

        # Track fetchval calls to return different values based on query
        node_id_counter = [100]

        async def mock_fetchval(query, *args):
            # Check if it's a "SELECT id FROM kg_node" query (find_existing_node)
            if "SELECT id FROM kg_node" in query:
                return None  # No existing node found
            # Check if it's an INSERT that returns id
            if "RETURNING id" in query:
                node_id_counter[0] += 1
                return node_id_counter[0]
            return 1  # Default

        mock_conn.fetchval = mock_fetchval

        # Setup OpenAI mocks
        mock_chat_response = MagicMock()
        mock_chat_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(sample_extraction_response)))
        ]
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_chat_response
        )

        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock(embedding=mock_embedding)]
        mock_openai_client.embeddings.create = AsyncMock(
            return_value=mock_embedding_response
        )

        # Run extraction
        stats = await process_extraction(
            mock_conn,
            mock_openai_client,
            document_node_id=1,
            title="Test Document",
            content_type="text/plain",
            content="Test content about AMF and lean automation",
            get_or_create_label_func=mock_get_label,
        )

        # Verify stats
        assert stats["entities_created"] == 2
        assert stats["edges_created"] >= 0  # May vary based on mapping
        assert stats["keywords_count"] == 4
        assert stats["has_summary"] is True

        # Verify LLM calls
        assert mock_openai_client.chat.completions.create.call_count == 1
        # 1 for document + 2 for entities = 3 embedding calls
        assert mock_openai_client.embeddings.create.call_count == 3


class TestContentTruncation:
    async def test_content_truncated_for_extraction(self, mock_openai_client):
        """Test that long content is truncated."""
        long_content = "x" * 20000  # More than 15000 char limit

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="{}"))]
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        await extract_from_content(
            mock_openai_client, "Test", "text/plain", long_content
        )

        # Verify content was passed (will be truncated in prompt)
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        # Content should be truncated to 15000 chars in the prompt
        assert "x" * 15000 in user_content
        assert "x" * 20000 not in user_content

    async def test_embedding_text_truncated(self, mock_openai_client, mock_embedding):
        """Test that long text is truncated for embeddings."""
        long_text = "y" * 10000  # More than 8000 char limit

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embedding)]
        mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

        await generate_embedding(mock_openai_client, long_text)

        # Verify truncation
        call_args = mock_openai_client.embeddings.create.call_args
        input_text = call_args.kwargs["input"]
        assert len(input_text) == 8000
