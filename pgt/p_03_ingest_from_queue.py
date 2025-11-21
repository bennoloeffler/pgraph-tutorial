import asyncio
import base64
import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pypdf import PdfReader
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from pgt.llm_extraction import process_extraction
from pgt.p_99_utils_llm import create_openai_client

load_dotenv()

console = Console()

QUEUE_TODO = Path("data-queue/todo")
QUEUE_DONE = Path("data-queue/done")


async def configure_json_codecs(conn: asyncpg.Connection) -> None:
    """Ensure asyncpg encodes/decodes JSON automatically."""
    await conn.set_type_codec(
        "json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


def ensure_directories() -> None:
    for path in (QUEUE_TODO, QUEUE_DONE):
        path.mkdir(parents=True, exist_ok=True)


def normalized_timestamp(raw: str | None) -> datetime:
    if raw:
        parsed = datetime.fromisoformat(raw)
    else:
        parsed = datetime.now(UTC)

    if parsed.tzinfo:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)

    return parsed


def guess_content_type(payload: dict[str, Any]) -> str:
    if payload.get("content_type"):
        return payload["content_type"]

    # Try to get filename from payload
    filename = payload.get("filename") or ""
    if not filename:
        # Try source object
        source = payload.get("source")
        if isinstance(source, dict):
            # New format: source.path
            filename = Path(source.get("path", "")).name
        elif isinstance(source, str):
            # Legacy format: source as string path
            filename = Path(source).name

    ext = Path(filename).suffix.lower()
    mapping = {
        ".pdf": "application/pdf",
        ".json": "application/json",
        ".txt": "text/plain",
        ".md": "text/markdown",
    }
    return mapping.get(ext, "application/octet-stream")


def extract_text_from_pdf(content_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        reader = PdfReader(io.BytesIO(content_bytes))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    except Exception as e:
        console.print(f"[yellow]PDF extraction error: {e}[/yellow]")
        return ""


def read_source_content(payload: dict[str, Any]) -> bytes | None:
    """Read content from source object or inline data."""
    source = payload.get("source")

    # Handle new abstracted source format
    if isinstance(source, dict):
        source_type = source.get("type", "")
        source_path = source.get("path", "")

        if source_type == "file" and source_path:
            file_path = Path(source_path)
            if file_path.exists():
                return file_path.read_bytes()
        # Future: handle other source types (http, s3, etc.)

    # Legacy: source as string path
    elif isinstance(source, str):
        source_path = Path(source)
        if source_path.exists():
            return source_path.read_bytes()

    # Fall back to base64 encoded data
    if payload.get("data_base64"):
        return base64.b64decode(payload["data_base64"])

    # Fall back to inline data
    data = payload.get("data")
    if isinstance(data, str):
        return data.encode()
    if isinstance(data, dict):
        return json.dumps(data).encode()
    return None


async def get_or_create_label(conn: asyncpg.Connection, name: str) -> int:
    return await conn.fetchval(
        """
        INSERT INTO kg_label (name) VALUES ($1)
        ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
        RETURNING id
        """,
        name,
    )


async def upsert_node_for_document(
    conn: asyncpg.Connection,
    label_id: int,
    properties: dict[str, Any],
    source: dict[str, Any],
    source_created_at: datetime,
) -> int:
    queue_file = source.get("queue_file")
    existing = await conn.fetchrow(
        """
        SELECT id FROM kg_node
        WHERE label_id = $1 AND source->>'queue_file' = $2
        """,
        label_id,
        queue_file,
    )

    if existing:
        await conn.execute(
            """
            UPDATE kg_node
            SET properties = $1, source = $2, source_created_at = $3
            WHERE id = $4
            """,
            properties,
            source,
            source_created_at,
            existing["id"],
        )
        return int(existing["id"])

    return await conn.fetchval(
        """
        INSERT INTO kg_node (label_id, properties, source, source_created_at)
        VALUES ($1, $2, $3, $4)
        RETURNING id
        """,
        label_id,
        properties,
        source,
        source_created_at,
    )


async def process_queue_file(
    conn: asyncpg.Connection,
    queue_path: Path,
    openai_client: AsyncOpenAI | None = None,
) -> None:
    payload = json.loads(queue_path.read_text())

    # Extract source info
    source = payload.get("source", {})
    if isinstance(source, dict):
        # New abstracted source format
        source_info: dict[str, Any] = {
            "queue_file": queue_path.name,
            "type": source.get("type"),
            "path": source.get("path"),
            "hints": source.get("hints", {}),
            "filename": payload.get("filename"),
            "size_bytes": payload.get("size_bytes"),
        }
    elif isinstance(source, str):
        # Legacy: source as string path
        source_info = {
            "queue_file": queue_path.name,
            "type": "file",
            "path": source,
            "hints": {},
            "filename": payload.get("filename"),
            "size_bytes": payload.get("size_bytes"),
        }
    else:
        source_info = {
            "queue_file": queue_path.name,
            "filename": payload.get("filename"),
            "size_bytes": payload.get("size_bytes"),
        }

    source_created_at = normalized_timestamp(
        payload.get("created_at") or payload.get("source_created_at")
    )

    content_type = guess_content_type(payload)
    raw_content = payload.get("data")
    content_bytes = read_source_content(payload)

    title = (
        payload.get("title") or payload.get("filename") or Path(queue_path.name).stem
    )

    # Extract text content for LLM processing
    text_content: str | None = None
    if isinstance(raw_content, str):
        text_content = raw_content
    elif isinstance(raw_content, dict):
        text_content = json.dumps(raw_content)
    elif content_bytes:
        # Extract text based on content type
        if content_type == "application/pdf":
            text_content = extract_text_from_pdf(content_bytes)
        elif content_type in ("text/plain", "text/markdown", "application/json"):
            try:
                text_content = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                pass

    properties: dict[str, Any] = {
        "title": title,
        "content_type": content_type,
        "queue_file": queue_path.name,
        "size_bytes": (
            len(content_bytes) if content_bytes else payload.get("size_bytes")
        ),
        "source": source if isinstance(source, dict) else None,
        "text": text_content,
    }

    async with conn.transaction():
        label_id = await get_or_create_label(conn, payload.get("label", "Document"))
        node_id = await upsert_node_for_document(
            conn,
            label_id,
            properties,
            source_info,
            source_created_at,
        )

        # Run LLM extraction if we have content and OpenAI client
        extraction_stats = None
        if openai_client and text_content:
            try:
                extraction_stats = await process_extraction(
                    conn,
                    openai_client,
                    node_id,
                    title,
                    content_type,
                    text_content,
                    get_or_create_label,
                )
            except Exception as exc:  # pragma: no cover
                console.print(
                    f"[yellow]⚠ Extraction failed for {queue_path.name}: {exc}[/yellow]"
                )

    if extraction_stats:
        console.print(
            f"[green]✓ Processed {queue_path.name}[/green] "
            f"(node_id={node_id}, entities={extraction_stats['entities_created']}, "
            f"edges={extraction_stats['edges_created']})"
        )
    else:
        console.print(
            f"[green]✓ Processed {queue_path.name}[/green] (node_id={node_id})"
        )


async def ingest_queue() -> None:
    ensure_directories()

    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "pgraph_tutorial")

    queue_files = sorted(
        [path for path in QUEUE_TODO.iterdir() if path.is_file()],
        key=lambda p: p.name,
    )

    if not queue_files:
        console.print("[yellow]No items in data-queue/todo[/yellow]")
        return

    console.print(
        Panel.fit("[bold cyan]Ingesting queue into PostgreSQL[/bold cyan]"),
    )

    conn = await asyncpg.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        database=db_name,
    )
    await configure_json_codecs(conn)

    # Initialize OpenAI client for LLM extraction
    openai_client: AsyncOpenAI | None = None
    if os.getenv("OPENAI_API_KEY"):
        openai_client = await create_openai_client()
        console.print("[cyan]LLM extraction enabled[/cyan]")
    else:
        console.print(
            "[yellow]OPENAI_API_KEY not set - skipping LLM extraction[/yellow]"
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[yellow]Processing queue files...", total=len(queue_files)
        )

        for queue_path in queue_files:
            try:
                await process_queue_file(conn, queue_path, openai_client)
                dest = QUEUE_DONE / queue_path.name
                queue_path.replace(dest)
            except Exception as exc:  # pragma: no cover - operational logging
                console.print(f"[red]Failed to ingest {queue_path.name}: {exc}[/red]")
            finally:
                progress.advance(task)

    await conn.close()
    console.print("[green]Queue ingestion complete[/green]")


def main() -> None:
    asyncio.run(ingest_queue())


if __name__ == "__main__":
    main()
