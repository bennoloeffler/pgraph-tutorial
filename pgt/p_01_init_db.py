import asyncio
import os

import asyncpg
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()

console = Console()


async def init_db():
    console.print(
        Panel.fit(
            "[bold cyan]Initializing pgraph-tutorial database[/bold cyan]",
            border_style="blue",
        )
    )

    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "pgraph_tutorial")

    # Connect to default postgres database to create our database if needed
    sys_conn = await asyncpg.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        database="postgres",
    )

    # Check if database exists, create if not
    exists = await sys_conn.fetchval(
        "SELECT 1 FROM pg_database WHERE datname = $1", db_name
    )
    if not exists:
        console.print(f"[yellow]Creating database '{db_name}'...[/yellow]")
        await sys_conn.execute(f'CREATE DATABASE "{db_name}"')
        console.print(f"[green]✓ Database '{db_name}' created[/green]")

    await sys_conn.close()

    # Connect to our database
    conn = await asyncpg.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        database=db_name,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Enable extensions
        task = progress.add_task("[yellow]Enabling extensions...", total=None)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        progress.update(task, description="[green]✓ Extensions enabled")

        # Create knowledge graph tables
        task = progress.add_task(
            "[yellow]Creating knowledge graph tables...", total=None
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kg_label (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            )
        """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kg_edge_type (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            )
        """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kg_node (
                id BIGSERIAL PRIMARY KEY,
                label_id INT NOT NULL REFERENCES kg_label(id),
                properties JSONB NOT NULL DEFAULT '{}'::jsonb,
                source JSONB NOT NULL DEFAULT '{}'::jsonb,
                source_created_at TIMESTAMP NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kg_edge (
                id BIGSERIAL PRIMARY KEY,
                src_id BIGINT NOT NULL REFERENCES kg_node(id) ON DELETE CASCADE,
                dst_id BIGINT NOT NULL REFERENCES kg_node(id) ON DELETE CASCADE,
                type_id INT NOT NULL REFERENCES kg_edge_type(id),
                properties JSONB NOT NULL DEFAULT '{}'::jsonb,
                UNIQUE(src_id, dst_id, type_id)
            )
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_kg_node_label
            ON kg_node(label_id)
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_kg_node_props_gin
            ON kg_node USING gin (properties)
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_kg_edge_src
            ON kg_edge(src_id)
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_kg_edge_dst
            ON kg_edge(dst_id)
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_kg_edge_type
            ON kg_edge(type_id)
        """
        )
        progress.update(task, description="[green]✓ Knowledge graph tables created")

        # Create node embeddings table for graph integration with pgvector
        task = progress.add_task(
            "[yellow]Creating graph embeddings table...", total=None
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kg_node_embedding (
                node_id BIGINT PRIMARY KEY REFERENCES kg_node(id) ON DELETE CASCADE,
                model_name TEXT NOT NULL DEFAULT 'text-embedding-3-small',
                short_description TEXT NOT NULL,
                embedding vector(1536) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_kg_node_embedding_hnsw
            ON kg_node_embedding
            USING hnsw (embedding vector_cosine_ops)
        """
        )
        progress.update(task, description="[green]✓ Graph embeddings table created")

    await conn.close()

    console.print()
    console.print(
        Panel.fit(
            "[bold green]Database initialized successfully![/bold green]",
            border_style="green",
        )
    )


def main():
    asyncio.run(init_db())


if __name__ == "__main__":
    main()
