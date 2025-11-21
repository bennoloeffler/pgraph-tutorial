import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

SOURCE_TODO = Path("data-source/todo")
SOURCE_DONE = Path("data-source/done")
QUEUE_TODO = Path("data-queue/todo")


def ensure_directories() -> None:
    for path in (SOURCE_TODO, SOURCE_DONE, QUEUE_TODO):
        path.mkdir(parents=True, exist_ok=True)


def guess_content_type(path: Path) -> str:
    mapping = {
        ".pdf": "application/pdf",
        ".json": "application/json",
        ".txt": "text/plain",
        ".md": "text/markdown",
    }
    return mapping.get(path.suffix.lower(), "application/octet-stream")


def build_queue_filename(source_path: Path) -> str:
    match = re.match(r"(\d+)", source_path.stem)
    prefix = match.group(1) if match else str(time.time_ns())
    return f"{prefix}_{source_path.stem}.json"


def get_file_creation_time(path: Path) -> datetime:
    """Get file creation time (birth time on macOS, ctime on Linux)."""
    stat_info = path.stat()
    # Try birth time first (macOS), fall back to mtime
    try:
        birth_time = stat_info.st_birthtime
    except AttributeError:
        birth_time = stat_info.st_mtime
    return datetime.fromtimestamp(birth_time, tz=UTC)


def payload_for_file(source_path: Path) -> dict[str, Any]:
    stat_info = source_path.stat()
    size_bytes = stat_info.st_size
    created_at = get_file_creation_time(source_path).isoformat()

    # Location after processing (file will be moved to done/)
    done_location = SOURCE_DONE / source_path.name

    payload: dict[str, Any] = {
        "type": "document",
        "content_type": guess_content_type(source_path),
        "source": {
            "type": "file",
            "path": str(done_location.resolve()),
            "hints": {},
        },
        "created_at": created_at,
        "size_bytes": size_bytes,
        "filename": source_path.name,
    }

    if source_path.suffix.lower() == ".json":
        try:
            original = json.loads(source_path.read_text())
            if isinstance(original, dict):
                payload["type"] = original.get("type", payload["type"])
                if "data" in original:
                    payload["data"] = original["data"]
                payload["created_at"] = original.get(
                    "created_at", payload["created_at"]
                )
                payload["content_type"] = original.get(
                    "content_type", payload["content_type"]
                )
                # Preserve source if provided in original
                if "source" in original:
                    payload["source"] = original["source"]
        except json.JSONDecodeError:
            pass

    return payload


def enqueue_all_sources() -> None:
    ensure_directories()
    source_files = sorted([p for p in SOURCE_TODO.iterdir() if p.is_file()])

    if not source_files:
        console.print("[yellow]No source files to enqueue[/yellow]")
        return

    console.print(Panel.fit("[bold cyan]Filling queue from data-source[/bold cyan]"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[yellow]Queueing files...", total=len(source_files))

        for source_path in source_files:
            payload = payload_for_file(source_path)
            queue_filename = build_queue_filename(source_path)
            queue_path = QUEUE_TODO / queue_filename
            queue_path.write_text(json.dumps(payload, indent=2))

            dest = SOURCE_DONE / source_path.name
            source_path.replace(dest)

            progress.advance(task)

    console.print(
        f"[green]Queued {len(source_files)} file(s) into data-queue/todo[/green]"
    )


def main() -> None:
    enqueue_all_sources()


if __name__ == "__main__":
    main()
