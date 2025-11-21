# Source Abstraction

The ingestion pipeline uses an abstracted source reference system that supports multiple data access methods (files, URLs, URIs, etc.).

## Source Object Schema

```json
{
  "source": {
    "type": "file",
    "path": "/absolute/path/to/file.pdf",
    "hints": {}
  }
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Source type: `file`, `http`, `s3`, `email`, etc. |
| `path` | string | Yes | Path, URL, or URI to access the resource |
| `hints` | object | Yes | Additional access hints (can be empty `{}`) |

## Source Types

### `file` - Local File System

```json
{
  "type": "file",
  "path": "/Users/benno/projects/data/document.pdf",
  "hints": {}
}
```

### `http` / `https` - Web Resources

```json
{
  "type": "https",
  "path": "https://example.com/document.pdf",
  "hints": {
    "auth": "bearer",
    "timeout": 30
  }
}
```

### `s3` - AWS S3 Objects

```json
{
  "type": "s3",
  "path": "s3://bucket-name/path/to/object.pdf",
  "hints": {
    "region": "eu-central-1",
    "profile": "default"
  }
}
```

### `email` - Email Messages

```json
{
  "type": "email",
  "path": "mailbox://inbox/message-id-123",
  "hints": {
    "provider": "outlook",
    "folder": "Inbox"
  }
}
```

## Queue File Format

Complete queue file example:

```json
{
  "type": "document",
  "content_type": "application/pdf",
  "source": {
    "type": "file",
    "path": "/Users/benno/projects/ai/bassi/pgraph-tutorial/data-source/done/document.pdf",
    "hints": {}
  },
  "created_at": "2025-04-07T14:52:36+00:00",
  "size_bytes": 651227,
  "filename": "document.pdf"
}
```

## Implementation Notes

- The `path` field should be an absolute path for files
- For files, the path points to the location in `data-source/done/` after enqueueing
- The `hints` object is extensible for future source-specific metadata
- Readers should handle missing `hints` gracefully (treat as empty object)
