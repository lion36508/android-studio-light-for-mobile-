from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LocalMemoryStore:
    """Explicit, local-only notes with SQLite persistence and optional FTS5 search.

    This store never writes automatically. The caller must invoke remember()
    following an explicit user action. No hosted service or embedding model is used.
    """

    FORMAT_VERSION = 1
    MAX_CONTENT_CHARS = 200_000
    MAX_TAGS = 32
    MAX_RESULTS = 100

    def __init__(self, database_path: str | Path):
        self.path = Path(database_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=10.0,
            isolation_level="DEFERRED",
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            version_row = connection.execute(
                "SELECT value FROM memory_metadata WHERE key='format_version'"
            ).fetchone()
            if version_row is not None and int(version_row["value"]) > self.FORMAT_VERSION:
                raise RuntimeError(
                    "Memory database was created by a newer Q-Mini version."
                )
            connection.execute(
                """
                INSERT INTO memory_metadata(key, value)
                VALUES ('format_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (str(self.FORMAT_VERSION),),
            )
            try:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                    USING fts5(content, tags, tokenize='unicode61')
                    """
                )
                connection.execute(
                    """
                    INSERT INTO memory_fts(rowid, content, tags)
                    SELECT id, content, tags_json
                    FROM memory_records
                    WHERE id NOT IN (SELECT rowid FROM memory_fts)
                    """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                # Some minimal SQLite builds omit FTS5; LIKE search remains available.
                self._fts_enabled = False

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _normalize_tags(tags: list[str] | tuple[str, ...] | None) -> list[str]:
        if tags is None:
            return []
        if not isinstance(tags, (list, tuple)):
            raise TypeError("tags must be a list or tuple of strings.")
        cleaned: list[str] = []
        for tag in tags:
            if not isinstance(tag, str):
                raise TypeError("Every memory tag must be a string.")
            value = tag.strip()
            if value and value not in cleaned:
                cleaned.append(value[:80])
        if len(cleaned) > LocalMemoryStore.MAX_TAGS:
            raise ValueError(f"At most {LocalMemoryStore.MAX_TAGS} tags are allowed.")
        return cleaned

    def remember(
        self,
        content: str,
        tags: list[str] | tuple[str, ...] | None = None,
        source: str = "user",
    ) -> dict[str, Any]:
        if not isinstance(content, str):
            raise TypeError("Memory content must be text.")
        normalized = content.strip()
        if not normalized:
            raise ValueError("Memory content cannot be empty.")
        if len(normalized) > self.MAX_CONTENT_CHARS:
            raise ValueError(
                f"Memory content exceeds {self.MAX_CONTENT_CHARS:,} characters."
            )
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Memory source must be a non-empty string.")
        source = source.strip()[:80]
        normalized_tags = self._normalize_tags(tags)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        tags_json = json.dumps(normalized_tags, ensure_ascii=False)
        now = self._utc_now()

        with self._connection() as connection:
            existing = connection.execute(
                "SELECT id, tags_json FROM memory_records WHERE content_sha256=?",
                (digest,),
            ).fetchone()
            if existing is not None:
                existing_tags = json.loads(existing["tags_json"])
                merged_tags = self._normalize_tags(existing_tags + normalized_tags)
                connection.execute(
                    "UPDATE memory_records SET tags_json=?, updated_at=? WHERE id=?",
                    (json.dumps(merged_tags, ensure_ascii=False), now, existing["id"]),
                )
                if self._fts_enabled:
                    connection.execute(
                        "DELETE FROM memory_fts WHERE rowid=?", (existing["id"],)
                    )
                    connection.execute(
                        "INSERT INTO memory_fts(rowid, content, tags) VALUES (?, ?, ?)",
                        (existing["id"], normalized, " ".join(merged_tags)),
                    )
                return {
                    "id": int(existing["id"]),
                    "created": False,
                    "content_sha256": digest,
                    "tags": merged_tags,
                }

            cursor = connection.execute(
                """
                INSERT INTO memory_records(
                    content, tags_json, source, content_sha256, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (normalized, tags_json, source, digest, now, now),
            )
            memory_id = int(cursor.lastrowid)
            if self._fts_enabled:
                connection.execute(
                    "INSERT INTO memory_fts(rowid, content, tags) VALUES (?, ?, ?)",
                    (memory_id, normalized, " ".join(normalized_tags)),
                )

        return {
            "id": memory_id,
            "created": True,
            "content_sha256": digest,
            "tags": normalized_tags,
        }

    @staticmethod
    def _as_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "content": str(row["content"]),
            "tags": json.loads(row["tags_json"]),
            "source": str(row["source"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def search(self, query: str = "", limit: int = 8) -> list[dict[str, Any]]:
        if not isinstance(query, str):
            raise TypeError("Memory search query must be text.")
        if not isinstance(limit, int) or not 1 <= limit <= self.MAX_RESULTS:
            raise ValueError(f"limit must be between 1 and {self.MAX_RESULTS}.")
        normalized = query.strip()

        with self._connection() as connection:
            if not normalized:
                rows = connection.execute(
                    "SELECT * FROM memory_records ORDER BY updated_at DESC, id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [self._as_record(row) for row in rows]

            if self._fts_enabled:
                terms = re.findall(r"\w+", normalized, flags=re.UNICODE)
                if terms:
                    match_query = " AND ".join(
                        '"' + term.replace('"', '""') + '"' for term in terms
                    )
                    try:
                        rows = connection.execute(
                            """
                            SELECT r.*
                            FROM memory_fts
                            JOIN memory_records AS r ON r.id=memory_fts.rowid
                            WHERE memory_fts MATCH ?
                            ORDER BY bm25(memory_fts), r.updated_at DESC
                            LIMIT ?
                            """,
                            (match_query, limit),
                        ).fetchall()
                        return [self._as_record(row) for row in rows]
                    except sqlite3.OperationalError:
                        # Fall through to a literal substring search.
                        pass

            escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped}%"
            rows = connection.execute(
                """
                SELECT * FROM memory_records
                WHERE content LIKE ? ESCAPE '\\'
                   OR tags_json LIKE ? ESCAPE '\\'
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (pattern, pattern, limit),
            ).fetchall()
            return [self._as_record(row) for row in rows]

    def get(self, memory_id: int) -> dict[str, Any] | None:
        if not isinstance(memory_id, int) or memory_id < 1:
            raise ValueError("memory_id must be a positive integer.")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM memory_records WHERE id=?", (memory_id,)
            ).fetchone()
        return self._as_record(row) if row is not None else None

    def forget(self, memory_id: int) -> bool:
        if not isinstance(memory_id, int) or memory_id < 1:
            raise ValueError("memory_id must be a positive integer.")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id FROM memory_records WHERE id=?", (memory_id,)
            ).fetchone()
            if row is None:
                return False
            if self._fts_enabled:
                connection.execute(
                    "DELETE FROM memory_fts WHERE rowid=?", (memory_id,)
                )
            connection.execute("DELETE FROM memory_records WHERE id=?", (memory_id,))
        return True

    def count(self) -> int:
        with self._connection() as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM memory_records"
            ).fetchone()[0])

    def export_json(self, output_path: str | Path) -> Path:
        target = Path(output_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_records ORDER BY id"
            ).fetchall()
        payload = {
            "format": "qmini-memory-export-v1",
            "count": len(rows),
            "records": [self._as_record(row) for row in rows],
        }
        temporary = target.with_name(target.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target
