from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


SCHEMA_VERSION = 2

_RECORD_IDENTITY_KEYS = frozenset({"task", "run", "step", "oligo"})


def _json_dumps_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _compact_record_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    metrics = record.get("metrics") or {}
    if not isinstance(metrics, dict):
        return {}
    compact = dict(metrics)
    for key in _RECORD_IDENTITY_KEYS:
        if key not in compact:
            continue
        if str(compact.get(key)) == str(record.get(key)):
            compact.pop(key, None)
    return compact


def open_db(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def set_meta(conn: sqlite3.Connection, *, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO meta(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value;
        """,
        (str(key), str(value)),
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, *, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (str(key),)).fetchone()
    if row is None:
        return None
    return str(row["value"])


def get_out_roots(conn: sqlite3.Connection) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    raw_json = get_meta(conn, key="out_roots_json")
    if raw_json:
        try:
            parsed = json.loads(raw_json)
        except Exception:
            parsed = []
        if isinstance(parsed, list):
            for item in parsed:
                root = str(item or "").strip()
                if not root or root in seen:
                    continue
                seen.add(root)
                ordered.append(root)

    explicit = str(get_meta(conn, key="out_root") or "").strip()
    if explicit and explicit not in seen:
        seen.add(explicit)
        ordered.append(explicit)
    return ordered


def set_out_roots(conn: sqlite3.Connection, *, roots: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in roots:
        root = str(item or "").strip()
        if not root or root in seen:
            continue
        seen.add(root)
        ordered.append(root)
    set_meta(conn, key="out_roots_json", value=json.dumps(ordered, ensure_ascii=False))
    set_meta(conn, key="out_root", value=ordered[0] if len(ordered) == 1 else "")
    return ordered


def merge_out_roots(conn: sqlite3.Connection, *, roots: Iterable[str]) -> list[str]:
    merged = get_out_roots(conn)
    seen = set(merged)
    for item in roots:
        root = str(item or "").strip()
        if not root or root in seen:
            continue
        seen.add(root)
        merged.append(root)
    return set_out_roots(conn, roots=merged)


def _infer_out_root_from_run_dir(run_dir: Path) -> Path:
    candidate = run_dir.parent.parent
    if candidate.name in {"03_runs", "runs"}:
        return candidate.parent
    return candidate


def infer_db_out_root(conn: sqlite3.Connection) -> Optional[str]:
    explicit = get_meta(conn, key="out_root")
    if explicit:
        return explicit

    known_roots = get_out_roots(conn)
    if len(known_roots) == 1:
        return known_roots[0]
    if len(known_roots) > 1:
        common = Path(os.path.commonpath(known_roots))
        if str(common) == os.path.sep:
            return None
        return str(common)

    run_dirs = [
        Path(str(row["run_dir"])).expanduser().resolve()
        for row in conn.execute("SELECT run_dir FROM runs ORDER BY run_dir")
    ]
    if not run_dirs:
        run_dirs = [
            Path(str(row["run_dir"])).expanduser().resolve()
            for row in conn.execute("SELECT DISTINCT run_dir FROM records ORDER BY run_dir")
        ]
    if not run_dirs:
        return None

    inferred_roots = {_infer_out_root_from_run_dir(run_dir) for run_dir in run_dirs}
    if len(inferred_roots) == 1:
        return str(next(iter(inferred_roots)))

    common = Path(os.path.commonpath([str(path) for path in run_dirs]))
    if common.name in {"03_runs", "runs"}:
        return str(common.parent)
    if str(common) == os.path.sep:
        return None
    return str(common)


def _init_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA foreign_keys=ON;")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_dir TEXT PRIMARY KEY,
            task TEXT NOT NULL,
            run TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            scores_mtime_ns INTEGER NOT NULL,
            scores_size INTEGER NOT NULL,
            updated_at REAL NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            run_dir TEXT NOT NULL,
            task TEXT NOT NULL,
            run TEXT NOT NULL,
            step INTEGER NOT NULL,
            oligo TEXT NOT NULL,
            structure_path TEXT,
            metrics_json TEXT NOT NULL,
            scores_mtime_ns INTEGER NOT NULL,
            scores_size INTEGER NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (run_dir, step, oligo)
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_records_task_run ON records(task, run);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_records_task ON records(task);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_records_oligo ON records(oligo);")

    run_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(runs)")
    }
    if "metadata_json" not in run_columns:
        cur.execute("ALTER TABLE runs ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}';")
    cur.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    cur.execute(
        """
        UPDATE meta
        SET value = ?
        WHERE key = 'schema_version' AND value != ?;
        """,
        (str(SCHEMA_VERSION), str(SCHEMA_VERSION)),
    )
    conn.commit()


def get_run_signature(conn: sqlite3.Connection, run_dir: str) -> Optional[tuple[int, int]]:
    row = conn.execute(
        "SELECT scores_mtime_ns, scores_size FROM runs WHERE run_dir = ?",
        (str(run_dir),),
    ).fetchone()
    if row is None:
        return None
    return int(row["scores_mtime_ns"]), int(row["scores_size"])


def upsert_run_signature(
    conn: sqlite3.Connection,
    *,
    run_dir: str,
    task: str,
    run: str,
    metadata: Dict[str, Any] | None = None,
    scores_mtime_ns: int,
    scores_size: int,
) -> None:
    conn.execute(
        """
        INSERT INTO runs(run_dir, task, run, metadata_json, scores_mtime_ns, scores_size, updated_at)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_dir) DO UPDATE SET
          task=excluded.task,
          run=excluded.run,
          metadata_json=excluded.metadata_json,
          scores_mtime_ns=excluded.scores_mtime_ns,
          scores_size=excluded.scores_size,
          updated_at=excluded.updated_at;
        """,
        (
            str(run_dir),
            str(task),
            str(run),
            _json_dumps_compact(metadata or {}),
            int(scores_mtime_ns),
            int(scores_size),
            time.time(),
        ),
    )
    conn.commit()


def get_run_metadata(conn: sqlite3.Connection, *, run_dir: str | Path) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT metadata_json FROM runs WHERE run_dir = ?",
        (str(Path(run_dir).expanduser().resolve()),),
    ).fetchone()
    if row is None:
        return {}
    try:
        raw = json.loads(str(row["metadata_json"] or "{}"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def upsert_records(conn: sqlite3.Connection, records: Iterable[Dict[str, Any]]) -> int:
    rows = []
    now = time.time()
    for r in records:
        rows.append(
            (
                str(r["run_dir"]),
                str(r["task"]),
                str(r["run"]),
                int(r["step"]),
                str(r["oligo"]),
                str(r["structure_path"]) if r.get("structure_path") else None,
                _json_dumps_compact(_compact_record_metrics(r)),
                int(r["scores_mtime_ns"]),
                int(r["scores_size"]),
                now,
            )
        )

    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO records(
          run_dir, task, run, step, oligo, structure_path,
          metrics_json, scores_mtime_ns, scores_size, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_dir, step, oligo) DO UPDATE SET
          task=excluded.task,
          run=excluded.run,
          structure_path=excluded.structure_path,
          metrics_json=excluded.metrics_json,
          scores_mtime_ns=excluded.scores_mtime_ns,
          scores_size=excluded.scores_size,
          updated_at=excluded.updated_at;
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def iter_records(
    conn: sqlite3.Connection,
    *,
    oligo: str | None = None,
    task: str | None = None,
) -> Iterator[sqlite3.Row]:
    where = []
    params = []
    if oligo is not None:
        where.append("oligo = ?")
        params.append(str(oligo))
    if task is not None:
        where.append("task = ?")
        params.append(str(task))
    sql = "SELECT * FROM records"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY task, run, step, oligo"
    cur = conn.execute(sql, params)
    yield from cur
