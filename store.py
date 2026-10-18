"""store.py -- SQLite project store (settings only, no vectors).

One RAG = one project row + its folder rows. Vectors stay in LanceDB; this
database only remembers setups so switching RAGs restores folders, weights,
models and chunk settings. Single file (app.db) -- copy it together with
index/ to move everything to another machine.

Stdlib sqlite3 only. All access happens on the GUI thread (tiny ops), never
on pipeline worker threads.
"""

from __future__ import annotations

import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DB_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    table_name TEXT NOT NULL,
    embed_model TEXT NOT NULL DEFAULT '',
    chat_model TEXT NOT NULL DEFAULT '',
    chat_provider TEXT NOT NULL DEFAULT 'ollama',
    chunk_size INTEGER NOT NULL DEFAULT 900,
    overlap INTEGER NOT NULL DEFAULT 120,
    replace_flag INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS project_folders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    extensions TEXT NOT NULL DEFAULT '',
    weight REAL NOT NULL DEFAULT 1.0,
    position INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_folders_project ON project_folders(project_id);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS saved_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_saved_project ON saved_queries(project_id);
"""

_SETTINGS_DEFAULTS = {
    "ollama_host": "",
    "default_embed_model": "",
    "default_chat_model": "",
    "default_provider": "ollama",
    "chunk_size": "900",
    "overlap": "120",
    "top_k": "8",
    "anthropic_key": "",
    "openai_key": "",
    "opencode_key": "",
    "thegrid_key": "",
    "custom_key": "",
    "custom_base_url": "",
    "rag_enabled": "1",
    "rag_min_cos": "0.30",
    "hybrid_search": "1",
    "vector_nprobes": "0",
    "vector_refine_factor": "0",
    "backup_include_keys": "0",
    "opencode_cmd": "opencode",
    "commandcode_cmd": "command-code",
}

def _connect(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL keeps readers from blocking the writer, so a stray GUI/CLI process
    # holding the file open cannot make a save fail with "database is locked".
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    except sqlite3.Error:                                # noqa: BLE001
        pass
    return conn


@contextmanager
def _session(path: str | Path):
    """Short-lived connection. `with sqlite3.connect()` alone does NOT close
    on Windows (it only commits), leaving the file locked -- this closes."""
    conn = _connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _to_int(value, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def slugify(name: str, fallback: str = "rag") -> str:
    """Safe table/folder slug: lowercase, runs of junk become one underscore."""
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return s or fallback


def init_db(path: str | Path) -> None:
    with _session(path) as conn:
        conn.executescript(_SCHEMA)
        conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('version', ?)",
                     (str(DB_VERSION),))
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an older database up to the current schema.

    `_SCHEMA` already defines the newest shape (``CREATE TABLE IF NOT EXISTS``
    never alters an existing table), so this only back-fills columns added
    after a database was first created. Each step must be idempotent.
    """
    try:
        cols = {r["name"] for r in
                conn.execute("PRAGMA table_info(projects)").fetchall()}
        # projects gained chat_provider after the first release (v1)
        if "chat_provider" not in cols:
            conn.execute("ALTER TABLE projects "
                         "ADD COLUMN chat_provider TEXT NOT NULL DEFAULT 'ollama'")
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('version', ?)",
                     (str(DB_VERSION),))
    except sqlite3.Error:                                # noqa: BLE001
        pass


def list_projects(path: str | Path) -> list[dict]:
    with _session(path) as conn:
        rows = conn.execute(
            "SELECT id, name, table_name, updated_at FROM projects "
            "ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]


def create_project(path: str | Path, *, name: str, table_name: str = "",
                   embed_model: str = "", chat_model: str = "",
                   chat_provider: str = "ollama",
                   chunk_size: int = 900, overlap: int = 120,
                   replace_flag: bool = False,
                   folders: list[dict] | tuple = ()) -> int:
    """Create a project. Raises sqlite3.IntegrityError on duplicate name."""
    now = time.time()
    with _session(path) as conn:
        cur = conn.execute(
            "INSERT INTO projects(name, table_name, embed_model, chat_model, "
            "chat_provider, chunk_size, overlap, replace_flag, "
            "created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (name.strip(), table_name.strip() or slugify(name),
             embed_model, chat_model, chat_provider or "ollama",
             _to_int(chunk_size, 900),
             _to_int(overlap, 120), int(bool(replace_flag)), now, now))
        pid = cur.lastrowid
        _write_folders(conn, pid, folders)
    return pid


def _write_folders(conn: sqlite3.Connection, pid: int, folders) -> None:
    conn.execute("DELETE FROM project_folders WHERE project_id = ?", (pid,))
    for i, f in enumerate(folders or ()):
        if not isinstance(f, dict):
            continue
        if not str(f.get("path", "")).strip():
            continue
        conn.execute(
            "INSERT INTO project_folders(project_id, path, extensions, weight, position)"
            " VALUES (?,?,?,?,?)",
            (pid, str(f["path"]), str(f.get("only", f.get("extensions", ""))),
             _to_float(f.get("weight", 1.0), 1.0), i))


def get_project(path: str | Path, pid: int) -> dict | None:
    """Full project in GUI shape (table/model/size/overlap/replace/folders)."""
    with _session(path) as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = ?",
                           (pid,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        fr = conn.execute(
            "SELECT path, extensions, weight FROM project_folders "
            "WHERE project_id = ? ORDER BY position", (pid,)).fetchall()
    data["folders"] = [{"path": r["path"], "only": r["extensions"],
                        "weight": r["weight"]} for r in fr]
    data["table"] = data.pop("table_name")
    data["model"] = data.pop("embed_model")
    data["size"] = str(data.pop("chunk_size"))
    data["overlap"] = str(data.pop("overlap"))
    data["replace"] = bool(data.pop("replace_flag"))
    data.setdefault("chat_provider", "ollama")
    return data


def save_project(path: str | Path, pid: int, data: dict) -> None:
    with _session(path) as conn:
        conn.execute(
            "UPDATE projects SET table_name=?, embed_model=?, chat_model=?, "
            "chat_provider=?, chunk_size=?, overlap=?, replace_flag=?, "
            "updated_at=? WHERE id=?",
            (str(data.get("table", "")), str(data.get("model", "")),
             str(data.get("chat_model", "")),
             str(data.get("chat_provider", "ollama") or "ollama"),
             _to_int(data.get("size", 900), 900),
             _to_int(data.get("overlap", 120), 120),
             int(bool(data.get("replace", False))), time.time(), pid))
        _write_folders(conn, pid, data.get("folders", []))


def rename_project(path: str | Path, pid: int, name: str) -> None:
    """Renames the project only -- the LanceDB table name stays (vectors live)."""
    with _session(path) as conn:
        conn.execute("UPDATE projects SET name=?, updated_at=? WHERE id=?",
                     (name.strip(), time.time(), pid))


def delete_project(path: str | Path, pid: int) -> None:
    with _session(path) as conn:
        conn.execute("DELETE FROM projects WHERE id = ?", (pid,))


# --------------------------------------------------------------------------- #
# saved queries -- per project, plain text the user can re-run
# --------------------------------------------------------------------------- #

def list_saved_queries(path: str | Path, pid: int) -> list[dict]:
    """Saved queries for a project, newest first."""
    with _session(path) as conn:
        rows = conn.execute(
            "SELECT id, query, created_at FROM saved_queries "
            "WHERE project_id = ? ORDER BY created_at DESC, id DESC",
            (pid,)).fetchall()
    return [dict(r) for r in rows]


def add_saved_query(path: str | Path, pid: int, query: str) -> int:
    """Save a query for a project. Idempotent: re-saving the same text
    returns the existing row instead of creating a duplicate."""
    q = (query or "").strip()
    if not q:
        raise ValueError("empty query")
    with _session(path) as conn:
        row = conn.execute(
            "SELECT id FROM saved_queries WHERE project_id = ? AND query = ?",
            (pid, q)).fetchone()
        if row is not None:
            return int(row["id"])
        cur = conn.execute(
            "INSERT INTO saved_queries(project_id, query, created_at) "
            "VALUES (?,?,?)", (pid, q, time.time()))
        sid = int(cur.lastrowid)
    return sid


def delete_saved_query(path: str | Path, sid: int) -> None:
    with _session(path) as conn:
        conn.execute("DELETE FROM saved_queries WHERE id = ?", (sid,))


# --------------------------------------------------------------------------- #
# settings -- flat key/value store for app-wide preferences
# --------------------------------------------------------------------------- #

def get_setting(path: str | Path, key: str, default: str = "") -> str:
    with _session(path) as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?",
                           (key,)).fetchone()
    if row is None:
        return _SETTINGS_DEFAULTS.get(key, default)
    return row["value"]


def set_setting(path: str | Path, key: str, value: str) -> None:
    with _session(path) as conn:
        conn.execute("INSERT INTO settings(key, value) VALUES (?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, str(value)))


def get_all_settings(path: str | Path) -> dict:
    """Every known key, stored value or builtin default."""
    out = dict(_SETTINGS_DEFAULTS)
    try:
        with _session(path) as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
    except Exception:                                # noqa: BLE001
        return out
    for r in rows:
        if r["key"] in out:
            out[r["key"]] = r["value"]
    return out


def set_all_settings(path: str | Path, values: dict) -> None:
    for k, v in values.items():
        if k in _SETTINGS_DEFAULTS:
            set_setting(path, k, v)
