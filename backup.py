"""backup.py -- full-bundle backup / restore / transfer for embed_process.

Everything the app needs to move to another machine in one file:

    app.db        project + settings database (SQLite)
    index/        LanceDB tables (vectors) + weights.json
    staging/      collected notes & fetched links (.md), per project
    manifest.json version, date, file list, per-file SHA-256

Two flavours are supported:

- **Full bundle (.zip)** -- ``export_bundle`` / ``import_bundle``. This is the
  "easy transfer of the databases": vectors, settings and staging move together.
- **Settings only (.json)** -- handled by the GUI (store.get_all_settings +
  project rows), no vectors. Kept for lightweight config shuffling.
- **Single project (.zip)** -- ``export_project`` / ``import_project``. One
  project's table + weights + staging; import **appends** and never overwrites
  an existing table or project.

Safety:

- ``import_bundle`` takes an **automatic snapshot** of the current data into
  ``backups/auto-<timestamp>/`` before touching anything, and verifies every
  file's checksum against the manifest before replacing a single byte.
- API keys are **excluded by default** (opt in with ``include_secrets=True``);
  the bundled copy of app.db has ``anthropic_key`` / ``openai_key`` blanked.

Stdlib only. No dependency on the GUI; paths default to this package's folder
and can be overridden, which is what the tests do.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

INDEX_DIR = HERE / "index"
DB_PATH = HERE / "app.db"
STAGING_DIR = HERE / "staging"
BACKUPS_DIR = HERE / "backups"

BUNDLE_KIND = "embed_process-bundle"
BUNDLE_VERSION = 1
PROJECT_KIND = "embed_process-project"
PROJECT_VERSION = 1
MANIFEST_NAME = "manifest.json"
_SECRET_KEYS = ("anthropic_key", "openai_key", "opencode_key",
                "thegrid_key", "custom_key")


class BackupError(RuntimeError):
    """Expected failure -- shown in the UI, not a crash."""


def _log(msg: str) -> None:
    """Best-effort progress line routed to the pipeline/GUI log sink."""
    try:
        import pipeline
        pipeline.log(msg)
    except Exception:                                    # noqa: BLE001
        pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _snapshot_db(src: Path, dst: Path) -> None:
    """Copy a live SQLite database consistently (handles WAL).

    A plain file copy can miss committed rows still sitting in the ``-wal``
    sidecar; the sqlite backup API always produces a coherent single file.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    src_conn = sqlite3.connect(str(src))
    dst_conn = sqlite3.connect(str(dst))
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()


def _strip_keys(db_path: Path) -> None:
    """Blank API keys in a *copied* database, before it goes into a bundle."""
    conn = sqlite3.connect(str(db_path))
    try:
        for key in _SECRET_KEYS:
            conn.execute("UPDATE settings SET value = '' WHERE key = ?", (key,))
        conn.commit()
    finally:
        conn.close()


def _read_setting(db_path: Path, key: str, default: str = "") -> str:
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute("SELECT value FROM settings WHERE key = ?",
                               (key,)).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return default
    return row[0] if row else default


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #

def _iter_files(root: Path):
    if not root.is_dir():
        return
    for p in sorted(root.rglob("*")):
        if p.is_file():
            yield p


def export_bundle(dest: str | Path, *,
                  index_dir: Path = INDEX_DIR,
                  db_path: Path = DB_PATH,
                  staging_dir: Path = STAGING_DIR,
                  include_staging: bool = True,
                  include_secrets: bool = False) -> dict:
    """Write a full backup bundle to `dest` (a .zip). Returns the manifest.

    Raises BackupError on any I/O problem. Never mutates the live data.
    """
    import tempfile

    dest = Path(dest)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise BackupError(f"cannot create {dest.parent}: {e}")

    entries: list[tuple[str, Path]] = []     # (arcname, source file)

    with tempfile.TemporaryDirectory(prefix="ep-export-") as td:
        tmp = Path(td)

        if Path(db_path).exists():
            db_copy = tmp / "app.db"
            _snapshot_db(Path(db_path), db_copy)
            if not include_secrets:
                _strip_keys(db_copy)
            entries.append(("app.db", db_copy))

        n_tables = 0
        for p in _iter_files(Path(index_dir)) or ():
            rel = p.relative_to(index_dir).as_posix()
            entries.append((f"index/{rel}", p))
        # count tables = *.lance directories directly under index/
        try:
            n_tables = sum(1 for c in Path(index_dir).iterdir()
                           if c.is_dir() and c.name.endswith(".lance"))
        except OSError:
            n_tables = 0

        n_staging = 0
        if include_staging:
            for p in _iter_files(Path(staging_dir)) or ():
                rel = p.relative_to(staging_dir).as_posix()
                entries.append((f"staging/{rel}", p))
                n_staging += 1

        # Read every file once: hash it for the manifest and reuse the bytes.
        files_meta: dict[str, str] = {}
        blobs: list[tuple[str, bytes]] = []
        for arc, src in entries:
            data = src.read_bytes()
            files_meta[arc] = _sha256_bytes(data)
            blobs.append((arc, data))

        manifest = {
            "app": "embed_process",
            "kind": BUNDLE_KIND,
            "version": BUNDLE_VERSION,
            "created_at": time.time(),
            "created_at_readable": time.strftime("%Y-%m-%d %H:%M:%S"),
            "include_secrets": bool(include_secrets),
            "include_staging": bool(include_staging),
            "counts": {
                "tables": n_tables,
                "staging_files": n_staging,
                "files": len(entries),
            },
            "files": files_meta,
        }
        if Path(db_path).exists():
            manifest["counts"]["projects"] = _count_projects(tmp / "app.db")

        try:
            with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr(MANIFEST_NAME,
                           json.dumps(manifest, indent=2, ensure_ascii=False))
                for arc, data in blobs:
                    z.writestr(arc, data)
        except OSError as e:
            raise BackupError(f"cannot write bundle: {e}")

    _log(f"backup: {len(entries)} file(s), {n_tables} table(s) -> {dest}")
    manifest["path"] = str(dest)
    return manifest


def _count_projects(db_path: Path) -> int:
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            return int(conn.execute(
                "SELECT COUNT(*) FROM projects").fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #

def read_manifest(zip_path: str | Path) -> dict:
    """Return the manifest of a bundle. Raises BackupError when unusable."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            if MANIFEST_NAME not in z.namelist():
                raise BackupError("not a backup bundle (no manifest.json)")
            return json.loads(z.read(MANIFEST_NAME).decode("utf-8"))
    except zipfile.BadZipFile:
        raise BackupError("not a valid .zip file")
    except BackupError:
        raise
    except Exception as e:                               # noqa: BLE001
        raise BackupError(f"cannot read bundle: {e}")


def verify_bundle(zip_path: str | Path) -> dict:
    """Check every file in the bundle against its manifest checksum.

    Returns {"ok": bool, "checked": n, "bad": [names], "manifest": {...}}.
    """
    return _verify_kind(zip_path, BUNDLE_KIND, "an embed_process bundle")


def verify_project(zip_path: str | Path) -> dict:
    """Same as ``verify_bundle`` but for a single-project package."""
    return _verify_kind(zip_path, PROJECT_KIND, "an embed_process project package")


def _verify_kind(zip_path: str | Path, kind: str, label: str) -> dict:
    manifest = read_manifest(zip_path)
    if manifest.get("kind") != kind:
        raise BackupError(f"not {label}")
    want = manifest.get("files") or {}
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        bad: list[str] = []
        for arc, digest in want.items():
            if arc not in names:
                bad.append(f"{arc} (missing)")
                continue
            if _sha256_bytes(z.read(arc)) != digest:
                bad.append(f"{arc} (checksum)")
    return {"ok": not bad, "checked": len(want), "bad": bad,
            "manifest": manifest}


# --------------------------------------------------------------------------- #
# snapshot of the current data (safety net before restore)
# --------------------------------------------------------------------------- #

def snapshot_current(*,
                     index_dir: Path = INDEX_DIR,
                     db_path: Path = DB_PATH,
                     staging_dir: Path = STAGING_DIR,
                     backups_dir: Path = BACKUPS_DIR,
                     keep: int = 5) -> Path:
    """Copy the live data into backups/auto-<timestamp>/ and prune old ones."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    dest = Path(backups_dir) / f"auto-{ts}"
    n = 2
    while dest.exists():
        dest = Path(backups_dir) / f"auto-{ts}-{n}"
        n += 1
    dest.mkdir(parents=True, exist_ok=True)

    if Path(db_path).exists():
        _snapshot_db(Path(db_path), dest / "app.db")
    if Path(index_dir).is_dir():
        shutil.copytree(Path(index_dir), dest / "index")
    if Path(staging_dir).is_dir():
        shutil.copytree(Path(staging_dir), dest / "staging")

    _prune_snapshots(Path(backups_dir), keep=keep)
    _log(f"snapshot: {dest}")
    return dest


def _prune_snapshots(backups_dir: Path, keep: int) -> None:
    if keep <= 0 or not backups_dir.is_dir():
        return
    autos = sorted((d for d in backups_dir.iterdir()
                    if d.is_dir() and d.name.startswith("auto-")),
                   key=lambda d: d.name, reverse=True)
    for old in autos[keep:]:
        shutil.rmtree(old, ignore_errors=True)


# --------------------------------------------------------------------------- #
# import
# --------------------------------------------------------------------------- #

def _replace_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def import_bundle(zip_path: str | Path, *,
                  index_dir: Path = INDEX_DIR,
                  db_path: Path = DB_PATH,
                  staging_dir: Path = STAGING_DIR,
                  backups_dir: Path = BACKUPS_DIR,
                  snapshot: bool = True) -> dict:
    """Replace the live data with the contents of a bundle.

    A verified snapshot of the current data is written to
    ``backups/auto-<timestamp>/`` first (unless ``snapshot=False``).
    Returns a summary dict. Raises BackupError on verification/interruption.
    """
    report = verify_bundle(zip_path)
    if not report["ok"]:
        raise BackupError(
            "bundle failed verification:\n  " + "\n  ".join(report["bad"][:10]))
    manifest = report["manifest"]

    snap_path = None
    if snapshot:
        snap_path = snapshot_current(index_dir=index_dir, db_path=db_path,
                                     staging_dir=staging_dir,
                                     backups_dir=backups_dir)

    import tempfile
    with tempfile.TemporaryDirectory(prefix="ep-import-") as td:
        tmp = Path(td)
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(tmp)
        except (OSError, zipfile.BadZipFile) as e:
            raise BackupError(f"cannot extract bundle: {e}")

        try:
            if (tmp / "app.db").is_file():
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tmp / "app.db", db_path)
            if (tmp / "index").is_dir():
                _replace_dir(tmp / "index", Path(index_dir))
            if (tmp / "staging").is_dir():
                _replace_dir(tmp / "staging", Path(staging_dir))
        except OSError as e:
            raise BackupError(f"restore failed: {e}")

    summary = {
        "kind": "restore",
        "projects": _count_projects(Path(db_path)),
        "tables": (manifest.get("counts") or {}).get("tables", 0),
        "staging_files": (manifest.get("counts") or {}).get("staging_files", 0),
        "snapshot": str(snap_path) if snap_path else None,
        "from": str(zip_path),
    }
    _log(f"restored bundle: {summary['projects']} project(s), "
         f"{summary['tables']} table(s)")
    return summary


# --------------------------------------------------------------------------- #
# single-project package (move one RAG without touching the others)
# --------------------------------------------------------------------------- #

def _as_int(value, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _read_project(db_path: Path, pid: int) -> dict | None:
    """Project row + folders straight from SQLite (no import of store here)."""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as e:
        raise BackupError(f"cannot open {db_path}: {e}")
    try:
        row = conn.execute("SELECT * FROM projects WHERE id = ?",
                           (pid,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        fr = conn.execute(
            "SELECT path, extensions, weight, position FROM project_folders "
            "WHERE project_id = ? ORDER BY position", (pid,)).fetchall()
        data["folders"] = [dict(f) for f in fr]
        return data
    except sqlite3.Error as e:
        raise BackupError(f"cannot read project {pid}: {e}")
    finally:
        conn.close()


def export_project(dest: str | Path, *, pid: int,
                   index_dir: Path = INDEX_DIR,
                   db_path: Path = DB_PATH,
                   staging_dir: Path = STAGING_DIR,
                   include_staging: bool = True) -> dict:
    """Write one project (its table + weights + staging) to a .zip.

    Unlike ``export_bundle`` this does **not** touch app.db: the project row is
    embedded in the manifest, so importing appends a project instead of
    replacing everything. Returns the manifest.
    """
    dest = Path(dest)
    proj = _read_project(Path(db_path), pid)
    if proj is None:
        raise BackupError(f"project {pid} not found")
    table = str(proj.get("table_name") or "").strip()
    if not table:
        raise BackupError(f"project {pid} has no table name")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise BackupError(f"cannot create {dest.parent}: {e}")

    index_dir = Path(index_dir)
    entries: list[tuple[str, Path]] = []

    table_src = index_dir / f"{table}.lance"
    has_table = table_src.is_dir()
    if has_table:
        for p in _iter_files(table_src) or ():
            rel = p.relative_to(index_dir).as_posix()
            entries.append((f"index/{rel}", p))

    # weights for just this table (folder importance follows the vectors)
    weights: dict = {}
    weights_src = index_dir / "weights.json"
    if weights_src.is_file():
        try:
            allw = json.loads(weights_src.read_text(encoding="utf-8"))
            if isinstance(allw, dict) and isinstance(allw.get(table), dict):
                weights = {table: allw[table]}
        except Exception:                                # noqa: BLE001
            weights = {}

    n_staging = 0
    src_stage = Path(staging_dir) / f"p{pid}"
    if include_staging and src_stage.is_dir():
        for p in _iter_files(src_stage) or ():
            rel = p.relative_to(src_stage).as_posix()
            entries.append((f"staging/{rel}", p))
            n_staging += 1

    files_meta: dict[str, str] = {}
    blobs: list[tuple[str, bytes]] = []
    for arc, src in entries:
        data = src.read_bytes()
        files_meta[arc] = _sha256_bytes(data)
        blobs.append((arc, data))

    manifest = {
        "app": "embed_process",
        "kind": PROJECT_KIND,
        "version": PROJECT_VERSION,
        "created_at": time.time(),
        "created_at_readable": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_pid": pid,
        "project": {
            "name": proj.get("name"),
            "table": table,
            "embed_model": proj.get("embed_model", ""),
            "chat_model": proj.get("chat_model", ""),
            "chat_provider": proj.get("chat_provider", "ollama"),
            "chunk_size": proj.get("chunk_size", 900),
            "overlap": proj.get("overlap", 120),
            "replace_flag": proj.get("replace_flag", 0),
            "folders": proj.get("folders", []),
        },
        "has_table": has_table,
        "include_staging": bool(include_staging),
        "weights": weights,
        "counts": {"files": len(entries), "staging_files": n_staging,
                   "tables": 1 if has_table else 0},
        "files": files_meta,
    }
    try:
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(MANIFEST_NAME,
                       json.dumps(manifest, indent=2, ensure_ascii=False))
            for arc, data in blobs:
                z.writestr(arc, data)
    except OSError as e:
        raise BackupError(f"cannot write project package: {e}")

    _log(f"export project '{proj.get('name')}': {len(entries)} file(s), "
         f"{n_staging} staging file(s) -> {dest}")
    manifest["path"] = str(dest)
    return manifest


def _unique_project_name(db_path: Path, name: str) -> str:
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            existing = {r[0] for r in
                        conn.execute("SELECT name FROM projects").fetchall()}
        finally:
            conn.close()
    except sqlite3.Error:
        existing = set()
    if name not in existing:
        return name
    cand = f"{name} (imported)"
    n = 2
    while cand in existing:
        cand = f"{name} (imported {n})"
        n += 1
    return cand


def _unique_table_name(index_dir: Path, table: str) -> str:
    """Never overwrite a table that is already on disk."""
    table = (table or "corpus").strip() or "corpus"
    if not (Path(index_dir) / f"{table}.lance").exists():
        return table
    base = table
    n = 1
    while (Path(index_dir) / f"{table}.lance").exists():
        table = f"{base}_imported" if n == 1 else f"{base}_imported{n}"
        n += 1
    return table


def _merge_weights(path: Path, table: str, mapping: dict) -> None:
    cur: dict = {}
    if Path(path).is_file():
        try:
            loaded = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cur = loaded
        except Exception:                                # noqa: BLE001
            cur = {}
    if isinstance(mapping, dict) and mapping:
        cur[table] = mapping
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(cur, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    except OSError as e:
        raise BackupError(f"cannot update weights.json: {e}")


def import_project(zip_path: str | Path, *,
                   index_dir: Path = INDEX_DIR,
                   db_path: Path = DB_PATH,
                   staging_dir: Path = STAGING_DIR,
                   backups_dir: Path = BACKUPS_DIR,
                   snapshot: bool = True) -> dict:
    """Add a single-project package to the live data (append, not replace).

    A verified snapshot of the current data is written to
    ``backups/auto-<timestamp>/`` first. Existing tables/projects are never
    overwritten: a clashing table name gets a ``_imported`` suffix and a
    clashing project name gets `` (imported)``. Returns a summary dict.
    """
    report = verify_project(zip_path)
    if not report["ok"]:
        raise BackupError(
            "package failed verification:\n  " + "\n  ".join(report["bad"][:10]))
    manifest = report["manifest"]
    proj = manifest.get("project") or {}
    name = str(proj.get("name") or "imported").strip() or "imported"
    src_table = str(proj.get("table") or "").strip()

    index_dir = Path(index_dir)
    db_path = Path(db_path)
    staging_dir = Path(staging_dir)

    snap_path = None
    if snapshot:
        snap_path = snapshot_current(index_dir=index_dir, db_path=db_path,
                                     staging_dir=staging_dir,
                                     backups_dir=backups_dir)

    table = _unique_table_name(index_dir, src_table)

    import tempfile
    with tempfile.TemporaryDirectory(prefix="ep-import-project-") as td:
        tmp = Path(td)
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(tmp)
        except (OSError, zipfile.BadZipFile) as e:
            raise BackupError(f"cannot extract package: {e}")

        copied_table = False
        src_table_dir = tmp / "index" / f"{src_table}.lance"
        if src_table_dir.is_dir():
            try:
                _replace_dir(src_table_dir, index_dir / f"{table}.lance")
                copied_table = True
            except OSError as e:
                raise BackupError(f"cannot install table '{table}': {e}")

        inner = (manifest.get("weights") or {}).get(src_table)
        if isinstance(inner, dict) and inner:
            _merge_weights(index_dir / "weights.json", table, inner)

        # create the project row (need the new pid before staging can be copied)
        folders = [{"path": f.get("path", ""),
                    "only": f.get("extensions", ""),
                    "weight": f.get("weight", 1.0)}
                   for f in (proj.get("folders") or []) if isinstance(f, dict)]
        new_name = _unique_project_name(db_path, name)
        try:
            import store
            store.init_db(db_path)
            pid = store.create_project(
                db_path, name=new_name, table_name=table,
                embed_model=str(proj.get("embed_model", "")),
                chat_model=str(proj.get("chat_model", "")),
                chat_provider=str(proj.get("chat_provider", "ollama") or "ollama"),
                chunk_size=_as_int(proj.get("chunk_size"), 900),
                overlap=_as_int(proj.get("overlap"), 120),
                replace_flag=bool(proj.get("replace_flag", 0)),
                folders=folders)
        except BackupError:
            raise
        except Exception as e:                           # noqa: BLE001
            raise BackupError(f"cannot create project '{new_name}': {e}")

        n_staging = 0
        src_stage = tmp / "staging"
        if src_stage.is_dir():
            dest_stage = staging_dir / f"p{pid}"
            try:
                if dest_stage.exists():
                    shutil.rmtree(dest_stage)
                shutil.copytree(src_stage, dest_stage)
                n_staging = sum(1 for _ in _iter_files(dest_stage) or ())
            except OSError as e:
                raise BackupError(f"cannot restore staging: {e}")

    summary = {
        "kind": "project-import",
        "pid": pid,
        "name": new_name,
        "table": table,
        "renamed": new_name != name,
        "table_renamed": bool(src_table and table != src_table),
        "staging_files": n_staging,
        "has_table": copied_table,
        "snapshot": str(snap_path) if snap_path else None,
        "from": str(zip_path),
    }
    _log(f"imported project '{new_name}' (table '{table}', "
         f"{n_staging} staging file(s))")
    return summary
