"""pipeline.py -- local RAG embedding pipeline on Ollama + LanceDB.

Subcommands
-----------
    ingest   walk a folder, extract, chunk, store text + metadata (no vectors)
             (--append adds to an existing table, deduped by source)
    embed    batch-embed pending rows into vectors
    query    hybrid (FTS + vector, RRF) search over an index
    stats    chunk counts by format / model / dims + on-disk size
    eval     run a CSV of queries against several tables, print a comparison
    reindex  build a new table for a different model, copying metadata
    export-md  convert a folder into .md sources (same extractors as ingest)

Library use
-----------
    This module is also importable -- the GUI (tk_app.py) drives the same
    `cmd_*` functions instead of spawning a subprocess. To make that safe:

    - `fail()` raises `PipelineError` instead of calling `sys.exit(1)`.
    - `log()` prints *and* forwards to any sink registered via `set_logger()`
      / `set_progress()`, so progress can reach a UI widget.

Hardware assumptions (GTX 1050 Ti 4 GB / 16 GB RAM):
    - num_ctx is pinned to 8192 (never the model's advertised max)
    - batch 8, lowered automatically on out-of-memory
    - OLLAMA_KV_CACHE_TYPE=q8_0 halves the KV cache
    - truncate=False so over-long input fails loudly instead of silently
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NoReturn

# --------------------------------------------------------------------------- #
# hardware / model configuration
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = "qwen3-embedding:0.6b"
NUM_CTX = 8192
BATCH = 8
MIN_BATCH = 2
TRUNCATE = False
CHARS_PER_TOKEN = 4
EMBED_TIMEOUT = 180
EMBED_RETRIES = 3
CHECKPOINT_EVERY = 64

# Ollama's /api/embed applies no prompt template; Qwen3 reports +1-5% retrieval
# when the query side carries an instruction. Always English -- that is what the
# model was trained with.
QUERY_INSTRUCT = (
    "Given a web search query, retrieve relevant passages "
    "that answer the query.\n"
)

INDEX_DIR = Path(__file__).parent / "index"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
CHUNK_MIN = 50
CODE_MAX = 600

DEFAULT_ENV = {"OLLAMA_KV_CACHE_TYPE": "q8_0"}


def _ensure_env() -> None:
    """Set KV-cache env vars before Ollama is touched (best effort)."""
    for k, v in DEFAULT_ENV.items():
        os.environ.setdefault(k, v)


# --------------------------------------------------------------------------- #
# imports that depend on the packages in requirements.txt
# --------------------------------------------------------------------------- #

def _import_ollama():
    try:
        import ollama  # type: ignore
    except ImportError:
        raise PipelineError("Missing dependency: pip install ollama")
    return ollama


def _import_lancedb():
    try:
        import lancedb  # type: ignore
    except ImportError:
        raise PipelineError("Missing dependency: pip install lancedb pyarrow")
    return lancedb


def _db():
    lancedb = _import_lancedb()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(INDEX_DIR))


def _client():
    _ensure_env()
    ollama = _import_ollama()
    return ollama.Client(
        host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        timeout=EMBED_TIMEOUT,
    )


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

class PipelineError(RuntimeError):
    """Expected failure.

    Raised instead of `sys.exit(1)` so the same code can run from the CLI
    (which converts it to exit code 1) or from the GUI (which shows it in a
    message box). Using RuntimeError means an unexpected bug in the GUI still
    surfaces as a real traceback.
    """


# Optional sinks. Set by the GUI to route progress into widgets; the CLI
# leaves them None. Neither is thread-safe on its own -- the GUI is
# responsible for marshalling to its UI thread.
_logger: Callable[[str], None] | None = None
_progress: Callable[[int, int], None] | None = None


def set_logger(fn: Callable[[str], None] | None) -> None:
    """Register a callback that receives every `log()` line."""
    global _logger
    _logger = fn


def set_progress(fn: Callable[[int, int], None] | None) -> None:
    """Register a callback that receives (done, total) counts."""
    global _progress
    _progress = fn


def emit_progress(done: int, total: int) -> None:
    if _progress is None:
        return
    try:
        _progress(done, total)
    except Exception:                                # noqa: BLE001
        pass                                         # never let UI errors kill a run


def log(msg: str) -> None:
    _emit(msg)
    if _logger is not None:
        try:
            _logger(msg)
        except Exception:                            # noqa: BLE001
            pass


def _emit(msg: str) -> None:
    """Print, tolerating a non-UTF-8 stdout (redirect to file / cp1252 pipe).

    On Windows, `python pipeline.py ... > out.txt` gives stdout a cp1252 codec
    and any box-drawing/check-mark character raises UnicodeEncodeError. That
    would abort an otherwise fine run, so degrade to '?' instead of dying.
    """
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        try:
            sys.stdout.reconfigure(errors="replace")   # TextIOWrapper only
            print(msg, flush=True)
        except Exception:                             # noqa: BLE001
            print(msg.encode("utf-8", "replace").decode("utf-8", "replace"),
                  flush=True)


def _table_names(db) -> list[str]:
    """List table names, tolerating the LanceDB rename to list_tables()."""
    fn = getattr(db, "list_tables", None) or getattr(db, "table_names")
    out = fn()
    # list_tables() returns a ListTablesResponse with a .tables list
    for attr in ("tables", "table_names"):
        inner = getattr(out, attr, None)
        if inner is not None:
            out = inner
            break
    return [str(t) for t in out]


def fail(msg: str) -> NoReturn:
    raise PipelineError(msg)


def est_tokens(text: str) -> int:
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN) if text else 0


def table_names() -> list[str]:
    try:
        return sorted(_table_names(_db()))
    except Exception as e:
        fail(f"cannot open index at {INDEX_DIR}: {e}")


def _open_table(name: str | None):
    db = _db()
    names = _table_names(db)
    if name:
        if name not in names:
            fail(f"table '{name}' not found. Available: {', '.join(names) or '(none)'}")
        return db.open_table(name)
    if not names:
        fail("no tables yet -- run: python pipeline.py ingest --src <folder>  then  ... embed")
    if len(names) > 1:
        log(f"note: multiple tables found, using '{names[0]}'. Pass --table to override.")
    return db.open_table(names[0])


def _meta_of(table) -> dict:
    """Read model_tag / dims from stored rows (source of truth, not config)."""
    try:
        rows = table.search().limit(64).to_list()
    except Exception:
        return {}
    tags = {r.get("model_tag") for r in rows if r.get("model_tag")}
    dims = {r.get("dims") for r in rows if r.get("dims")}
    return {
        "model_tag": sorted(tags)[0] if len(tags) == 1 else None,
        "model_tags": sorted(t for t in tags if t),
        "dims": sorted(dims)[0] if len(dims) == 1 else None,
        "dims_set": sorted(d for d in dims if d),
    }


def weights_file() -> Path:
    """Per-table folder weights live next to the index, not inside it.

    A sidecar JSON (instead of a new column) means old tables keep working:
    missing entries simply behave as weight 1.0.
    """
    return INDEX_DIR / "weights.json"


def load_weights(table: str) -> dict[str, float]:
    """Return {label: weight} for a table. Empty when nothing was stored."""
    try:
        data = json.loads(weights_file().read_text(encoding="utf-8"))
    except Exception:                                # noqa: BLE001
        return {}
    if not isinstance(data, dict):
        return {}
    raw = data.get(table, {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def save_weights(table: str, weights: dict[str, float]) -> None:
    """Persist {label: weight} for a table, keeping other tables intact."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(weights_file().read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except Exception:                                # noqa: BLE001
        data = {}
    data[table] = {str(k): float(v) for k, v in weights.items()}
    weights_file().write_text(json.dumps(data, indent=2), encoding="utf-8")


def apply_weights(hits: list[dict], weights: dict[str, float]) -> list[dict]:
    """Re-rank vector hits by per-folder weight.

    The folder label is the first segment of `source` (`label/rel/path`).
    Score becomes cos * weight, converted back to a distance so downstream
    code (display, RRF) keeps working unchanged. Rows without a stored
    weight, and FTS-only rows without a distance, pass through untouched.
    """
    if not weights or all(w == 1.0 for w in weights.values()):
        return hits
    out: list[dict] = []
    for h in hits:
        h = dict(h)
        src = str(h.get("source", ""))
        label = src.split("/")[0] if "/" in src else ""
        w = weights.get(label, 1.0)
        d = h.get("_distance")
        if d is not None and w != 1.0:
            try:
                h["_distance"] = 1.0 - (1.0 - float(d)) * w
                h["_weight"] = w
            except (TypeError, ValueError):
                pass
        out.append(h)
    out.sort(key=lambda r: (r.get("_distance") is None,
                            r.get("_distance") or 0.0))
    return out


def embed_texts(client, model: str, texts: list[str], *, verbose: bool = False):
    """Embed a batch with truncation disabled and a bounded context.

    Returns (vectors, prompt_eval_counts). Raises RuntimeError on over-length
    input (truncate=False) or persistent OOM. Transient errors (timeout,
    connection reset, 429/500) are retried EMBED_RETRIES times with backoff
    and a halved batch instead of hanging or dying on one bad batch.
    """
    batch = BATCH
    attempt = 0
    while True:
        try:
            resp = client.embed(
                model=model,
                input=texts,
                truncate=TRUNCATE,
                options={"num_ctx": NUM_CTX},
            )
            counts = []
            if getattr(resp, "prompt_eval_count", None) is not None:
                counts = [resp.prompt_eval_count] * len(texts)
            elif isinstance(resp, dict) and resp.get("prompt_eval_count") is not None:
                counts = [resp["prompt_eval_count"]] * len(texts)
            return (resp.embeddings if not isinstance(resp, dict) else resp["embeddings"]), counts
        except Exception as e:                       # noqa: BLE001
            msg = str(e)
            if "out of memory" in msg.lower() or "allocate" in msg.lower():
                if batch <= MIN_BATCH:
                    raise RuntimeError(
                        f"VRAM exhausted even at batch {batch}: {msg}\n"
                        "Lower NUM_CTX, or unload other models with `ollama ps`."
                    ) from e
                batch = max(MIN_BATCH, batch // 2)
                if verbose:
                    log(f"  OOM -> retrying with batch {batch}")
                texts = texts[:batch]
                attempt = 0
                continue
            transient = any(
                s in msg.lower()
                for s in ("timeout", "timed out", "connection", "reset",
                          "broken pipe", "429", "500", "502", "503", "504")
            )
            if transient and attempt < EMBED_RETRIES:
                attempt += 1
                wait = 2.0 * attempt
                if verbose:
                    log(f"  transient error ({msg[:120]}) -> "
                        f"retry {attempt}/{EMBED_RETRIES} in {wait:.0f}s "
                        f"with batch {max(MIN_BATCH, batch // 2)}")
                time.sleep(wait)
                batch = max(MIN_BATCH, batch // 2)
                texts = texts[:batch]
                continue
            raise


def embed_one(client, model: str, text: str):
    vecs, counts = embed_texts(client, model, [text])
    return vecs[0], (counts[0] if counts else None)


def probe_dims(client, model: str) -> int:
    """One cheap call to learn the embedding dimension, without storing it.

    LanceDB needs a fixed-size vector column at table-creation time, so the
    dimension has to be known before the first vector is written.
    """
    vec, _ = embed_one(client, model, "dimension probe")
    return len(vec)


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #

@dataclass
class SourceSpec:
    """One folder to ingest, with its own extension filter and optional label.

    The label disambiguates files when several folders land in one table:
    `docs/readme.md` vs `code/readme.md`.
    """
    path: str
    only: str = ""
    label: str = ""
    weight: float = 1.0

    @property
    def root(self) -> Path:
        return Path(self.path).expanduser().resolve()

    @property
    def ext_filter(self) -> set[str] | None:
        if not self.only:
            return None
        return {("." + e.strip().lstrip(".")).lower()
                for e in self.only.split(",") if e.strip()}


def _collect_files(spec: SourceSpec) -> list[Path]:
    root = spec.root
    if not root.exists():
        fail(f"source not found: {root}")
    import extract as X
    ext_filter = spec.ext_filter
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        if p.stat().st_size == 0:
            continue
        if ext_filter and p.suffix.lower() not in ext_filter:
            continue
        if X.detect_kind(p) == "unknown":
            continue
        out.append(p)
    return out


def ingest_sources(
    specs: list[SourceSpec],
    *,
    table: str,
    model: str,
    dims: int = 0,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    replace: bool = False,
    append: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Extract + chunk one or more folders into a single table.

    Shared by the CLI (single `--src`) and the GUI (several folder rows).
    Returns a summary dict; raises PipelineError on failure.
    """
    import extract as X
    import chunking as C

    if not specs:
        fail("no source folders given")

    # gather every candidate file first, so we can report a total up front
    labels = [spec.label.strip().strip("/") for spec in specs]
    work: list[tuple[Path, Path, str]] = []      # (file, root, label)
    for spec, label in zip(specs, labels):
        for p in _collect_files(spec):
            work.append((p, spec.root, label))

    if not work:
        fail("no supported files under the given folder(s)")

    db = None if dry_run else _db()
    name = table
    existing = None
    existing_sources: set[str] = set()
    if db is not None and name in _table_names(db):
        if replace:
            log(f"\ndropping existing table '{name}'")
            db.drop_table(name)
        elif append:
            existing = db.open_table(name)
        else:
            fail(f"table '{name}' already exists -- pass --replace to rebuild, "
                 f"--append to add, or --table <new name>")

    # Append mode: learn the table's shape and contents so new rows fit it.
    if existing is not None:
        meta0 = _meta_of(existing)
        if meta0.get("model_tags") and model not in meta0["model_tags"]:
            fail(
                f"table '{name}' already holds vectors from {meta0['model_tags']}.\n"
                f"Appending data for {model} would mix incompatible vector spaces.\n"
                f"Use the same model, or reindex for {model}."
            )
        schema_dims = 0
        try:
            for field in existing.schema:
                if field.name == "vector":
                    schema_dims = field.type.list_size or 0
        except Exception:                                # noqa: BLE001
            pass
        if not schema_dims and meta0.get("dims"):
            schema_dims = meta0["dims"]
        if dims and schema_dims and dims != schema_dims:
            fail(f"--dims {dims} does not match table '{name}' ({schema_dims})")
        if schema_dims:
            dims = schema_dims
        try:
            existing_sources = {
                str(r.get("source", "")) for r in
                existing.search().select(["source"]).to_list()
            }
        except Exception:                                # noqa: BLE001
            existing_sources = set()
        log(f"appending to '{name}' ({len(existing_sources)} sources indexed)")

    log(f"scanning {len(specs)} folder(s)")
    log(f"{len(work)} candidate files")

    rows: list[dict] = []
    skipped: list[str] = []
    skipped_existing = 0
    by_kind: Counter = Counter()

    for fnum, (path, root, label) in enumerate(work, start=1):
        try:
            title, blocks, kind = X.extract(path, root=root)
        except Exception as e:                       # noqa: BLE001  (Skipped included)
            skipped.append(f"{path} ({e})")
            continue

        rel = str(path.relative_to(root)).replace("\\", "/")
        if label:
            rel = f"{label}/{rel}"
        if rel in existing_sources:
            skipped_existing += 1
            emit_progress(fnum, len(work))
            continue
        by_kind[kind] += len(blocks)
        counter = 0

        for b in blocks:
            if kind == "code":
                chunks = C.chunk_code(
                    b.text, section=b.section, title=title,
                    max_size=CODE_MAX, min_size=CHUNK_MIN,
                    start_index=counter, meta={"source_rel": rel},
                )
            else:
                chunks = C.chunk_prose(
                    b.text, section=b.section, title=title, kind="prose",
                    size=size, overlap=overlap, min_size=CHUNK_MIN,
                    start_index=counter, meta={"source_rel": rel},
                )
            for i, ch in enumerate(chunks):
                # Drop only *trailing* tiny fragments. The first chunk of a
                # block is always kept -- a short but meaningful paragraph
                # still belongs in the index.
                if i > 0 and ch.token_count < CHUNK_MIN:
                    continue
                counter += 1
                rows.append({
                    "id": str(uuid.uuid4()),
                    "text_raw": ch.text_raw,
                    "embed_input": ch.embed_input,
                    "title": title,
                    "section": ch.section,
                    "source": rel,
                    "kind": "code" if kind == "code" else "prose",
                    "chunk_index": ch.chunk_index,
                    "token_count": ch.token_count,
                    "model_tag": "",           # filled by `embed`
                    "dims": 0,                 # filled by `embed`
                    "embedded": 0,
                    "ingested_at": time.time(),
                })

        if verbose:
            log(f"  {rel}: {len(blocks)} blocks")

        emit_progress(fnum, len(work))

    if not rows:
        if existing is not None and skipped_existing:
            log(f"nothing new -- all {skipped_existing} file(s) already indexed "
                f"in '{name}'")
            return {"files": len(work), "blocks": dict(by_kind), "chunks": 0,
                    "skipped": len(skipped), "skipped_existing": skipped_existing,
                    "appended": True, "sources": [str(s.root) for s in specs],
                    "labels": [lb for lb in labels if lb],
                    "weights": {lb: specs[k].weight for k, lb in enumerate(labels) if lb}}
        fail("no chunks produced (files too small or all filtered out)")

    log(f"\nblocks by format: {dict(by_kind)}")
    log(f"chunks produced:  {len(rows)}")
    if existing is not None and skipped_existing:
        log(f"already indexed, skipped: {skipped_existing}")
    if skipped:
        log(f"skipped: {len(skipped)}")
        for s in skipped[:10]:
            log(f"  - {s}")

    summary = {"files": len(work), "blocks": dict(by_kind), "chunks": len(rows),
               "skipped": len(skipped), "skipped_existing": skipped_existing,
               "appended": existing is not None,
               "sources": [str(s.root) for s in specs],
               "labels": [lb for lb in labels if lb],
               "weights": {lb: specs[k].weight for k, lb in enumerate(labels) if lb}}

    if dry_run:
        sample = rows[0]
        log("\n--dry-run, first chunk --")
        log(f"section: {sample['section']}")
        log(f"tokens:  {sample['token_count']}")
        log("embed_input preview:\n" + sample["embed_input"][:600])
        return summary

    # LanceDB needs a fixed-size vector column at creation time, so learn the
    # dimension now (one cheap call, nothing stored). Fall back to dims if
    # Ollama is offline; `embed` will still verify the real vector size.
    if not dims:
        try:
            dims = probe_dims(_client(), model)
            log(f"probed dims={dims} from {model}")
        except Exception as e:                       # noqa: BLE001
            fail(
                f"could not probe embedding dimension from Ollama ({e}).\n"
                f"Start Ollama, or pass --dims <n> (e.g. --dims 1024 for {model})."
            )
    summary["dims"] = dims

    import pyarrow as pa  # type: ignore
    # Every row must carry the same columns as the schema, including an empty
    # vector placeholder (filled later by `embed`).
    for r in rows:
        r["vector"] = [0.0] * dims
    schema = pa.schema([
        ("id", pa.string()),
        ("text_raw", pa.string()),
        ("embed_input", pa.string()),
        ("title", pa.string()),
        ("section", pa.string()),
        ("source", pa.string()),
        ("kind", pa.string()),
        ("chunk_index", pa.int64()),
        ("token_count", pa.int64()),
        ("model_tag", pa.string()),
        ("dims", pa.int64()),
        ("embedded", pa.int64()),
        ("ingested_at", pa.float64()),
        ("vector", pa.list_(pa.float32(), dims)),
    ])
    if existing is not None:
        for r in rows:
            r["vector"] = [0.0] * dims
        try:
            existing.add(rows)
        except Exception as e:                       # noqa: BLE001
            fail(f"could not append to '{name}': {e}")
        log(f"\nappended {len(rows)} rows -> table '{name}'")
    else:
        db.create_table(name, data=rows, schema=schema)
        log(f"\nwrote table '{name}' -> {INDEX_DIR}")
    try:
        save_weights(name, {**load_weights(name), **summary["weights"]})
    except Exception as e:                            # noqa: BLE001
        log(f"  (weights not saved: {e})")
    log("next: python pipeline.py embed --table " + name)
    return summary


def cmd_ingest(args) -> None:
    ingest_sources(
        [SourceSpec(path=args.src, only=args.only, label="",
                    weight=args.weight)],
        table=args.table, model=args.model, dims=args.dims,
        size=args.size, overlap=args.overlap,
        replace=args.replace, append=args.append,
        dry_run=args.dry_run, verbose=args.verbose,
    )


# --------------------------------------------------------------------------- #
# embed
# --------------------------------------------------------------------------- #

def cmd_embed(args) -> None:
    model = args.model or DEFAULT_MODEL
    table = _open_table(args.table)
    name = table.name if hasattr(table, "name") else (args.table or "?")
    meta = _meta_of(table)

    if meta.get("model_tags") and model not in meta["model_tags"]:
        fail(
            f"table '{name}' already holds vectors from {meta['model_tags']}.\n"
            f"Embedding {model} into it would mix incompatible vector spaces.\n"
            f"Use: python pipeline.py reindex --model {model}"
        )

    try:
        rows = table.search().to_list()
    except Exception as e:
        fail(f"cannot read table '{name}': {e}")

    todo = [r for r in rows if not r.get("embedded")]
    if not todo:
        log(f"table '{name}' is fully embedded ({len(rows)} rows, model={meta.get('model_tag')})")
        _ensure_fts_index(table)
        return {"embedded": 0, "dims": meta.get("dims"),
                "model": meta.get("model_tag"), "table": name,
                "truncations": 0}

    log(f"embedding {len(todo)} / {len(rows)} rows with {model}")
    log(f"num_ctx={NUM_CTX} truncate={TRUNCATE} batch={BATCH}")
    client = _client()

    # verify the model is actually available
    try:
        client.show(model)
    except Exception:
        try:
            client.pull(model)
        except Exception as e:
            fail(f"cannot pull model '{model}': {e}")

    updated: list[dict] = []
    dims: int | None = meta.get("dims")
    started = time.time()
    done = 0
    truncations = 0
    flushed = 0
    skipped_overlong = 0

    # Pre-filter rows the server would reject outright (truncate=False).
    # Previously a single over-long chunk aborted the whole run mid-way.
    ok_todo: list[dict] = []
    for r in todo:
        try:
            tc = int(r.get("token_count") or 0)
        except (TypeError, ValueError):
            tc = 0
        if tc > NUM_CTX:
            skipped_overlong += 1
            truncations += 1
            continue
        ok_todo.append(r)
    if skipped_overlong:
        log(f"skipping {skipped_overlong} row(s) exceeding "
            f"num_ctx={NUM_CTX} (truncate=False)")
    todo = ok_todo
    if not todo:
        fail(f"nothing to embed: all {len(ok_todo) + skipped_overlong} pending "
             f"rows exceed num_ctx={NUM_CTX}")

    def checkpoint() -> None:
        """Persist progress so a hang/crash loses at most one checkpoint."""
        nonlocal flushed
        pending = updated[flushed:]
        if not pending:
            return
        _merge_rows(table, pending)
        flushed = len(updated)
        log(f"  checkpoint: {flushed}/{n} saved")

    def flush(batch_rows: list[dict]) -> int:
        """Embed one batch. Returns how many rows were actually embedded.

        `embed_texts` may shrink the batch on OOM/transient retry and return
        fewer vectors than requested -- the caller re-queues the remainder
        instead of dropping it.
        """
        nonlocal dims, done, truncations
        texts = [p["embed_input"] for p in batch_rows]
        try:
            vecs, _ = embed_texts(client, model, texts, verbose=True)
        except RuntimeError as e:
            # persist whatever we already embedded before dying
            if updated:
                try:
                    _merge_rows(table, updated)
                    fail(f"{e}\\nembedded {len(updated)} rows already saved -- "
                         f"rerun `embed` to continue from here")
                except Exception:
                    pass
            fail(str(e))
        except Exception as e:                       # noqa: BLE001
            if updated[flushed:]:
                try:
                    _merge_rows(table, updated[flushed:])
                except Exception:
                    pass
            fail(f"embedding failed after {done} rows: {e}")

        if dims is None:
            dims = len(vecs[0])
        for p, v in zip(batch_rows, vecs):
            if len(v) != dims:
                fail(f"inconsistent vector size {len(v)} != {dims} -- aborting before corrupting the index")
            updated.append({**p, "vector": v, "model_tag": model,
                            "dims": dims, "embedded": 1})
        done += len(vecs)
        return len(vecs)

    n = len(todo)
    i = 0
    while i < n:
        got = flush(todo[i:i + BATCH])
        if got <= 0:
            fail("embedding made no progress -- aborting to avoid an infinite loop")
        i += got
        rate = done / max(1e-6, time.time() - started)
        log(f"  {done}/{n}  ({rate:.1f} rows/s, dims={dims})")
        emit_progress(min(done, n), n)
        if len(updated) - flushed >= CHECKPOINT_EVERY:
            try:
                checkpoint()
            except Exception as e:                   # noqa: BLE001
                log(f"  checkpoint failed ({e}); continuing in memory")
    emit_progress(n, n)

    if not updated:
        fail("nothing embedded")

    if updated[flushed:]:
        _merge_rows(table, updated[flushed:])
    flushed = len(updated)
    log(f"\nembedded {len(updated)} rows -> dims={dims} model={model}")
    _ensure_fts_index(table)
    if truncations:
        log(f"WARNING: {truncations} rows exceed num_ctx={NUM_CTX}; they were "
            f"skipped before sending (truncate=False) -- inspect with --verbose")
    log(f"elapsed {time.time() - started:.1f}s")
    log("next: python pipeline.py query \"your question\"")
    return {"embedded": len(updated), "dims": dims, "model": model,
            "table": name, "truncations": truncations}


def _merge_rows(table, rows: list[dict]) -> None:
    """Upsert embedded rows into the table, keyed on `id`.

    LanceDB's `merge` only *adds* columns; updating existing columns (vector,
    model_tag, dims, embedded) requires `merge_insert`. We send the full row
    so matched rows are updated and (theoretically) new ids inserted.
    """
    import pyarrow as pa  # type: ignore

    dims = len(rows[0].get("vector") or [])
    vec_type = pa.list_(pa.float32(), dims) if dims else pa.list_(pa.float32())

    cols = ["id", "text_raw", "embed_input", "title", "section", "source",
            "kind", "chunk_index", "token_count", "model_tag", "dims",
            "embedded", "ingested_at", "vector"]
    arrays = []
    for c in cols:
        vals = [r.get(c) for r in rows]
        if c == "vector":
            arrays.append(pa.array([list(map(float, v)) for v in vals], type=vec_type))
        elif c in ("chunk_index", "token_count", "dims", "embedded"):
            arrays.append(pa.array(vals, type=pa.int64()))
        elif c == "ingested_at":
            arrays.append(pa.array(vals, type=pa.float64()))
        else:
            arrays.append(pa.array(vals, type=pa.string()))
    data = pa.table(arrays, names=cols)

    b = table.merge_insert(on="id")
    b.when_matched_update_all()
    b.execute(data)


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #

def cmd_query(args) -> None:
    table = _open_table(args.table)
    name = table.name if hasattr(table, "name") else (args.table or "?")
    meta = _meta_of(table)

    if not meta.get("model_tag"):
        fail(f"table '{name}' has no embedded rows yet -- run: python pipeline.py embed --table {name}")

    model = args.model or meta["model_tag"]
    if model != meta["model_tag"]:
        fail(
            f"table '{name}' was built with {meta['model_tag']}, but you asked for {model}.\n"
            "Vectors from different models are not comparable -- this is refused on purpose."
        )

    client = _client()
    qtext = QUERY_INSTRUCT + args.text
    vec, seen = embed_one(client, model, qtext)

    if meta.get("dims") and len(vec) != meta["dims"]:
        fail(f"query vector dim {len(vec)} != index dim {meta['dims']}")

    k = args.top_k
    try:
        # cosine always: the embedding model emits normalized vectors, so cosine
        # is the correct similarity. Using LanceDB's default (L2) would both
        # re-rank differently and make the displayed `cos=` meaningless.
        res = (table.search(vec, vector_column_name="vector")
               .metric("cosine").limit(k * 3))          # .metric(), not a kwarg
        hits = res.to_list()
    except Exception as e:
        # fall back to pure vector search if the metric/query form is rejected
        try:
            hits = table.search(vec, vector_column_name="vector").limit(k * 3).to_list()
        except Exception:
            fail(f"search failed: {e}")

    # folder importance: re-rank by per-label weight before fusion
    hits = apply_weights(hits, load_weights(name))

    # optional FTS half, fused with RRF
    if args.hybrid:
        fts_hits = _fts_search(table, args.text, k * 3)
        hits = _rrf([hits, fts_hits], k)
    else:
        hits = hits[:k]

    mode = "hybrid" if args.hybrid else "vector"
    log(f"model={model}  table={name}  dims={meta.get('dims')}  "
        f"mode={mode}  hits={len(hits)}\n")
    for i, h in enumerate(hits[:k], 1):
        score = h.get("_distance")
        rel = ""
        if score is not None:
            try:
                rel = f"  cos={1 - float(score):.4f}"
            except (TypeError, ValueError):
                rel = f"  dist={score}"
        w = h.get("_weight")
        wtag = ""
        if isinstance(w, (int, float)) and not isinstance(w, bool) and w != 1.0:
            wtag = f"  w={w:g}"
        log(f"[{i}] {h.get('title','')} > {h.get('section','')}{rel}{wtag}")
        log(f"    {h.get('source','')}  (kind={h.get('kind')}, {h.get('token_count')} tok)")
        body = (h.get("text_raw") or "").replace("\n", " ")
        log(f"    {body[:args.width]}\n")


def _fts_search(table, text: str, k: int) -> list[dict]:
    try:
        return table.search(text, query_type="fts").limit(k).to_list()
    except Exception:
        return []


def _ensure_fts_index(table) -> bool:
    """Build a full-text-search index on text_raw if it is missing.

    Needed for hybrid (vector + BM25 via RRF) search. Safe to call repeatedly.
    """
    try:
        table.create_fts_index("text_raw", use_tantivy=False, replace=True)
        return True
    except Exception as e:                            # noqa: BLE001
        log(f"  FTS index not built ({e}); hybrid falls back to vector-only")
        return False


def _rrf(result_sets: list[list[dict]], k: int, rr: int = 60) -> list[dict]:
    """Fuse ranked lists with Reciprocal Rank Fusion.

    Rows seen in more than one list keep fields from both: the vector half's
    `_distance` is preserved when the FTS half (which only has `_score`)
    overwrites the entry.
    """
    scores: dict[str, float] = defaultdict(float)
    rows: dict[str, dict] = {}
    for rs in result_sets:
        for rank, r in enumerate(rs, 1):
            rid = r.get("id") or id(r)
            scores[rid] += 1.0 / (rr + rank)
            cur = rows.get(rid)
            if cur is None:
                rows[rid] = dict(r)
            else:
                cur.update(r)                     # union, no field loss
    ordered = sorted(scores, key=scores.get, reverse=True)[:k]
    out = []
    for rid in ordered:
        r = dict(rows[rid])
        r["_rrf"] = scores[rid]
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #

def cmd_stats(args) -> None:
    db = _db()
    names = _table_names(db)
    if not names:
        log(f"index empty: {INDEX_DIR}")
        return

    total_rows = 0
    for n in names:
        t = db.open_table(n)
        try:
            rows = t.search().to_list()
        except Exception as e:
            log(f"{n}: unreadable ({e})")
            continue
        total_rows += len(rows)
        by_kind = Counter(r.get("kind") for r in rows)
        by_source = Counter(r.get("source") for r in rows)
        by_tag = Counter(r.get("model_tag") or "(not embedded)" for r in rows)
        dims = {r.get("dims") for r in rows if r.get("dims")}
        embedded = sum(1 for r in rows if r.get("embedded"))
        toks = sum(int(r.get("token_count") or 0) for r in rows)
        d = list(dims)[0] if len(dims) == 1 else dims
        vec_bytes = 0
        if isinstance(d, int):
            vec_bytes = embedded * d * 4

        log(f"\ntable      : {n}")
        log(f"rows       : {len(rows)}  (embedded {embedded})")
        log(f"model_tag  : {dict(by_tag)}")
        log(f"dims       : {d}")
        log(f"kind       : {dict(by_kind)}")
        log(f"tokens     : {toks:,} (~{toks / 4.0:,.0f} KB text)")
        log(f"vectors    : {vec_bytes / 1e6:.1f} MB as float32"
            + (f" / {embedded * d:.0f} KB as int8" if isinstance(d, int) else ""))
        log(f"sources    : {len(by_source)} files")
        if args.verbose:
            for s, c in by_source.most_common(25):
                log(f"    {c:5d}  {s}")

    log(f"\ntotal rows across index: {total_rows}")
    try:
        size = sum(f.stat().st_size for f in INDEX_DIR.rglob("*") if f.is_file())
        log(f"index on disk: {size / 1e6:.1f} MB  ({INDEX_DIR})")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #

def cmd_eval(args) -> None:
    if not Path(args.queries).exists():
        fail(f"queries file not found: {args.queries}")

    queries: list[str] = []
    with open(args.queries, encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        if "," in sample.splitlines()[0] and sample.lstrip()[0] != '"':
            for row in csv.reader(fh):
                if row and row[0].strip():
                    queries.append(row[0].strip())
        else:
            queries = [ln.strip() for ln in fh if ln.strip()]

    if args.gold:
        pairs: list[tuple[str, str]] = []
        with open(args.gold, encoding="utf-8-sig", newline="") as fh:
            for i, row in enumerate(csv.reader(fh)):
                if len(row) >= 2 and row[0].strip():
                    # skip a header row if present
                    if i == 0 and row[0].strip().lower() in {"query", "q", "question"} \
                            and row[1].strip().lower() in {"gold", "gold_source", "source", "expected", "file"}:
                        continue
                    pairs.append((row[0].strip(), row[1].strip()))
    else:
        pairs = [(q, "") for q in queries]

    tables = args.tables.split(",") if args.tables else table_names()
    if not tables:
        fail("no tables to evaluate")

    client = _client()
    results: dict[str, dict] = {}

    for tname in tables:
        table = _open_table(tname)
        meta = _meta_of(table)
        model = meta.get("model_tag")
        if not model:
            log(f"skip {tname}: not embedded")
            continue
        log(f"evaluating {tname} ({model}) on {len(pairs)} queries ...")
        has_gold = any(g for _, g in pairs)
        hits_total = 0
        ndcg_total = 0.0
        scored = 0
        use_hybrid = bool(getattr(args, "hybrid", False))
        for qtext, gold in pairs:
            try:
                vec, _ = embed_one(client, model, QUERY_INSTRUCT + qtext)
                try:
                    vhits = (table.search(vec, vector_column_name="vector")
                             .metric("cosine").limit(30).to_list())
                except Exception:                      # noqa: BLE001
                    vhits = (table.search(vec, vector_column_name="vector")
                             .limit(30).to_list())
                vhits = apply_weights(vhits, load_weights(tname))
                if use_hybrid:
                    fhits = _fts_search(table, qtext, 30)
                    hits = _rrf([vhits, fhits], 10) if fhits else vhits[:10]
                else:
                    hits = vhits[:10]
            except Exception as e:
                log(f"  query failed: {e}")
                continue
            if gold:
                scored += 1
                gold_l = gold.lower()
                ranked = []
                for rank, h in enumerate(hits, 1):
                    blob = " ".join(str(h.get(c, "")) for c in ("source", "title", "section", "text_raw")).lower()
                    ranked.append(1.0 if gold_l in blob else 0.0)
                if any(ranked):
                    hits_total += 1
                ndcg_total += _ndcg(ranked)
        n = max(1, len(pairs))
        results[tname] = {
            "model": model,
            "hit_rate": (hits_total / scored) if (has_gold and scored) else None,
            "ndcg@10": (ndcg_total / scored) if (has_gold and scored) else None,
            "queries": n,
        }

    log("\n=== evaluation ===")
    log(f"{'table':<28} {'model':<30} {'nDCG@10':>8} {'HitRate':>8}")
    if not any(r["ndcg@10"] is not None for r in results.values()):
        log("(no --gold supplied -- cannot score; only a query smoke-test ran)")

    def _key(item):
        v = item[1]["ndcg@10"]
        return v if v is not None else -1.0
    for t, r in sorted(results.items(), key=_key, reverse=True):
        nd = f"{r['ndcg@10']:.4f}" if r["ndcg@10"] is not None else "    n/a"
        hr = f"{r['hit_rate']:.3f}" if r["hit_rate"] is not None else "    n/a"
        log(f"{t:<28} {r['model']:<30} {nd:>8} {hr:>8}")

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["table", "model", "ndcg_at_10", "hit_rate", "queries"])
            w.writeheader()
            for t, r in results.items():
                nd = r["ndcg@10"]
                w.writerow({"table": t, "model": r["model"],
                            "ndcg_at_10": round(nd, 4) if nd is not None else "",
                            "hit_rate": r["hit_rate"] if r["hit_rate"] is not None else "",
                            "queries": r["queries"]})
        log(f"\nwrote {args.out}")


def _ndcg(ranked: list[float]) -> float:
    if not any(ranked):
        return 0.0
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ranked))
    ideal = sorted(ranked, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


# --------------------------------------------------------------------------- #
# reindex
# --------------------------------------------------------------------------- #

def cmd_reindex(args) -> None:
    src_table = _open_table(args.table)
    src_name = src_table.name if hasattr(src_table, "name") else (args.table or "?")
    rows = src_table.search().to_list()

    model = args.model or DEFAULT_MODEL
    new_name = args.dest or f"{src_name}__{model.replace(':', '_').replace('/', '_')}"

    db = _db()
    if new_name in _table_names(db):
        if args.replace:
            db.drop_table(new_name)
        else:
            fail(f"table '{new_name}' exists -- pass --replace")

    # copy metadata only; vectors deliberately dropped. The new table must be
    # sized for the *new* model's dimension, which often differs from the source
    # (e.g. qwen3-embedding 0.6b=1024, 4b=2560). Probe it.
    dims = args.dims
    if not dims:
        try:
            dims = probe_dims(_client(), model)
            log(f"probed dims={dims} from {model}")
        except Exception as e:                       # noqa: BLE001
            fail(
                f"could not probe dims from Ollama ({e}).\n"
                f"Start Ollama, or pass --dims <n> for {model}."
            )
    stripped = []
    for r in rows:
        r = dict(r)
        r["model_tag"] = ""
        r["dims"] = 0
        r["embedded"] = 0
        r["vector"] = [0.0] * dims
        stripped.append(r)

    import pyarrow as pa  # type: ignore
    schema = pa.schema([
        ("id", pa.string()), ("text_raw", pa.string()), ("embed_input", pa.string()),
        ("title", pa.string()), ("section", pa.string()), ("source", pa.string()),
        ("kind", pa.string()), ("chunk_index", pa.int64()), ("token_count", pa.int64()),
        ("model_tag", pa.string()), ("dims", pa.int64()), ("embedded", pa.int64()),
        ("ingested_at", pa.float64()), ("vector", pa.list_(pa.float32(), dims)),
    ])
    db.create_table(new_name, data=stripped, schema=schema)
    log(f"created '{new_name}' from '{src_name}' with {len(stripped)} rows (no vectors, dims={dims})")
    try:
        save_weights(new_name, load_weights(src_name))
    except Exception:                                # noqa: BLE001
        pass
    log(f"next: python pipeline.py embed --table {new_name} --model {model}")


# --------------------------------------------------------------------------- #
# export-md
# --------------------------------------------------------------------------- #

def blocks_to_markdown(title: str, blocks, kind: str,
                       lang_fallback: str = "") -> str:
    """Render extracted blocks as one Markdown document.

    Shared by export-md and link fetching, so both produce exactly what
    ingest would embed.
    """
    parts = [f"# {title}", ""]
    for b in blocks:
        if b.section:
            parts += [f"## {b.section}", ""]
        is_code = (kind == "code") or (b.meta.get("kind_hint") == "code")
        if is_code:
            lang = b.meta.get("lang") or lang_fallback
            parts += [f"```{lang}", b.text, "```", ""]
        else:
            parts += [b.text, ""]
    return "\n".join(parts).strip() + "\n"

def export_markdown(src: str, dest: str, *, only: str = "") -> dict:
    """Convert every supported file under src into .md files under dest.

    Uses the same extractors as ingest, so the output is exactly what the
    pipeline would embed -- clean sources you can review, edit, and ingest
    as Markdown. The output mirrors the source tree with an .md suffix.

    Pick a dest *outside* src, otherwise the next run would pick up the
    generated files again. Returns a summary dict.
    """
    import extract as X

    spec = SourceSpec(path=src, only=only)
    files = _collect_files(spec)
    root = spec.root
    if not files:
        fail(f"no supported files under {root}")
    out_root = Path(dest).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped: list[str] = []
    for fnum, path in enumerate(files, start=1):
        try:
            title, blocks, kind = X.extract(path, root=root)
        except Exception as e:                       # noqa: BLE001
            skipped.append(f"{path} ({e})")
            continue
        if not blocks:
            skipped.append(f"{path} (no content)")
            continue

        parts = blocks_to_markdown(
            title, blocks, kind,
            lang_fallback=path.suffix.lstrip(".") if kind == "code" else "")

        out = (out_root / path.relative_to(root)).with_suffix(".md")
        if out.exists():
            # notes.txt + notes.md would collide -- keep both
            out = out.with_name(f"{out.stem}_{path.suffix.lstrip('.')}.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(parts, encoding="utf-8")
        written += 1
        emit_progress(fnum, len(files))

    log(f"exported {written}/{len(files)} files -> {out_root}")
    if skipped:
        log(f"skipped: {len(skipped)}")
        for s in skipped[:10]:
            log(f"  - {s}")
    return {"files": len(files), "written": written,
            "skipped": len(skipped), "src": str(root),
            "dest": str(out_root)}


def cmd_export_md(args) -> None:
    export_markdown(args.src, args.dest, only=args.only)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Local RAG embedding pipeline (Ollama + LanceDB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("ingest", help="extract + chunk a folder (no vectors)")
    g.add_argument("--src", required=True, help="folder to scan")
    g.add_argument("--table", default="chunks", help="target table name (default: chunks)")
    g.add_argument("--model", default=DEFAULT_MODEL, help="used only to probe vector dims")
    g.add_argument("--dims", type=int, default=0, help="skip the Ollama probe, use this dimension")
    g.add_argument("--size", type=int, default=CHUNK_SIZE, help=f"chunk size in tokens (default {CHUNK_SIZE})")
    g.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help=f"overlap tokens (default {CHUNK_OVERLAP})")
    g.add_argument("--only", default="", help="comma list of extensions, e.g. .md,.pdf,.py")
    g.add_argument("--weight", type=float, default=1.0, help="importance weight for this source (default 1.0)")
    g.add_argument("--replace", action="store_true", help="drop the table if it exists")
    g.add_argument("--append", action="store_true", help="add to the existing table (dedupe by source)")
    g.add_argument("--dry-run", action="store_true", help="show a sample chunk, write nothing")
    g.add_argument("-v", "--verbose", action="store_true")
    g.set_defaults(func=cmd_ingest)

    g = sub.add_parser("embed", help="embed pending rows")
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.add_argument("--table", default=None)
    g.set_defaults(func=cmd_embed)

    g = sub.add_parser("query", help="hybrid search")
    g.add_argument("text", help="the query")
    g.add_argument("--table", default=None)
    g.add_argument("--model", default=None)
    g.add_argument("--top-k", type=int, default=8)
    g.add_argument("--width", type=int, default=220, help="chars of each hit to print")
    g.add_argument("--hybrid", dest="hybrid", action="store_true",
                   help="hybrid vector + FTS/RRF (default)")
    g.add_argument("--no-hybrid", dest="hybrid", action="store_false",
                   help="vector-only (skip the FTS/RRF half)")
    g.set_defaults(func=cmd_query, hybrid=True)

    g = sub.add_parser("stats", help="index statistics")
    g.add_argument("-v", "--verbose", action="store_true", help="per-file chunk counts")
    g.set_defaults(func=cmd_stats)

    g = sub.add_parser("eval", help="compare tables over a query set")
    g.add_argument("--queries", required=True, help="txt/csv, one query per line (or col 0)")
    g.add_argument("--gold", default="", help="csv: query,source-glob -- enables hit-rate")
    g.add_argument("--tables", default="", help="comma list (default: all)")
    g.add_argument("--out", default="", help="write results csv")
    g.add_argument("--hybrid", action="store_true",
                   help="score with hybrid vector + FTS/RRF (default: vector-only)")
    g.set_defaults(func=cmd_eval, hybrid=False)

    g = sub.add_parser("reindex", help="new table for a different model")
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.add_argument("--table", default=None, help="source table")
    g.add_argument("--dest", default=None, help="destination table name")
    g.add_argument("--dims", type=int, default=0, help="skip the Ollama probe, use this dimension")
    g.add_argument("--replace", action="store_true")
    g.set_defaults(func=cmd_reindex)

    g = sub.add_parser("export-md", help="convert a folder into .md sources")
    g.add_argument("--src", required=True, help="folder to convert")
    g.add_argument("--dest", required=True, help="output folder for .md files (outside --src)")
    g.add_argument("--only", default="", help="comma list of extensions, e.g. .docx,.pdf,.html")
    g.set_defaults(func=cmd_export_md)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _ensure_env()
    try:
        args.func(args)
    except PipelineError as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
