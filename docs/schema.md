# Schema & Config

## LanceDB schema

One table per index. Every row is one chunk. `vector` is a **fixed-size**
`list<float32>` so the dimension must be known before the first vector is
written.

| Column | Type | Meaning |
|--------|------|---------|
| `id` | string | UUID, primary key for upserts. |
| `text_raw` | string | Clean source text — what the user sees. |
| `embed_input` | string | Prefixed text — what the model embeds. |
| `title` | string | Document title (first heading or filename). |
| `section` | string | Section / heading the chunk falls under. |
| `source` | string | File path relative to the ingest root. |
| `kind` | string | `prose` or `code`. |
| `chunk_index` | int64 | Position within its block. |
| `token_count` | int64 | Estimated tokens, used for the truncation guard. |
| `model_tag` | string | Which model produced the vector. **Guard.** |
| `dims` | int64 | Vector dimension. **Guard.** |
| `embedded` | int64 | `1` once embedded, else `0` (drives resume). |
| `ingested_at` | float64 | Unix timestamp. |
| `vector` | list<float32> | The embedding. |

## Full-text index (hybrid search)

`embed` builds a BM25 full-text index on `text_raw` via `_ensure_fts_index`
after vectors are written. `query` / Chat read it for the keyword half of
[Hybrid Search](hybrid-search.md) (vector + BM25 fused with RRF, `rr=60`).
If the index is missing (older table, build failure), search degrades
gracefully to vector-only — rerun `embed` to rebuild it.

## Guards

Two columns exist purely to stop silent corruption.

**`model_tag`** — `embed` refuses to mix models in one table, and `query`
refuses to run if the requested model differs from the index. Vectors from
different models are not comparable, so a wrong-model query would return
plausible-looking nonsense.

**`dims`** — vector dimensions differ between models (`qwen3-embedding:0.6b` →
1024, `4b` → 2560). A mismatch aborts before anything is written.

!!! danger "Changing models requires a full re-index"
    Use [`reindex`](commands.md#reindex) to build a clean table for the new
    model, then `embed` it. Never point a different model at an existing
    table.

## Why the dimension is probed

LanceDB needs a fixed-size vector column at table-creation time, but the
dimension is only known after the first embedding. So `ingest` and `reindex`
make **one cheap probe call** to Ollama to learn `dims`, then fill in
placeholder zeros. Nothing is stored from the probe.

If Ollama is offline, pass `--dims <n>` explicitly (e.g. `--dims 1024`).

## Hardware configuration

Chosen for a 4 GB VRAM card. These are pinned in `pipeline.py`.

| Setting | Value | Reason |
|---------|-------|--------|
| `NUM_CTX` | `8192` | Well below the model's advertised 32 K. KV cache grows with context, so a smaller window keeps the model resident. Chunks are ~900 tokens to fit comfortably. |
| `TRUNCATE` | `False` | Over-long input makes Ollama **error** instead of silently cutting. A loud failure beats a silently truncated chunk. |
| `BATCH` | `8`, halves on OOM | Starts at 8, auto-halves to a floor of 2 if VRAM runs out. |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | Halves the KV cache footprint. Set on the Ollama server, not per-request. |

!!! note "Truncation guard"
    `token_count` is compared against `NUM_CTX`. A chunk that exceeds it is
    caught rather than quietly embedded wrong.

## Storage math

1024-dim float32 = 4 bytes per value.

| Scale | float32 | int8 | 256-dim (MRL) |
|-------|---------|------|----------------|
| 100 K vectors | 0.4 GB | 0.1 GB | 0.1 GB |
| 1 M vectors | 4.0 GB | 1.0 GB | 1.0 GB |

Run `stats` for your real numbers before optimizing — int8 quantization cuts
storage 4×, and Matryoshka (MRL) truncation to 256 dims gives another 4× **if
recall holds**. Verify with `eval` on a gold set before adopting either.

## Pitfalls found during implementation

These are the non-obvious ones worth knowing if you modify the pipeline:

1. **`table.merge()` cannot update columns.** LanceDB's `merge` only *adds*
   columns. Updating `vector` / `model_tag` / `dims` / `embedded` needs
   `merge_insert(on="id")` with `when_matched_update_all()`.
2. **`Table.search()` ignores a `metric=` keyword.** It must be set via
   `.metric("cosine")`. Passing it as a keyword silently falls back to L2,
   which both re-ranks differently and makes the reported similarity wrong —
   the bug is invisible because results still look plausible.
3. **`log()` must survive a non-UTF-8 stdout.** `python pipeline.py ... >
   out.txt` on Windows gives stdout a cp1252 codec, and any `✓`/`·` character
   raises `UnicodeEncodeError` and aborts an otherwise fine run.
4. **`list_tables()` returns a response object.** In LanceDB 0.17+ the returned
   list lives under a `.tables` attribute, not directly.
5. **The min-size filter must key off chunk position.** Filtering every chunk
   below 50 tokens silently deletes whole short blocks (a 2-block HTML page
   vanished entirely). Only *trailing* fragments should be dropped.
6. **RRF must union rows, not overwrite.** The FTS half has `_score` where the
   vector half has `_distance`; replacing the vector row loses its cosine.
7. **Never open a modal dialog from a constructor.** A dialog in `__init__`
   blocks before the window appears, and a headless test hangs with no output.
8. **Every supported extension needs a handler entry.** `.xhtml` was listed as
   supported but missing from the dispatch table — a `KeyError` on first use.
9. **OOM retries must re-queue the remainder.** Halving the batch and embedding
   only the head silently skipped the tail rows for that run (recoverable via
   resume, but progress lied). The embed loop now advances by the number of
   vectors actually returned.
10. **Never rebuild the FTS index per query.** `create_fts_index(replace=True)`
    on every search is pure overhead — build once after `embed`, then only
    read.
