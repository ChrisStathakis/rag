# embed_process — local RAG embedding pipeline

Runs entirely on your machine: **Ollama** (inference) + **LanceDB** (storage +
hybrid search). Built for a GTX 1050 Ti (4 GB VRAM) / 16 GB RAM, English-only
corpus.

Default model: `qwen3-embedding:0.6b` (639 MB, 32 K context, 1024-dim,
MTEB(eng,v2) 70.70 / MTEB-Code 75.41).

## Install

```powershell
cd D:\christos_projects\simvoulatoras\embed_process
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
ollama pull qwen3-embedding:0.6b
```

The pipeline works without a GPU (CPU is fine for a 0.6 b model), but to force
GPU offload start Ollama with the KV cache quantized — this halves VRAM:

```powershell
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"
ollama serve
```

Verify the model actually landed on the GPU by checking `ollama ps` and the
server log for `offloaded N/M layers`.

## Quick start

```powershell
# 1. inspect what would be produced (writes nothing)
.\.venv\Scripts\python.exe pipeline.py ingest --src <corpus_folder> --table corpus --dry-run

# 2. extract + chunk + write text + metadata (no vectors yet)
.\.venv\Scripts\python.exe pipeline.py ingest --src <corpus_folder> --table corpus

# 3. embed the pending rows into vectors (batched, resumable)
.\.venv\Scripts\python.exe pipeline.py embed --table corpus

# 4. search (hybrid = vector + BM25 fused with RRF)
.\.venv\Scripts\python.exe pipeline.py query "how does attention work" --table corpus
```

A tiny sample corpus ships in `sample/` (one file per format) so you can smoke
test before pointing it at your real documents:

```powershell
.\.venv\Scripts\python.exe pipeline.py ingest --src sample --table sample --replace
.\.venv\Scripts\python.exe pipeline.py embed  --table sample
.\.venv\Scripts\python.exe pipeline.py query  "cosine similarity function" --table sample
```

## Subcommands

| command  | what it does |
|----------|--------------|
| `ingest` | walk a folder, extract text, chunk it, store text + metadata (no vectors). `--dry-run` previews one chunk. `--only .md,.pdf,.py` filters extensions. |
| `embed`  | batch-embed every row where `embedded=0`. Auto-pulls the model if missing. Resumable — rerun it after an interruption and it continues. |
| `query`  | hybrid search (vector + FTS/RRF). Refuses to run if the requested model does not match the index. |
| `stats`  | chunk counts by kind/model/dims, token totals, on-disk size. |
| `eval`   | run a query set over one or more tables, print nDCG@10 + HitRate. `--gold` CSV enables scoring. |
| `reindex`| build a fresh table for a *different* model (copies text + metadata, drops vectors, probes new dims). |

## Why these settings

- **`num_ctx=8192`** — pinned well below the model's advertised 32 K. On 4 GB
  VRAM the KV cache grows with context length, so a smaller window keeps the
  model resident. Chunks are sized to ~900 tokens to fit comfortably.
- **`truncate=False`** — over-long input makes Ollama *error* instead of
  silently cutting text. A loud failure beats a silently truncated chunk.
- **Batch 8, auto-halves on OOM** — if the GPU runs out mid-run the batch size
  is cut and retried down to a floor of 2.
- **`embed_input` vs `text_raw`** — the model sees a contextualised
  `Document: {title} > {section}` prefix (Qwen3 reports +1-5 % retrieval with
  instruction/prefix); the user sees the clean `text_raw`.
- **`model_tag` + `dims` columns** — anti-corruption guards. Changing the model
  means a full re-index (`reindex`), never mixing vector spaces. `query`
  refuses on any mismatch.

## Chunking

- **Prose** (md/txt/rst/html/docx/pdf): recursive split into sentence-ish
  units, packed to ~900 tokens with ~120-token overlap. Never cuts mid-word.
- **Code**: split per function/class by a lightweight parser; imports are kept
  as a header on every piece. Oversized functions are cut on line boundaries
  with one-line overlap.

## Supported formats

`.md .markdown .rst .txt` · `.html .htm` · `.docx` · `.pdf` · and ~30 code
extensions (`.py .js .ts .go .rs .java .c .cpp .rb ...`). Unknown extensions
are skipped and logged, never raised.

## Files

```
extract.py    5 format handlers -> common Block(text, section, meta)
chunking.py   prose splitter + structural code splitter -> Chunk
pipeline.py   CLI: ingest / embed / query / stats / eval / reindex / export-md
store.py      SQLite project store (settings only, no vectors)
chat_providers.py  Anthropic/OpenAI backends for Ask (stdlib only)
tk_app.py     CustomTkinter GUI driving the same functions (no subprocess)
sample/       one file per format, for smoke tests
index/        LanceDB storage, created at runtime
.venv/        virtualenv (create it yourself; not checked in)
```

## Desktop app (CustomTkinter)

Rather than the CLI you can drive the whole pipeline from a window. The GUI
**imports and calls `pipeline.py` directly** — no subprocess, no server — so
CLI and GUI share exactly one code path and there is nothing to keep in sync.

```powershell
.\.venv\Scripts\python.exe tk_app.py
```

What it gives you over the CLI:

- **Multiple folders at once.** Each folder is a row with its own label and
  extension filter. Files are stored as `label/relative/path`, so
  `docs/readme.md` and `code/readme.md` never collide in one index.
- **Filtered scope.** `Label | folder | extensions` per folder; blank means all.
- **Live progress.** The progress bar and log stream from the worker thread,
  so the window never freezes and you can watch files/chunks go by.
- **Ollama status.** A dot in the status bar shows whether the daemon is
  reachable and populates the model dropdown from the *embedding* models you
  actually have installed.
- **Guards kept.** Re-index is explicit, and the model-mismatch check still
  refuses to search an index built with a different model.

Steps: add folders → set table name → **1 · Ingest** → **2 · Embed** (enabled
once ingest succeeds) → **Search**.

Both entry points share `index/`, so you can start something in the GUI and
finish it from the CLI.

## User documentation (MkDocs, flat HTML)

End-user manual lives in `docs/` and builds to plain `.html` files
(`use_directory_urls: false` — `gui.html`, not `gui/index.html`):

```powershell
.\.venv\Scripts\pip.exe install -r requirements-docs.txt
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

Open `site/index.html` straight from disk. Pages: Install, Desktop App, CLI
Commands, How It Works, Schema & Config, Troubleshooting.

## Notes

- **Reranker is deferred** (v2). A separate rerank model
  (`bge-reranker-v2-m3`) will not fit alongside the embedder in 4 GB VRAM.
  Hybrid + RRF is the mitigation for now.
- **Storage**: 1024-dim float32 = 4 bytes/value. int8 quantization cuts this
  4×; MRL lets you store 256-dim truncations for a further 4× if recall holds.
  Check `stats` for your real numbers before optimizing.
