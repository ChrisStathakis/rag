# embed_process — Local RAG Embedding Pipeline

> A local retrieval pipeline that turns a folder of documents into a searchable
> index. Runs entirely on your machine: **Ollama** for inference and
> **LanceDB** for storage + hybrid search.

Built for a **GTX 1050 Ti (4 GB VRAM) / 16 GB RAM**, English-only corpus.

## Overview

| Piece | Choice | Why |
|-------|--------|-----|
| Inference | [Ollama](https://ollama.com) | Local, no API key, GPU offload |
| Embedding model | `qwen3-embedding:0.6b` | 639 MB, 32 K context, 1024-dim. MTEB(eng,v2) 70.70, MTEB-Code 75.41 |
| Storage | [LanceDB](https://lancedb.com) | Zero-copy, hybrid vector + FTS, no server |
| Search | hybrid (vector + BM25 fused with RRF) | Dense recall plus exact keyword match |

## Install

```powershell
cd embed_process
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
ollama pull qwen3-embedding:0.6b
```

Full steps — Ollama setup, GPU offload check, smoke test — are on the
[Install](install.md) page.

## Quick start

A tiny sample corpus ships in `sample/` (one file per format) so you can smoke
test before pointing the pipeline at real documents:

```powershell
# 1. preview what would be produced (writes nothing)
.\.venv\Scripts\python.exe pipeline.py ingest --src sample --table sample --dry-run

# 2. extract + chunk + write text + metadata (no vectors yet)
.\.venv\Scripts\python.exe pipeline.py ingest --src sample --table sample --replace

# 3. embed the pending rows into vectors (batched, resumable)
.\.venv\Scripts\python.exe pipeline.py embed --table sample

# 4. search (hybrid = vector + BM25 fused with RRF)
.\.venv\Scripts\python.exe pipeline.py query "cosine similarity function" --table sample
```

Then point it at your own corpus:

```powershell
.\.venv\Scripts\python.exe pipeline.py ingest --src <your_folder> --table corpus
.\.venv\Scripts\python.exe pipeline.py embed  --table corpus
.\.venv\Scripts\python.exe pipeline.py query  "your question" --table corpus
```

!!! tip "Several folders in one index"
    The CLI takes a single `--src`, but `pipeline.ingest_sources()` accepts a
    list of `SourceSpec(path, only, label)` rows and merges them into one table
    — that is what the [desktop app](gui.md) does with its folder rows.

## Next steps

- [Install](install.md) — requirements, Ollama, smoke test
- [Desktop App](gui.md) — the CustomTkinter window and multi-folder workflow
- [Commands](commands.md) — every subcommand and its flags
- [Hybrid Search](hybrid-search.md) — vector + BM25 fusion, when to use each mode
- [Extractors & Chunking](extractors.md) — how each format is parsed and split
- [Schema & Config](schema.md) — the LanceDB columns, guards, and hardware settings
- [Troubleshooting](troubleshooting.md) — common failures and what to do
