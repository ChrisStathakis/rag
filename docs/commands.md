# Commands

`pipeline.py` exposes six subcommands. The normal flow is
`ingest` → `embed` → `query`, with `stats` / `eval` / `reindex` for
inspection and comparison.

```powershell
.\.venv\Scripts\python.exe pipeline.py <command> [options]
```

## ingest

Walk a folder, extract text, chunk it, and store text + metadata. **No vectors
yet** — that is `embed`'s job. Splitting the two means you can re-run `embed`
without re-parsing the corpus.

| Flag | Meaning |
|------|---------|
| `--src` | Folder to scan (required). Recurses; skips hidden/empty/unknown files. |
| `--table` | Target table name (default `chunks`). |
| `--model` | Used only to probe vector dimensions. |
| `--dims` | Skip the Ollama probe and use this dimension. |
| `--size` | Chunk size in tokens (default 900). |
| `--overlap` | Overlap tokens (default 120). |
| `--only` | Comma list of extensions, e.g. `.md,.pdf,.py`. |
| `--weight` | Importance weight for this source (default `1.0`). See weights below. |
| `--replace` | Drop the table if it already exists. |
| `--append` | Add to the existing table instead of failing. Files whose `source` is already indexed are skipped; model/dims mismatches are refused. |
| `--dry-run` | Show a sample chunk, write nothing. Keeps any existing table untouched. |
| `-v` | Verbose (per-file block counts). |

```powershell
# preview, then build
.\.venv\Scripts\python.exe pipeline.py ingest --src sample --table sample --dry-run
.\.venv\Scripts\python.exe pipeline.py ingest --src sample --table sample --replace
# later: add more without rebuilding
.\.venv\Scripts\python.exe pipeline.py ingest --src new_docs --table sample --append
.\.venv\Scripts\python.exe pipeline.py embed --table sample   # only new rows
```

!!! tip "Measure before you commit"
    `--dry-run` prints block counts and a sample chunk without writing
    anything. Use it to sanity-check chunk size and volume first.

## embed

Batch-embed every row where `embedded=0`. Auto-pulls the model if missing.
**Resumable** — rerun it after an interruption and it continues from where it
left off (already-embedded rows are skipped).

| Flag | Meaning |
|------|---------|
| `--model` | Embedding model (default `qwen3-embedding:0.6b`). |
| `--table` | Table to embed. Defaults to the first/only table. |

```powershell
.\.venv\Scripts\python.exe pipeline.py embed --table sample
```

!!! note "Batch size halves on out-of-memory"
    If the GPU runs out mid-run, the batch size is cut and retried down to a
    floor of 2. If it still fails at batch 2, lower `NUM_CTX` or unload other
    models with `ollama ps`.

## query

Hybrid search: dense vector match fused with BM25 keyword match via Reciprocal
Rank Fusion (RRF).

| Flag | Meaning |
|------|---------|
| `text` | The query (required, positional). |
| `--table` | Table to search. |
| `--model` | Must match the table's `model_tag` — refused otherwise. |
| `--top-k` | Number of hits to return (default 8). |
| `--width` | Characters of each hit to print (default 220). |
| `--no-hybrid` | Vector-only; skip the FTS/RRF half. |

```powershell
.\.venv\Scripts\python.exe pipeline.py query "how does attention work" --table sample
```

!!! danger "Model mismatch is refused on purpose"
    Vectors from different models are not comparable. If you pass a `--model`
    that differs from the index's `model_tag`, `query` errors out instead of
    returning silently wrong results. To switch models, use `reindex`.

## stats

Index statistics: row counts by kind / model / dims, token totals, on-disk
size. Shows every table in the index.

| Flag | Meaning |
|------|---------|
| `-v` | Per-file chunk counts. |

```powershell
.\.venv\Scripts\python.exe pipeline.py stats -v
```

## eval

Run a query set over one or more tables and compare them. This is how you
decide whether a model or chunking change actually helped.

| Flag | Meaning |
|------|---------|
| `--queries` | txt/csv, one query per line (or column 0). Required. |
| `--gold` | CSV `query,source-glob` — enables hit-rate and nDCG scoring. |
| `--tables` | Comma list of tables (default: all). |
| `--out` | Write results to a CSV. |

```powershell
.\.venv\Scripts\python.exe pipeline.py eval --queries sample_queries.txt --tables sample
# with scoring
.\.venv\Scripts\python.exe pipeline.py eval --queries sample_queries.txt --gold sample_gold.csv --tables sample
```

!!! warning "Without `--gold`, nothing is scored"
    A bare `eval` only smoke-tests that the queries run. nDCG and hit-rate
    stay `n/a` until you supply a gold set.

### Model A/B example

Comparing the two Qwen3 embedding sizes on the sample corpus:

| Table | Model | nDCG@10 | HitRate |
|-------|-------|---------|---------|
| `sample_4b` | `qwen3-embedding:4b` | 0.9461 | 1.000 |
| `sample` | `qwen3-embedding:0.6b` | 0.9396 | 1.000 |

The 4 b model is ~0.7 % better but **~8× slower** (10.0 s vs 1.2 s for the same
rows) and uses 2560-dim vectors instead of 1024. On a 4 GB card the 0.6 b model
is the right trade-off.

## reindex

Build a fresh table for a **different** model. Copies text + metadata, drops
the vectors, and re-probes the new model's dimensions.

| Flag | Meaning |
|------|---------|
| `--model` | The new embedding model (default `qwen3-embedding:0.6b`). |
| `--table` | Source table. |
| `--dest` | Destination table name (default `<src>__<model>`). |
| `--dims` | Skip the Ollama probe, use this dimension. |
| `--replace` | Overwrite the destination if it exists. |

```powershell
.\.venv\Scripts\python.exe pipeline.py reindex --table sample --model qwen3-embedding:4b --dest sample_4b --replace
.\.venv\Scripts\python.exe pipeline.py embed   --table sample_4b --model qwen3-embedding:4b
```

!!! note "Changing the model means a full re-embed"
    You cannot mix vector spaces. `reindex` gives you a clean table sized for
    the new model, then `embed` fills it. The original table is untouched.
    Folder weights are copied to the new table automatically.

## Folder weights

`ingest --weight 2.0` (or the Weight dropdown in the app) stores a per-folder
multiplier in `index/weights.json`. At query time each hit scores
`cos × weight` before fusion, and boosted hits show `w=…` in the output.
Weights apply live — no re-ingest needed to change them.

## export-md

Convert a folder into `.md` sources with the same extractors ingest uses:

| Flag | Meaning |
|------|---------|
| `--src` | Folder to convert (required). |
| `--dest` | Output folder (required, outside `--src`). |
| `--only` | Comma list of extensions, e.g. `.docx,.pdf,.html`. |

```powershell
.\.venv\Scripts\python.exe pipeline.py export-md --src raw_docs --dest md_docs
.\.venv\Scripts\python.exe pipeline.py ingest --src md_docs --table corpus --replace
```
