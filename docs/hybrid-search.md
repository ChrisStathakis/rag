# Hybrid Search

Every query in this project can run in one of two modes:

| Mode | Flag (CLI) | App | What happens |
|------|-----------|-----|--------------|
| **Hybrid** (default) | `query "..."` / `--hybrid` | **Hybrid** checkbox on | Dense vector match **+** BM25 keyword match, fused with RRF |
| **Vector-only** | `--no-hybrid` | **Hybrid** checkbox off | Dense vector match only |

## Where to toggle it

| Surface | Control | Default | Persisted |
|---------|---------|---------|-----------|
| CLI `query` | `--hybrid` / `--no-hybrid` | hybrid on | n/a (per run) |
| CLI `eval` | `--hybrid` (opt-in) | vector-only, so existing numbers don't shift | n/a (per run) |
| App 3 · Chat | **Hybrid** checkbox next to **Send** | on | no — reset from Settings on next launch |
| App 4 · Settings | **Hybrid search (vector + keyword…)** + **Save settings** | on (`hybrid_search=1` in `app.db`) | yes, also synced to the Chat checkbox |

Programmatic use: `tk_app._retrieve(q, topk, table, model, hybrid=...)`
(and `_search` / `_chat`, all default `hybrid=True`); `pipeline.cmd_query`
reads `args.hybrid` from the same flags.

```powershell
.\.venv\Scripts\python.exe pipeline.py query "how does attention work" --table sample
.\.venv\Scripts\python.exe pipeline.py query "how does attention work" --table sample --no-hybrid
```

## When to use which

- **Keep hybrid on** for exact terms: error codes, function names, file names,
  quoted phrases, IDs. BM25 catches the literal token when the embedding
  paraphrases it away.
- **Switch to vector-only** when the question is conceptual ("explain X simply",
  translated queries, synonyms). If hybrid returns keyword-stuffed but
  off-topic chunks, compare with `--no-hybrid` — the difference tells you
  which half misled.
- **Chat (Ask)** uses the same toggle: the Hybrid checkbox next to **Send**
  controls the silent corpus lookup. The default follows
  **4 · Settings → Hybrid search** and is saved with **Save settings**.

## How it works

1. **Vector half.** The query is embedded with the table's model
   (`QUERY_INSTRUCT` prefix + cosine metric, `k × 3` candidates) and re-ranked
   by per-folder weights (`score = cos × weight`, see below).
2. **Keyword half.** A full-text (BM25) search over `text_raw`
   (`k × 3` candidates). Built by `embed` via `_ensure_fts_index`; if the
   index is missing (older table), this half returns nothing and the query
   gracefully falls back to vector-only.
3. **Fusion.** Both ranked lists are combined with **Reciprocal Rank Fusion**
   (RRF, `rr = 60`): `score(id) = Σ 1 / (60 + rank)`. A chunk in both lists
   outranks one seen once. Fusion unions fields: vector `_distance` is
   preserved through fusion, so the displayed `cos=` stays meaningful;
   FTS-only fields (`_score`) are kept alongside; fused rows also carry `_rrf`.
   Each half contributes up to `k × 3` candidates; the fused list is truncated
   to `k`.

The query log shows the mode: `mode=hybrid` / `mode=vector` (CLI) and
`mode=...` in the app's query summary.

## Chat interaction

Chat retrieval calls the same `_retrieve(..., hybrid)` path, then keeps at
most 4 chunks with cosine `>=` **Min similarity** (`4 · Settings →
rag_min_cos`, default `0.30`). Two nuances:

- FTS-only hits carry no `_distance`, so they have no cosine: they are kept
  but ranked last, and survive only when at least one scored hit also passes
  the threshold (otherwise the turn falls back to a general answer).
- Greetings, RAG-off, unembedded tables, and below-threshold turns all answer
  without corpus context — the user never sees a "no hits" error. The status
  bar shows `grounded in N chunk(s) ✓` when the corpus contributed.

## Fallback matrix

| Situation | Result |
|-----------|--------|
| `hybrid=False` (`--no-hybrid`, checkbox off) | FTS never called; pure cosine top-`k`. |
| FTS index missing (older table, build failed) | Keyword half returns `[]`; query degrades to vector-only. |
| FTS returns `[]` but hybrid on | CLI still fuses (no-op); app takes vector top-`k` directly. Same ordering either way. |

Restore full hybrid by rerunning `embed` on that table — it rebuilds the FTS
index at the end (`_ensure_fts_index`, `replace=True`). The FTS index is
never rebuilt per query (too expensive).

## Folder weights

Weights (`ingest --weight 2.0`, or the Weight dropdown) multiply the cosine
score **before** fusion, so a boosted folder wins in both halves' ordering.
Changing a weight needs no re-ingest; boosted hits show `w=…`.

## Evaluating hybrid vs vector

`eval` defaults to vector-only so existing numbers don't shift. Opt in per run:

```powershell
.\.venv\Scripts\python.exe pipeline.py eval --queries sample_queries.txt --gold sample_gold.csv --tables sample
.\.venv\Scripts\python.exe pipeline.py eval --queries sample_queries.txt --gold sample_gold.csv --tables sample --hybrid
```

Compare the two `nDCG@10` columns: if `--hybrid` scores clearly higher, your
queries are keyword-heavy and hybrid should stay on.

## Troubleshooting

- **Hybrid looks identical to vector-only.** The FTS index is probably missing
  (table built before FTS, or index build failed). Rerun `embed` — it rebuilds
  the index — and watch for `FTS index not built (...)` in the log.
- **Keyword junk outranks good semantic hits.** Lower the influence of BM25 by
  searching vector-only, or raise the folder weight of the authoritative source.
- **Model mismatch errors.** Unrelated to hybrid: vectors from different models
  are never comparable. Use `reindex` to switch models.

Related: [CLI Commands](commands.md) · [Desktop App](gui.md) ·
[Troubleshooting](troubleshooting.md).
