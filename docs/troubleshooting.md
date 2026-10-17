# Troubleshooting

Common failures, in the order you are likely to meet them.

## Ollama not reachable

```
could not probe embedding dimension from Ollama (...)
```

or the app's status dot is red.

- Start the daemon: `ollama serve`.
- Check the address: `http://localhost:11434/api/tags` should answer in a
  browser. If Ollama runs elsewhere, set `$env:OLLAMA_HOST` first.
- Ingest needs Ollama only for the one-call dimension probe; you can skip it
  with `--dims 1024` (for `qwen3-embedding:0.6b`).

## Model missing

`embed` auto-pulls a missing model (`client.pull`). If the pull fails, check
disk space and run `ollama pull qwen3-embedding:0.6b` by hand.

## Out of memory on the GPU

The batch starts at 8 and halves automatically to a floor of 2. If it still
fails:

1. `ollama ps` — unload models you are not using.
2. Restart Ollama with `$env:OLLAMA_KV_CACHE_TYPE = "q8_0"`.
3. As a last resort, lower `NUM_CTX` in `pipeline.py` (default 8192).

Interrupted runs are resumable: rerun `embed` and it continues from the rows
still marked `embedded=0`. If a run dies mid-batch, already-embedded rows are
merged before the error is raised.

## Model mismatch refused

```
table 'x' was built with qwen3-embedding:0.6b, but you asked for ...
Vectors from different models are not comparable -- this is refused on purpose.
```

This is a guard, not a bug. Vectors from different models live in different
spaces. Use `reindex` to build a clean table for the new model, then `embed`
it. The old table is untouched.

## Query dimension mismatch

```
query vector dim ... != index dim ...
```

The model that answers the query must be the model that built the index.
Re-check the `--model` flag or the app's model dropdown.

## Hybrid falls back to vector-only

If the FTS index was never built (e.g. a table from an older version), search
still works — the BM25 half is skipped and only dense results are fused. To
restore full hybrid search, rerun `embed` on that table (it rebuilds the FTS
index at the end).

## Empty results / missing files

- `no supported files under ...` — the folder has no supported extensions, or
  everything is hidden/empty. Check [supported formats](extractors.md) and the
  per-folder extension filter (blank means all).
- `no chunks produced` — files were found but filtered out or too small.
- `skipped: N` in the ingest log lists files that failed to parse. One bad
  file never aborts a run.

## Garbled log output when redirecting

`python pipeline.py ... > out.txt` on Windows gives stdout a cp1252 codec.
The pipeline degrades unprintable characters to `?` instead of crashing, so
redirects are safe.

## Table already exists

`ingest` refuses to overwrite. Pass `--replace`, `--append`, pick another
`--table`, or tick **Replace table** in the app (which asks
Append / Rebuild / Cancel when the table exists).

## Weights do not seem to apply

- Weights live in `index/weights.json`, keyed by table then folder label. If
  you renamed the table or the folder (hence its label), the entry no longer
  matches and everything behaves as `1.0` — re-run ingest or a query from the
  app to re-save them.
- Boosted hits print `w=…` next to `cos=`. No marker means weight `1.0` (or an
  FTS-only hit, which carries no cosine to scale).

## Chat has no model

The provider selector offers Ollama (local dropdown) plus hosted providers:
Anthropic (Claude), OpenAI (ChatGPT), OpenCode, The Grid, and **custom** (any
OpenAI-compatible base URL). Hosted providers use a free-text model name + a
key from **4 · Settings**. For Ollama with `(no chat model)` showing, pull a
small one: `ollama pull qwen3:0.6b` (fits next to the embedder on 4 GB VRAM).
Sending refuses empty models/keys (and an empty Custom base URL) with a message
instead of hanging.

## Chat errors appear inline

Failures (bad key, network, empty answer) show as a small ⚠ line at the
bottom of the conversation — the chat keeps working. Note: RAG problems
are never fatal — when retrieval fails or finds nothing, the assistant
simply answers without corpus context.

## API chat errors

First stop: the **Test** button next to each key in **4 · Settings** fires one
tiny live request and prints the result inline (green = worked, red = the
provider's message). It tests the key, the model and — for `custom` — the base
URL together, so a red line tells you which of the three is wrong.

- `HTTP 401` / `403` — wrong or revoked key. Re-paste it in Settings → Save.
- `HTTP 402` — out of credits/balance (The Grid returns this). Top up the
  account, then resend.
- `HTTP 404` — the model/endpoint was not found. The Grid wants an **instrument**
  string (`text-prime`, `code-max`), not an upstream name; OpenCode needs a
  model served on its OpenAI/Chat-Completions surface (some are
  responses/Anthropic/Google-only and won't work here).
- `307`/redirects are followed automatically (The Grid uses one); you do not
  need to change the base URL — do not append `/chat/completions` yourself.
- `request failed` / timeouts — no network or provider outage; retry later.
  The conversation history is kept, so you can just resend.
- Hosted models bill per token; every message re-sends the last 3
  exchanges plus retrieved context, so keep top-k small (4–6).

## "opencode not found"

**OpenCode** / **CmdCode** (tab 2 · Index) need their CLIs on PATH
(`opencode`, or `command-code` via `npm i -g command-code` plus sign-in).
Install, restart the app so PATH refreshes, and try again. Without fresh
search results for the exact question, the app asks before opening
context-free.
