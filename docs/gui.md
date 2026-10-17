# Desktop App

`tk_app.py` is a CustomTkinter window over the same pipeline. It calls the
pipeline **directly** — no subprocess, no server — so the CLI and GUI share a
single code path.

```powershell
.\.venv\Scripts\python.exe tk_app.py
```

!!! note "Install first"
    `customtkinter` is in `requirements.txt`. The GUI and CLI can use the same
    `.venv`.

The window has four tabs: **1 · Sources** (folders + staging),
**2 · Index** (ingest/embed, log), **3 · Chat** (search, answers, history),
and **4 · Settings** (see below).

## What it gives you over the CLI

The app exists for three things the CLI does badly: **several folders at
once**, **per-folder filtering**, and **live progress**.

### Multiple folders, one index

Each folder is a row with its own **label** and **extension filter**. Files are
stored as `label/relative/path`, so two folders that both contain
`readme.md` never collide in one index.

| Column | Meaning |
|--------|---------|
| Label | Defaults to the folder name. Becomes the path prefix. |
| Folder | Absolute path. Click **Add folder** to pick one (or edit in place). |
| Extensions | Comma list, e.g. `.md,.pdf,.py`. **Blank means all formats.** |
| Weight | Importance multiplier for this folder (see below). Default `1.0`. |
| Browse | Re-point this row at a different folder (no-op on cancel). |
| ✕ | Removes the row. |

### Folder importance (weights)

Some folders matter more than others. Each row has a **Weight** dropdown
(`0.5 / 1.0 / 1.5 / 2.0`):

- Weights are saved per table in `index/weights.json` at ingest time — no
  database migration, old tables keep working (missing = `1.0`).
- At query time each hit's cosine is multiplied by its folder's weight
  (`score = cos × weight`), then re-ranked before fusion with BM25.
- Changing a weight does **not** need a re-ingest: values are re-saved live
  whenever you run a query, and hits show `w=1.5` so you can see the boost.
- `reindex` copies the weights to the new table.

The folder label is the first segment of the stored path (`docs/readme.md`),
so weights follow labels, not absolute paths.

### Collecting sources: folders, notes, links

Tab **1 · Sources** has the folder rows plus a **Staging** area
(`staging/p<id>/`, one per project) for material you create or fetch:

- **+ Note** — paste text + a title → saved as `.md`.
- **+ Link** — give a page URL → downloaded and converted with the same
  extractors ingest uses (best with articles/docs pages) → `.md`.
- Staging files list with per-file delete; ingest automatically includes the
  staging folder (label `staging`, weight `1.0`) alongside your folder rows.
- Books (PDF/DOCX) go in through folders, as before.

### Filtered scope

Same knobs as the CLI's `--only`, `--size`, `--overlap`, plus `--replace` as a
checkbox. Set chunk size/overlap before hitting Ingest.

### Live progress

The progress bar and log stream from the worker thread, so the window never
freezes. The button label shows `title done/total` while running.

### Ollama status

The dot in the status bar shows whether the Ollama daemon is reachable, and the
model dropdown is populated with only the **embedding** models you actually
have installed — not every model in your library.

## Pipeline flow

1. Pick a RAG project (header dropdown) → add folders/notes/links in
   **1 · Sources**
2. **2 · Index**: set **Table** name → **1 · Ingest** → **2 · Embed**
3. **3 · Chat**: talk to your assistant (RAG in the background)
4. **4 · Settings**: hosts, keys, models, backup (below)

## Projects — one RAG each (SQLite)

Each RAG is a project: folders + extensions + weights, table, models, chunk
settings. Projects live in **`app.db`** (SQLite, stdlib — no new
dependency) and autosave on every switch and on close. No Save button, no
lost setups.

- Header: `[RAG: dropdown]` to switch, `+ New` to create (table name defaults
  to a slug of the project name), `Rename` (project only — the index table
  keeps its name), `Delete` (asks whether to also drop the index table).
- Old `projects/*.json` files are imported once automatically.
- Move everything to another machine with **4 · Settings → Backup / Transfer →
  Full bundle**: one `.zip` holding `app.db` + `index/` + `staging/`. Restore it
  on the other machine (or back here) in one step.

## Chat (tab 3)

The **Chat** tab holds the question bar, the model row, and the conversation
history (newest exchange on top, **Clear** wipes it).

The tab is a plain **conversation** (like any chat app): type a message,
press Enter or **Send**. Your messages appear as bubbles on the right,
assistant replies as formatted markdown on the left, oldest at top with
auto-scroll. **New chat** starts fresh; **Copy answer** copies the last
reply.

**RAG is a silent helper.** Before each reply the app quietly retrieves
relevant chunks from the active project's index and injects them as
background context. When nothing relevant is found — or no index exists —
the assistant just answers normally. No source cards, no citations, no
errors about hits. The status bar shows `grounded in N chunk(s) ✓` when
the corpus contributed.

- **Provider + model** pick who you talk to: **Ollama** (local dropdown),
  **Claude** / **ChatGPT** (free-text model name + API key from Settings).
- Conversation history (last 3 exchanges) is always sent, so follow-ups
  like "explain more simply" resolve naturally.
- **Query** (tab 2 · Index) still exists for raw search: hits render as
  one compact list message in the conversation.
- **Hybrid toggle.** The **Hybrid** checkbox next to **Send** switches between
  hybrid (vector + keyword, default) and vector-only retrieval for both Query
  and Chat. The default follows **4 · Settings → Hybrid search**. Details and
  trade-offs: [Hybrid Search](hybrid-search.md).
- **Saved queries.** Under the message bar, **Save current** keeps the text of
  the box as a per-project saved query, and the **Saved:** dropdown lists them.
  Pick one to drop it back into the box; the ✕ deletes the selected one. They
  are stored in `app.db` (`saved_queries` table) and follow the project, so each
  RAG keeps its own shortlist. Saving the same text twice is a no-op. Same list
  from the CLI: `pipeline.py saved list|add|rm --project <name|id>`.
- **OpenCode** / **CmdCode** (tab 2 · Index) open a terminal with the
  agent in your first source folder and copy the question + sources.
  Require the CLIs on PATH (`opencode`, or `command-code` via
  `npm i -g command-code`).

## Settings (tab 4)

- **Ollama**: host override (blank = default) + Test button. Applied live;
  also read from the `OLLAMA_HOST` env var at startup.
- **Defaults for new projects**: embed/chat model, provider, chunk size,
  overlap, top-k. Saved with **Save settings** (numbers validated).
- **API keys**: Anthropic (Claude) and OpenAI (ChatGPT), masked with
  show/hide + clear. Stored in `app.db` as plaintext — fine for a local
  single-user machine, do not share the file.
- **Backup / Transfer**:
  - **Full bundle (.zip)** — `app.db` + `index/` (all vector tables +
    `weights.json`) + `staging/`, with a `manifest.json` (version, counts,
    per-file SHA-256). This is the whole installation in one file. API keys
    are **excluded by default**; tick **include API keys** to keep them.
    **Restore** verifies every checksum, takes a **snapshot** of your current
    data into `backups/auto-<timestamp>/`, then replaces everything.
  - **Settings only (.json)** — the lightweight old behaviour: settings +
    projects + folder weights, no vectors. Restore replaces settings and
    projects (index tables are kept) after confirmation.
  - **This project only (.zip)** — **Export project…** writes just the current
    project (its `index/<table>.lance`, its `weights.json` entry and its
    `staging/p<id>/`) to a `.zip`. **Import project…** **adds** it to the
    installation; nothing is overwritten — a clashing table name becomes
    `<table>_imported` and a clashing project name gets ` (imported)`, and a
    snapshot is taken to `backups/` first. Use this to move a single RAG or to
    share one index without shipping your whole `app.db`.
  - **Open data folder** opens the directory holding `index/`, `app.db`,
    `staging/` and `backups/`.
- **Ollama models**: local list with sizes + Refresh; **Pull** streams
  progress to the log; **Delete** asks first (tables built with a deleted
  model stop working until you pull it again).

If the table already exists and **Replace** is off, you get a choice:
**Append** new files (already indexed are skipped), **Rebuild** from scratch,
or Cancel. `embed` afterwards only processes the new rows.

**Query**, **Stats**, **Re-index**, and **Export MD** are also in the toolbar.
Re-index always asks for confirmation and opens the Embed step for the new
table afterwards.

### Sync — one-click incremental update

**Sync** is the "catch up" button: re-run ingest in append mode (files whose
`source` is already in the table are skipped, so only new or renamed files are
chunked) and, if anything new landed, embed those rows in the same step. Run it
after adding files to a folder you already indexed — no need to remember the
Ingest → Embed order. It is safe to run repeatedly; when there is nothing new
it reports `0 new chunk(s)` and stops.

### Sources — delete a file or folder from the index

**Sources…** opens a list of every `source` in the current table with its chunk
count. The ✕ on a row deletes every chunk for that source and rebuilds the
keyword (FTS) index so hybrid search stops returning it. Deleting works at two
levels because a source is a path:

- deleting `docs/readme.md` removes exactly that file's chunks;
- deleting a label/folder prefix like `docs` removes every file under it.

Same thing from the CLI:
`pipeline.py delete --table corpus --source docs/readme.md`
(use the folder form `--source docs` to drop a whole folder).

### Stop a running job

**Stop** cancels the current step at the next safe boundary (between files while
ingest/export scans, and between embed batches). It is disabled when idle and
turns on only while a job runs. Work already committed is kept:

- `embed` flushes the rows it has already embedded before stopping, so a rerun
  continues where it left off;
- `ingest`/`export-md` abort before writing, so the new table is either fully
  written or untouched.

## Export folder to Markdown

**Export MD** converts a whole folder into clean `.md` sources for RAG, using
the same extractors as ingest — what you see is what would be embedded:

1. Pick the input folder, then an output folder (**outside** the input, or the
   next run would pick up the generated files again).
2. Every supported file becomes one `.md` (title as `#`, sections as `##`,
   code in fenced blocks), mirroring the source tree.
3. Review/edit the `.md` files, then ingest the output folder normally.

Same thing from the CLI: `pipeline.py export-md --src <in> --dest <out>`.

## Guards kept

The app keeps the same protections as the CLI:

- **Model mismatch is refused.** Searching a 0.6 b index with the 4 b model
  errors out instead of returning nonsense. Use **Re-index** to switch models.
- **Buttons lock while a step runs**, so you cannot start a second job on top
  of the first.
- **Stop is cooperative.** A running job checks for a stop between units and
  keeps whatever it already committed, so you can cancel a long embed and rerun
  it later without losing work.
- **One UI pump, never dies.** Every message from the worker is handled in a
  guard block — a single bad message cannot freeze the window.

## Why no Electron

Electron was considered and dropped: it would have meant rewriting the working
extract/chunk/LanceDB logic in JavaScript and bundling ~150-250 MB of
`node_modules` for what is, functionally, a folder picker and a progress bar.

Calling Python in-process gets the same UI with ~0 extra dependencies and
reuses tested code — including every bug fixed along the way.

## Shared with the CLI

Both entry points write to the same `index/` directory. You can start a job in
the GUI, close it, and finish from the CLI (or vice versa) — the `embedded`
column is what makes that resumable.

## Where is my data?

- Index storage: `embed_process/index/` (LanceDB files, created at runtime).
- Project settings: `embed_process/app.db` (SQLite, single file).
- Collected notes/links: `embed_process/staging/p<id>/` (plain `.md`).
- Auto-snapshots before a restore: `embed_process/backups/auto-<timestamp>/`
  (the newest 5 are kept).
- Full-bundle backup/restore lives in `backup.py` (stdlib only).
- Table name: whatever you type in **Table**. One table per corpus; `stats`
  lists them all.
- Tick **Replace table** only when you want to rebuild from scratch — it drops
  the existing table including its vectors.

Stuck? See [Troubleshooting](troubleshooting.md).
