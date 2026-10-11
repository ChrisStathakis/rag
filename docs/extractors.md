# Extractors & Chunking

Every format is parsed into a common `Block(text, section, meta)`, so the rest
of the pipeline is format-agnostic. There are five extractors and two chunkers.

## The five extractors

| Format | File | Approach |
|--------|------|----------|
| Text / Markdown | `md txt markdown rst` | Direct read. Markdown headings become sections; fenced code blocks are split out. |
| HTML | `html htm xhtml` | BeautifulSoup. Strips `nav`, `footer`, `header`, `script`, `style`, `aside`, `form`, `iframe`. Walks headings + content in document order so each paragraph keeps its section. |
| DOCX | `docx` | python-docx. `Heading N` styles become sections; tables are flattened to pipe-separated rows. |
| PDF | `pdf` | pypdf. One block per page; repeated header/footer lines (short lines on >50% of pages) are removed. |
| Code | `py js ts go rs java c cpp rb sh sql ...` | A lightweight parser splits per function/class across ~30 languages. |

!!! note "Unknown extensions are skipped, never raised"
    A file with an unsupported extension is logged and skipped. One bad file
    cannot abort an ingest run.

## Two chunkers

### Prose (recursive)

Used for everything that is not code.

- Splits into sentence-ish units, then packs them up to **~900 tokens** with
  **~120 tokens of overlap**.
- Never cuts mid-word or mid-sentence.
- A trailing fragment below **50 tokens** is dropped, but the **first** chunk
  of every block is always kept — a short but meaningful paragraph still belongs
  in the index.

### Code (structural)

Code blocks already arrive split per function/class from the extractor.

- Target **~600 tokens** per piece.
- Imports and top-of-file comments are kept as a **header on every piece**, so
  the retriever sees dependencies.
- Oversized functions are cut on **line boundaries** with a one-line overlap —
  never mid-line.

## `embed_input` vs `text_raw`

Each chunk stores two views of the same text:

| Field | Purpose |
|-------|---------|
| `embed_input` | What the model sees — a `Document: {title} > {section}` prefix plus the text. |
| `text_raw` | What the user sees — the clean source text, no prefix. |

Qwen3 reports +1–5% retrieval quality when the input carries this kind of
contextual prefix, so the model gets the richer version while search results
display the clean one.

!!! note "Ollama applies no prompt template to `/api/embed`"
    Any instruction or prefix has to be passed in explicitly. The query side
    adds `"Given a web search query, retrieve relevant passages that answer the
    query."` — always in English, the language the model was trained on.

## Per-format metadata

Each block carries a `meta` dict that records where it came from, so a search
hit can always be traced back to its file and location:

| Extractor | Metadata |
|-----------|----------|
| Code | `kind`, `symbol` (function/class name), `source_rel` |
| PDF | `page` number |
| All | `kind` (prose / html / docx / pdf / code), `source_rel` (relative path) |
