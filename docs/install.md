# Install

Everything runs locally on Windows: Python for the pipeline, Ollama for
embeddings, LanceDB for storage (a folder, not a server).

## Requirements

| Need | Minimum | Notes |
|------|---------|-------|
| Windows | 10 / 11 | PowerShell commands below |
| Python | 3.11+ | 3.14 works; venv recommended |
| RAM | 8 GB | 16 GB comfortable for large corpora |
| VRAM | none required | GTX 1050 Ti 4 GB is the tuning target; CPU works too, slower |
| Disk | 2 GB free | Model (639 MB) + venv (~1 GB) + index |

## 1. Python environment

```powershell
cd embed_process
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

`requirements.txt` installs the pipeline (`ollama`, `lancedb`, `pyarrow`,
`pylance`, `pypdf`, `beautifulsoup4`, `python-docx`) and the desktop app
(`customtkinter`).

## 2. Ollama + model

[Install Ollama for Windows](https://ollama.com/download), then:

```powershell
ollama pull qwen3-embedding:0.6b
```

To keep the model on a small GPU, start Ollama with the KV cache quantized
(halves VRAM). This is set on the **server**, not per request:

```powershell
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"
ollama serve
```

!!! note "Verify GPU offload"
    Check `ollama ps` and the server log for `offloaded N/M layers`. Only the
    log confirms the model actually landed on the GPU.

If Ollama listens somewhere else, point the pipeline at it:

```powershell
$env:OLLAMA_HOST = "http://localhost:11434"
```

## 3. Smoke test

A tiny corpus ships in `sample/` (one file per format):

```powershell
.\.venv\Scripts\python.exe pipeline.py ingest --src sample --table sample --dry-run
.\.venv\Scripts\python.exe pipeline.py ingest --src sample --table sample --replace
.\.venv\Scripts\python.exe pipeline.py embed  --table sample
.\.venv\Scripts\python.exe pipeline.py query  "cosine similarity function" --table sample
```

Or open the [desktop app](gui.md):

```powershell
.\.venv\Scripts\python.exe tk_app.py
```

## 4. Building this documentation (optional)

These pages are built with MkDocs into flat `.html` files:

```powershell
.\.venv\Scripts\pip.exe install -r requirements-docs.txt
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

Open `site/index.html` straight from disk — no server needed. `use_directory_urls`
is `false`, so every page is a plain file (`gui.html`, not `gui/index.html`).
