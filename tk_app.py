"""tk_app.py -- CustomTkinter GUI for the embed_process pipeline.

A thin desktop front-end over pipeline.py: pick folders, filter them, run the
ingest/embed/search steps. It calls the pipeline functions **directly** (no
subprocess, no server) so there is one code path for CLI and GUI.

    python tk_app.py

Requires: pip install -r requirements.txt  (includes customtkinter)
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

import customtkinter as ctk

# --------------------------------------------------------------------------- #
# path setup -- this file lives next to pipeline.py, so local imports resolve
# whether it is launched from the folder or via `python -m tk_app`
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pipeline as P                                        # noqa: E402
import store                                            # noqa: E402
import backup                                           # noqa: E402

DB_PATH = HERE / "app.db"
STAGING_LABEL = "staging"

# Chat providers. "ollama" is local; the rest are hosted. All hosted ones
# speak OpenAI Chat Completions except Anthropic, so they share one backend
# in chat_providers.py -- only the base URL and the key differ.
CHAT_PROVIDERS = ["ollama", "anthropic", "openai", "opencode", "thegrid",
                  "custom"]
PROVIDER_KEY_SETTING = {
    "anthropic": "anthropic_key",
    "openai": "openai_key",
    "opencode": "opencode_key",
    "thegrid": "thegrid_key",
    "custom": "custom_key",
}
PROVIDER_MODEL_HINT = {
    "openai": "OpenAI model, e.g. gpt-4o-mini",
    "opencode": "OpenCode model, e.g. deepseek-v4.1-flash-free",
    "thegrid": "Grid instrument, e.g. text-prime",
    "custom": "model id at your endpoint",
}

# --------------------------------------------------------------------------- #
# appearance
# --------------------------------------------------------------------------- #

ctk.set_appearance_mode("system")          # follows Windows light/dark
ctk.set_default_color_theme("blue")

FONT = ("Segoe UI", 13)
FONT_MONO = ("Cascadia Mono", 12)
PAD = 10


def _short(path: str, n: int = 46) -> str:
    """Show the tail of long paths so rows stay readable."""
    p = str(path).replace("\\", "/")
    if len(p) <= n:
        return p
    return "…/" + p[-(n - 2):]


# --------------------------------------------------------------------------- #
# worker result queue -- the only channel from worker threads to the UI
# --------------------------------------------------------------------------- #

class Queue:
    """Typed message bus between worker threads and the UI thread."""

    def __init__(self) -> None:
        self._q: "queue.Queue[tuple]" = queue.Queue()

    def put(self, kind: str, *payload) -> None:
        self._q.put((kind, payload))

    def get_nowait(self):
        return self._q.get_nowait()


BUS = Queue()


# --------------------------------------------------------------------------- #
# markdown rendering -- small subset, no new dependencies
#
# Chat answers arrive as Markdown (bold, lists, code, [n] citations). Tkinter
# has no HTML widget, so a line parser maps the common constructs onto Text
# tags: headings, bold/italic, inline + fenced code, bullet/numbered lists,
# blockquotes, and styled [n] citations.
# --------------------------------------------------------------------------- #

_MD_HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
_MD_LIST = re.compile(r"^(\s*)(?:([-*+])|(\d+)[.)])\s+(.*)$")
_MD_QUOTE = re.compile(r"^>\s?(.*)$")
_MD_RULE = re.compile(r"^([-*_]\s*){3,}$")
_MD_FENCE = re.compile(r"^```(\w*)\s*$")
_MD_INLINE = re.compile(
    r"(\*\*.+?\*\*|__[^_]+?__|\*[^*\n]+?\*|_[^_\n]+_|`[^`\n]+?`|\[\d+\])")


def _md_palette() -> dict:
    import customtkinter as _ctk
    if _ctk.get_appearance_mode() == "Dark":
        return {"bg": "#212121", "fg": "#EDEDED",
                "code_bg": "#2B2B2B", "accent": "#4CC2FF", "muted": "#9E9E9E"}
    return {"bg": "#F9F9FA", "fg": "#1A1A1A",
            "code_bg": "#ECECEC", "accent": "#0B6E99", "muted": "#5A5A5A"}


def _md_fonts(widget: tk.Text) -> dict:
    """Bold/italic/mono variants of the widget font. Kept on the widget so
    Tk does not garbage-collect them out from under the tags."""
    base = tkfont.Font(font=widget.cget("font"))
    fonts = {
        "bold": base.copy(),
        "italic": base.copy(),
        "mono": tkfont.Font(family="Consolas", size=base.cget("size") - 1),
        "h1": base.copy(), "h2": base.copy(),
    }
    fonts["bold"].configure(weight="bold")
    fonts["italic"].configure(slant="italic")
    fonts["h1"].configure(weight="bold", size=base.cget("size") + 4)
    fonts["h2"].configure(weight="bold", size=base.cget("size") + 2)
    kept = getattr(widget, "_ep_fonts", [])
    kept.extend(fonts.values())
    widget._ep_fonts = kept                            # type: ignore[attr-defined]
    return fonts


def _md_insert_inline(widget: tk.Text, text: str, fonts: dict,
                      base_tags: tuple = ()) -> None:
    for part in _MD_INLINE.split(text):
        if not part:
            continue
        tags = list(base_tags)
        body = part
        if len(part) >= 4 and part.startswith("**") and part.endswith("**"):
            tags.append("bold")
            body = part[2:-2]
        elif len(part) >= 4 and part.startswith("__") and part.endswith("__"):
            tags.append("bold")
            body = part[2:-2]
        elif len(part) >= 2 and part.startswith("`") and part.endswith("`"):
            tags.append("code")
            body = part[1:-1]
        elif re.fullmatch(r"\[\d+\]", part):
            tags.append("cite")
        elif len(part) >= 2 and part[0] == "*" and part[-1] == "*":
            tags.append("italic")
            body = part[1:-1]
        elif len(part) >= 2 and part[0] == "_" and part[-1] == "_":
            tags.append("italic")
            body = part[1:-1]
        widget.insert("end", body, tuple(tags))


def render_markdown(widget: tk.Text, text: str) -> None:
    """Replace the widget contents with formatted Markdown. Leaves the widget
    enabled state to the caller (typically disabled right after)."""
    pal = _md_palette()
    fonts = _md_fonts(widget)
    widget.configure(bg=pal["bg"], fg=pal["fg"], insertbackground=pal["fg"],
                     relief="flat", highlightthickness=0, padx=8, pady=6,
                     wrap="word", cursor="xterm")
    for tag, kw in {
        "bold": {"font": fonts["bold"]},
        "italic": {"font": fonts["italic"]},
        "code": {"font": fonts["mono"], "background": pal["code_bg"]},
        "codeblock": {"font": fonts["mono"], "background": pal["code_bg"],
                      "lmargin1": 12, "lmargin2": 12},
        "h1": {"font": fonts["h1"], "spacing1": 8, "spacing3": 4},
        "h2": {"font": fonts["h2"], "spacing1": 6, "spacing3": 3},
        "cite": {"foreground": pal["accent"], "font": fonts["bold"]},
        "quote": {"foreground": pal["muted"], "lmargin1": 16, "lmargin2": 16},
        "list": {"lmargin1": 20, "lmargin2": 20},
        "muted": {"foreground": pal["muted"]},
    }.items():
        widget.tag_config(tag, **kw)

    widget.delete("1.0", "end")
    in_fence = False
    for raw in (text or "").strip().split("\n"):
        line = raw.rstrip()
        if _MD_FENCE.match(line.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            widget.insert("end", line + "\n", ("codeblock",))
            continue
        if not line.strip():
            widget.insert("end", "\n")
            continue
        m = _MD_HEADING.match(line)
        if m:
            widget.insert("end", m.group(2).strip() + "\n",
                          ("h1" if len(m.group(1)) <= 2 else "h2",))
            continue
        if _MD_RULE.match(line.strip()):
            widget.insert("end", ("─" * 40) + "\n", ("muted",))
            continue
        m = _MD_QUOTE.match(line)
        if m:
            _md_insert_inline(widget, m.group(1), fonts, ("quote",))
            widget.insert("end", "\n")
            continue
        m = _MD_LIST.match(line)
        if m:
            bullet = "• " if m.group(2) else f"{m.group(3)}. "
            widget.insert("end", "  " + bullet, ("list",))
            _md_insert_inline(widget, m.group(4), fonts, ("list",))
            widget.insert("end", "\n")
            continue
        _md_insert_inline(widget, line, fonts)
        widget.insert("end", "\n")


_FM_FENCE = re.compile(r"^---\s*$")


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML-ish frontmatter from a chunk body.

    Returns (meta, body). Handles ``---\\nkey: value\\n---`` at the top of the
    text; a malformed block (no closing ``---``) is returned untouched as
    body with an empty meta. Values are shown as-is -- no YAML dependency.
    """
    text = text or ""
    lines = text.split("\n")
    if not lines or not _FM_FENCE.match(lines[0].strip()):
        return {}, text.strip()
    meta: dict[str, str] = {}
    for idx in range(1, len(lines)):
        line = lines[idx]
        if _FM_FENCE.match(line.strip()):
            body = "\n".join(lines[idx + 1:]).strip()
            return meta, body
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lstrip("-").strip()
            val = val.strip().strip('"').strip("'")
            if key:
                meta[key] = val
    return {}, text.strip()   # no closing fence: keep the text as-is


def _fm_chips(meta: dict) -> str:
    """Compact 'key: value · key: value' line for the meta display."""
    parts = []
    for key, val in meta.items():
        if not val:
            continue
        shown = val if len(val) <= 40 else val[:37] + "…"
        parts.append(f"{key}: {shown}")
    return "  ·  ".join(parts)


def hit_body(h: dict) -> str:
    """Clean chunk text (no YAML frontmatter) -- for copy/paste and prompts."""
    _, body = split_frontmatter(h.get("text_raw") or "")
    return (body or (h.get("text_raw") or "").strip()).strip()


# --------------------------------------------------------------------------- #
# projects -- legacy JSON import (settings now live in SQLite, see store.py)
# --------------------------------------------------------------------------- #

PROJECT_VERSION = 1


def load_project_file(path: str | Path) -> dict:
    """Read + validate a project file. Raises PipelineError when unusable."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:                            # noqa: BLE001
        raise P.PipelineError(f"cannot read project file: {e}")
    if not isinstance(data, dict) or not isinstance(data.get("folders"), list):
        raise P.PipelineError("not a project file (need {folders: [...]})")
    folders = []
    for f in data["folders"]:
        if not isinstance(f, dict) or not str(f.get("path", "")).strip():
            continue
        try:
            weight = float(f.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        folders.append({"path": str(f["path"]),
                        "only": str(f.get("only", "")),
                        "weight": weight})
    data = dict(data)
    data["folders"] = folders
    return data


def _weight_label(weight: float) -> str:
    """Canonical menu text for a weight (falls back to %g for exotic values)."""
    try:
        f = float(weight)
    except (TypeError, ValueError):
        return "1.0"
    for cand in ("0.5", "1.0", "1.5", "2.0"):
        if f == float(cand):
            return cand
    return "%g" % f


# --------------------------------------------------------------------------- #
# staging -- per-project md sources (pasted notes, fetched links)
# --------------------------------------------------------------------------- #

def staging_path(pid: int) -> Path:
    """Staging folder for a project. Pure (no mkdir) so reads stay side-effect
    free; writers create it."""
    return HERE / "staging" / f"p{pid}"


def slugify_filename(name: str, fallback: str = "note") -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", (name or "").strip()).strip("-")
    return s[:80] or fallback


def _unique_md(staging: Path, slug: str) -> Path:
    out = staging / f"{slug}.md"
    n = 2
    while out.exists():
        out = staging / f"{slug}-{n}.md"
        n += 1
    return out


def save_note(staging: Path, title: str, text: str) -> Path:
    """Save pasted text as a Markdown source. Raises PipelineError."""
    title = (title or "").strip() or "Untitled note"
    if not (text or "").strip():
        raise P.PipelineError("note is empty")
    staging.mkdir(parents=True, exist_ok=True)
    out = _unique_md(staging, slugify_filename(title))
    out.write_text(f"# {title}\n\n{text.strip()}\n", encoding="utf-8")
    return out


def fetch_link(url: str, staging: Path) -> Path:
    """Download a page and store it as a Markdown source.

    Same extractors as ingest, so the file is exactly what the pipeline
    would embed. Works best with articles/docs pages. Raises PipelineError.
    """
    import urllib.parse
    import urllib.request

    url = (url or "").strip()
    if not url:
        raise P.PipelineError("no URL given")
    if not url.startswith(("http://", "https://", "file://")):
        url = "https://" + url
    req = urllib.request.Request(
        url, headers={"User-Agent": "embed_process/1.0 (+local RAG)"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            raw = r.read()
    except Exception as e:                            # noqa: BLE001
        raise P.PipelineError(f"download failed: {e}")
    text = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")

    staging.mkdir(parents=True, exist_ok=True)
    tmp = staging / "_fetch.tmp.html"
    tmp.write_text(text, encoding="utf-8")
    try:
        import extract as X
        title, blocks, kind = X.extract(tmp, root=staging)
    except Exception as e:                            # noqa: BLE001
        raise P.PipelineError(f"page not readable: {e}")
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    if not blocks:
        raise P.PipelineError("page had no readable content")
    if not title or title.startswith("_fetch"):
        u = urllib.parse.urlparse(url)
        title = ((u.netloc + " " + u.path.strip("/")).strip()) or url
    out = _unique_md(staging, slugify_filename(title))
    out.write_text(P.blocks_to_markdown(title, blocks, kind),
                   encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# background runner
# --------------------------------------------------------------------------- #

class Runner:
    """Runs pipeline callables **one at a time** on a worker thread, routing
    output to the UI.

    Jobs are queued: `run()` enqueues and returns at once, so a long embed no
    longer freezes the window -- you can switch projects and line up the next
    step, which starts by itself when the current one finishes. Serial
    execution is deliberate: `pipeline.log()` / `set_progress()` / `stop()` are
    module-level globals, so two live workers would clobber each other's log
    routing and Stop.

    Each job records the project + table it was queued for, so a result that
    lands *after* the user switched projects is reported in the log instead of
    being rendered into the wrong project's chat (`App._on_done`).
    """

    def __init__(self, app: "App") -> None:
        self.app = app
        self._queue: list[tuple] = []     # (title, fn, args, kwargs, ctx)
        self._busy = False
        self._current: str | None = None

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def current(self) -> str | None:
        """Title of the running job, or None when idle."""
        return self._current

    @property
    def pending(self) -> int:
        """How many jobs are waiting behind the current one."""
        return len(self._queue)

    def run(self, title: str, fn, *args, **kwargs) -> None:
        """Queue a job: starts now when idle, else waits its turn."""
        # Capture the project/table *now*. The args are already bound by
        # value, so a later project switch cannot retarget this job.
        ctx = {"title": title, "pid": self.app.project_id,
               "table": self.app.table_var.get().strip()}
        self._queue.append((title, fn, args, kwargs, ctx))
        if self._busy:
            self.app._say(f"queued: {title} — {len(self._queue)} waiting")
            self.app._update_queue_badge()
            return
        self._start_next()

    def _start_next(self) -> None:
        if not self._queue:
            return
        title, fn, args, kwargs, ctx = self._queue.pop(0)
        self._busy = True
        self._current = title
        self.app._last_job = title       # lets chat errors render inline
        self.app._job_ctx = ctx          # read by _on_done for tagging
        self.app.set_busy(True)
        self.app.set_progress(0, 1, title)
        self.app._update_queue_badge()

        def _work() -> None:
            try:
                result = fn(*args, **kwargs)
                BUS.put("done", result)
            except P.PipelineError as e:
                if str(e) == P.STOP_MESSAGE:
                    BUS.put("stopped")
                else:
                    BUS.put("failed", str(e))
            except Exception as e:                        # noqa: BLE001
                BUS.put("crashed", f"{type(e).__name__}: {e}")
            finally:
                P.set_logger(None)
                P.set_progress(None)
                BUS.put("finished")

        P.clear_stop()
        P.set_logger(lambda m: BUS.put("log", m))
        P.set_progress(lambda d, t: BUS.put("progress", d, t))
        threading.Thread(target=_work, daemon=True,
                         name=f"pipeline-{title}").start()

    def finish(self) -> None:
        """The current job ended: start the next one, or go idle."""
        self._busy = False
        self._current = None
        if self._queue:
            self._start_next()
            return
        self.app.set_busy(False)
        self.app.set_progress(1, 1, "idle")
        try:
            # Embed is enabled by a completed ingest, so keep doing it here.
            self.app.btn_embed.configure(state="normal")
        except Exception:                                # noqa: BLE001
            pass
        self.app._update_queue_badge()

    def clear_queue(self) -> int:
        """Drop the waiting jobs (Stop). Returns how many were dropped."""
        n = len(self._queue)
        self._queue.clear()
        self.app._update_queue_badge()
        return n


# --------------------------------------------------------------------------- #
# the application
# --------------------------------------------------------------------------- #

class App(ctk.CTk):
    WEIGHTS = ("0.5", "1.0", "1.5", "2.0")

    def __init__(self) -> None:
        super().__init__()
        self.title("embed_process — local RAG pipeline")
        self.geometry("1120x780")
        self.minsize(960, 660)
        self.configure(font=FONT)

        self.runner = Runner(self)
        self.folder_rows: list[ctk.CTkFrame] = []
        self.last_summary: dict | None = None
        self.last_hits: list[dict] = []
        self.last_query = ""
        self.last_answer = ""
        self._last_job = ""
        self._job_ctx: dict = {}          # project/table of the running job
        self.chat_models: list[str] = []
        self._history: list = []                    # chat rows, oldest first
        self.conversation: list[dict] = []          # multi-turn memory
        self._staging_rows: list = []
        self.project_id: int | None = None
        self._project_names: dict[str, int] = {}
        self._switching_project = False
        self._pending_row = None
        self._pending_status = None
        self._pending_body = None
        self._pending_t0: float = 0.0
        self._pending_after: str | None = None
        self._pending_note = ""

        self._build()
        # settings first: defaults, then the active project may override the
        # provider/model. Reversing this order let the global default clobber
        # the project's saved chat provider on every startup.
        self._load_settings()
        self._init_projects()
        self.after(80, self._drain)
        self.after(200, self.probe_ollama)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- layout ----------------

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_projectbar()

        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=PAD, pady=(PAD, 0))
        self.tabs.add("1 · Sources")
        self.tabs.add("2 · Index")
        self.tabs.add("3 · Chat")
        self.tabs.add("4 · Settings")
        self.tabs.set("1 · Sources")

        sources = self.tabs.tab("1 · Sources")
        sources.grid_columnconfigure(0, weight=1)
        self._build_sources(sources)
        self._build_staging(sources)

        index = self.tabs.tab("2 · Index")
        index.grid_columnconfigure(0, weight=1)
        self._build_settings(index)
        self._build_actions(index)
        self._build_log(index)

        chat = self.tabs.tab("3 · Chat")
        chat.grid_columnconfigure(0, weight=1)
        chat.grid_rowconfigure(0, weight=1)
        self._build_query(chat)

        settings = self.tabs.tab("4 · Settings")
        settings.grid_columnconfigure(0, weight=1)
        settings.grid_rowconfigure(0, weight=1)
        self._build_settings_tab(settings)

        self._build_status()

    def _build_projectbar(self) -> None:
        frm = ctk.CTkFrame(self)
        frm.grid(row=0, column=0, sticky="ew", padx=PAD, pady=(PAD, 0))
        ctk.CTkLabel(frm, text="RAG", font=(FONT[0], 14, "bold")
                     ).grid(row=0, column=0, sticky="w", padx=PAD, pady=8)
        self.project_var = ctk.StringVar(value="")
        self.project_menu = ctk.CTkOptionMenu(
            frm, variable=self.project_var, values=[], width=240,
            command=self._on_project_picked)
        self.project_menu.grid(row=0, column=1, sticky="w", padx=(0, 8))
        ctk.CTkButton(frm, text="+ New", width=80, command=self.do_new_project
                      ).grid(row=0, column=2, padx=(0, 8))
        self.btn_new_project = frm.grid_slaves(row=0, column=2)[0]
        ctk.CTkButton(frm, text="Rename", width=80, fg_color="transparent",
                      border_width=1, command=self.do_rename_project
                      ).grid(row=0, column=3, padx=(0, 8))
        self.btn_rename_project = frm.grid_slaves(row=0, column=3)[0]
        ctk.CTkButton(frm, text="Delete", width=80, fg_color="transparent",
                      border_width=1, command=self.do_delete_project
                      ).grid(row=0, column=4)
        self.btn_delete_project = frm.grid_slaves(row=0, column=4)[0]

    def _build_sources(self, parent) -> None:
        frm = ctk.CTkFrame(parent)
        frm.grid(row=0, column=0, sticky="ew", padx=PAD, pady=(PAD, 0))
        frm.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(frm, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", pady=(6, 0), padx=PAD)
        head.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(head, text="Sources", font=(FONT[0], 14, "bold")
                     ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(head, text="Add folder", width=110, command=self.add_folder
                      ).grid(row=0, column=2, padx=(6, 0))
        ctk.CTkButton(head, text="Clear", width=70, fg_color="transparent",
                      border_width=1, command=self.clear_folders
                      ).grid(row=0, column=3, padx=(6, 0))

        # column headers for the folder rows
        hdr = ctk.CTkFrame(frm, fg_color="transparent")
        hdr.grid(row=1, column=0, sticky="ew", pady=(8, 0), padx=PAD)
        for txt, col, w in (("Label", 0, 0), ("Folder", 1, 0),
                            ("Extensions (blank = all)", 2, 0),
                            ("Weight", 3, 0)):
            ctk.CTkLabel(hdr, text=txt, anchor="w", text_color=("gray35", "gray70")
                         ).grid(row=0, column=col, sticky="w", padx=(0, 12))
        hdr.grid_columnconfigure(1, weight=1)

        self.folder_list = ctk.CTkScrollableFrame(frm, height=148,
                                                  label_text="")
        self.folder_list.grid(row=2, column=0, sticky="ew", padx=PAD, pady=(2, PAD))
        self.folder_list.grid_columnconfigure(1, weight=1)

        # one placeholder row so the UI is never blank on startup
        self._new_row("")

    def _build_staging(self, parent) -> None:
        """Notes & links collected as .md sources for the active project."""
        frm = ctk.CTkFrame(parent)
        frm.grid(row=1, column=0, sticky="ew", padx=PAD, pady=(0, PAD))
        frm.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frm, text="Staging — notes & links (.md)",
                     font=(FONT[0], 14, "bold")
                     ).grid(row=0, column=0, sticky="w", padx=PAD, pady=(8, 0))
        self.staging_lbl = ctk.CTkLabel(frm, text="", anchor="w",
                                        text_color=("gray45", "gray65"))
        self.staging_lbl.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ctk.CTkButton(frm, text="+ Note", width=80, command=self.do_add_note
                      ).grid(row=0, column=2, padx=(6, 0), pady=(8, 0))
        ctk.CTkButton(frm, text="+ Link", width=80, command=self.do_add_link
                      ).grid(row=0, column=3, padx=(6, PAD), pady=(8, 0))

        self.staging_list = ctk.CTkScrollableFrame(frm, height=90,
                                                   label_text="")
        self.staging_list.grid(row=1, column=0, columnspan=4, sticky="ew",
                               padx=PAD, pady=(2, PAD))
        self.staging_list.grid_columnconfigure(0, weight=1)

    def _refresh_staging(self) -> None:
        # NOTE: destroy tracked rows only -- winfo_children() of a
        # CTkScrollableFrame lists its canvas/scrollbar, not our rows.
        for row in getattr(self, "_staging_rows", []):
            try:
                row.destroy()
            except Exception:                                # noqa: BLE001
                pass
        self._staging_rows = []
        files = []
        if self.project_id:
            d = staging_path(self.project_id)
            if d.is_dir():
                files = sorted(p for p in d.glob("*.md") if p.is_file())
            self.staging_lbl.configure(text=f"{len(files)} file(s) · {d}")
        for p in files:
            row = ctk.CTkFrame(self.staging_list)
            row.pack(fill="x", padx=2, pady=2)
            row.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(row, text=p.name, anchor="w").grid(
                row=0, column=0, sticky="w", padx=(6, 4))
            ctk.CTkButton(row, text="✕", width=28, fg_color="transparent",
                          border_width=1,
                          command=lambda pp=p: self._delete_staging(pp)
                          ).grid(row=0, column=1, padx=(0, 4))
            self._staging_rows.append(row)
        if not files:
            lbl = ctk.CTkLabel(self.staging_list,
                               text="(empty — add notes or links)",
                               text_color=("gray50", "gray60"))
            lbl.pack(anchor="w", padx=8, pady=6)
            self._staging_rows.append(lbl)

    def _delete_staging(self, path: Path) -> None:
        try:
            path.unlink()
            self._say(f"staging deleted: {path.name}")
        except OSError as e:
            messagebox.showerror("Delete failed", str(e))
        self._refresh_staging()

    def _build_settings(self, parent) -> None:
        frm = ctk.CTkFrame(parent)
        frm.grid(row=1, column=0, sticky="ew", padx=PAD, pady=(PAD, 0))
        # columns 0..8 are all used by the widgets below
        for i in range(9):
            frm.grid_columnconfigure(i, weight=0)

        ctk.CTkLabel(frm, text="Table").grid(row=0, column=0, sticky="w",
                                             padx=PAD, pady=PAD)
        self.table_var = ctk.StringVar(value="corpus")
        ctk.CTkEntry(frm, textvariable=self.table_var, width=180
                     ).grid(row=0, column=1, sticky="w", padx=(0, 16))

        ctk.CTkLabel(frm, text="Embed model").grid(row=0, column=2, sticky="w")
        self.model_var = ctk.StringVar(value=P.DEFAULT_MODEL)
        self.model_menu = ctk.CTkOptionMenu(frm, variable=self.model_var,
                                            values=[P.DEFAULT_MODEL], width=230)
        self.model_menu.grid(row=0, column=3, sticky="w", padx=(0, 16))

        ctk.CTkLabel(frm, text="Chunk size").grid(row=0, column=4, sticky="w")
        self.size_var = ctk.StringVar(value=str(P.CHUNK_SIZE))
        ctk.CTkEntry(frm, textvariable=self.size_var, width=64
                     ).grid(row=0, column=5, sticky="w", padx=(4, 16))

        ctk.CTkLabel(frm, text="Overlap").grid(row=0, column=6, sticky="w")
        self.overlap_var = ctk.StringVar(value=str(P.CHUNK_OVERLAP))
        ctk.CTkEntry(frm, textvariable=self.overlap_var, width=64
                     ).grid(row=0, column=7, sticky="w", padx=(4, 16))

        self.replace_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(frm, text="Replace table", variable=self.replace_var
                        ).grid(row=0, column=8, sticky="w", padx=(4, PAD))

    def _build_actions(self, parent) -> None:
        frm = ctk.CTkFrame(parent, fg_color="transparent")
        frm.grid(row=2, column=0, sticky="ew", padx=PAD, pady=PAD)
        for i in range(8):
            frm.grid_columnconfigure(i, weight=0)

        self.btn_ingest = ctk.CTkButton(frm, text="1 · Ingest",
                                        command=self.do_ingest)
        self.btn_ingest.grid(row=0, column=0, padx=(0, 8), pady=(0, 6))

        self.btn_embed = ctk.CTkButton(frm, text="2 · Embed", state="disabled",
                                       command=self.do_embed)
        self.btn_embed.grid(row=0, column=1, padx=(0, 8), pady=(0, 6))

        self.btn_sync = ctk.CTkButton(frm, text="Sync", fg_color="transparent",
                                      border_width=1, command=self.do_sync)
        self.btn_sync.grid(row=0, column=2, padx=(0, 8), pady=(0, 6))

        self.btn_query = ctk.CTkButton(frm, text="Query", command=self.do_query)
        self.btn_query.grid(row=0, column=3, padx=(0, 8), pady=(0, 6))
        self.btn_stats = ctk.CTkButton(frm, text="Stats", command=self.do_stats)
        self.btn_stats.grid(row=0, column=4, padx=(0, 8), pady=(0, 6))
        self.btn_reindex = ctk.CTkButton(frm, text="Re-index",
                                         fg_color="transparent", border_width=1,
                                         command=self.do_reindex)
        self.btn_reindex.grid(row=0, column=5, padx=(0, 8), pady=(0, 6))
        self.btn_export = ctk.CTkButton(frm, text="Export MD",
                                        fg_color="transparent", border_width=1,
                                        command=self.do_export_md)
        self.btn_export.grid(row=0, column=6, padx=(0, 8), pady=(0, 6))

        # row 1: terminal agents + source manager + stop, then progress/status
        self.btn_opencode = ctk.CTkButton(frm, text="OpenCode",
                                          fg_color="transparent", border_width=1,
                                          command=self.do_opencode)
        self.btn_opencode.grid(row=1, column=0, padx=(0, 8))
        self.btn_commandcode = ctk.CTkButton(frm, text="CmdCode",
                                             fg_color="transparent", border_width=1,
                                             command=self.do_commandcode)
        self.btn_commandcode.grid(row=1, column=1, padx=(0, 8))
        self.btn_manage_sources = ctk.CTkButton(
            frm, text="Sources…", fg_color="transparent", border_width=1,
            command=self.do_manage_sources)
        self.btn_manage_sources.grid(row=1, column=2, padx=(0, 8))
        self.btn_stop = ctk.CTkButton(frm, text="Stop", width=88,
                                      fg_color="transparent", border_width=1,
                                      state="disabled", command=self.do_stop)
        self.btn_stop.grid(row=1, column=3, padx=(0, 8))

        self.progress = ctk.CTkProgressBar(frm)
        self.progress.grid(row=1, column=4, columnspan=2, sticky="ew",
                           padx=(16, 8))
        self.progress.set(0)
        self.status_lbl = ctk.CTkLabel(frm, text="idle", width=150, anchor="w")
        self.status_lbl.grid(row=1, column=6)

    def _build_log(self, parent) -> None:
        self.log = ctk.CTkTextbox(parent, font=FONT_MONO, wrap="none",
                                  activate_scrollbars=True)
        self.log.grid(row=3, column=0, sticky="nsew", padx=PAD, pady=(0, PAD))
        self.log.configure(state="disabled")
        parent.grid_rowconfigure(3, weight=3, minsize=190)

    def _build_query(self, parent) -> None:
        frm = ctk.CTkFrame(parent, fg_color="transparent")
        self.query_frm = frm                    # for tests / geometry checks
        frm.grid(row=0, column=0, sticky="nsew", padx=PAD, pady=PAD)
        frm.grid_columnconfigure(0, weight=1)
        frm.grid_rowconfigure(1, weight=1)          # conversation takes all

        head = ctk.CTkFrame(frm, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(head, text="Assistant", font=(FONT[0], 14, "bold")
                     ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(head, text="Copy answer", width=100,
                      fg_color="transparent", border_width=1,
                      command=self.do_copy_answer
                      ).grid(row=0, column=1, sticky="e", padx=(0, 6))
        ctk.CTkButton(head, text="New chat", width=90,
                      fg_color="transparent", border_width=1,
                      command=self.clear_chat
                      ).grid(row=0, column=2, sticky="e")

        self.results = ctk.CTkScrollableFrame(frm, label_text="")
        self.results.grid(row=1, column=0, sticky="nsew")
        self.results.grid_columnconfigure(0, weight=1)
        self.results_empty = ctk.CTkLabel(
            self.results,
            text="Send a message below to start the conversation.\n"
                 "Your indexed corpus is used as background knowledge.",
            text_color=("gray55", "gray65"), justify="center")
        self.results_empty.pack(expand=True, pady=40)

        dock = ctk.CTkFrame(frm)
        dock.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        dock.grid_columnconfigure(0, weight=1)

        self.query_var = ctk.StringVar()
        entry = ctk.CTkEntry(dock, textvariable=self.query_var,
                             placeholder_text="write a message…",
                             height=36, font=(FONT[0], 13))
        entry.grid(row=0, column=0, columnspan=4, sticky="ew",
                   padx=PAD, pady=(PAD, 4))
        entry.bind("<Return>", lambda _e: self.do_send())

        row2 = ctk.CTkFrame(dock, fg_color="transparent")
        row2.grid(row=1, column=0, sticky="ew", padx=PAD, pady=(0, PAD))
        row2.grid_columnconfigure(3, weight=1)
        self.provider_var = ctk.StringVar(value="ollama")
        self.provider_menu = ctk.CTkOptionMenu(
            row2, variable=self.provider_var,
            values=list(CHAT_PROVIDERS), width=110,
            command=self._on_provider_picked)
        self.provider_menu.grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.chat_model_var = ctk.StringVar(value="")
        self.chat_menu = ctk.CTkOptionMenu(row2, variable=self.chat_model_var,
                                           values=["(checking…)"], width=170)
        self.chat_menu.grid(row=0, column=1, sticky="w", padx=(0, 4))
        self.chat_entry_var = ctk.StringVar(value="")
        self.chat_entry = ctk.CTkEntry(row2, textvariable=self.chat_entry_var,
                                       placeholder_text="type model name",
                                       width=170)
        # kept for Settings/Query, not shown here -- chat uses settings top-k
        self.topk_var = ctk.StringVar(value="8")
        self.hybrid_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(row2, text="Hybrid", variable=self.hybrid_var,
                        width=80).grid(row=0, column=2, sticky="w",
                                       padx=(0, 4))
        ctk.CTkButton(row2, text="Send", width=90, font=(FONT[0], 13),
                      command=self.do_send).grid(row=0, column=4, padx=(8, 0))

        # saved queries -- per project; pick one to load it into the box
        saved_row = ctk.CTkFrame(dock, fg_color="transparent")
        saved_row.grid(row=2, column=0, sticky="ew", padx=PAD, pady=(0, PAD))
        saved_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(saved_row, text="Saved:", anchor="w").grid(
            row=0, column=0, padx=(0, 6))
        self.saved_var = ctk.StringVar(value="(none)")
        self.saved_menu = ctk.CTkOptionMenu(
            saved_row, variable=self.saved_var, values=["(none)"], width=280,
            command=self._on_saved_picked)
        self.saved_menu.grid(row=0, column=1, sticky="w", padx=(0, 6))
        self.btn_save_query = ctk.CTkButton(
            saved_row, text="Save current", width=110, fg_color="transparent",
            border_width=1, command=self.do_save_query)
        self.btn_save_query.grid(row=0, column=2, padx=(0, 6))
        self.btn_del_query = ctk.CTkButton(
            saved_row, text="✕", width=32, fg_color="transparent",
            border_width=1, command=self.do_delete_saved_query)
        self.btn_del_query.grid(row=0, column=3)
        self._saved_map: dict[str, tuple[int, str]] = {}

    def _build_settings_tab(self, parent) -> None:
        wrap = ctk.CTkScrollableFrame(parent, label_text="")
        wrap.grid(row=0, column=0, sticky="nsew", padx=PAD, pady=PAD)
        wrap.grid_columnconfigure(1, weight=1)
        self.sv: dict[str, ctk.StringVar] = {}

        def var(key: str, default: str = "") -> ctk.StringVar:
            v = ctk.StringVar(value=default)
            self.sv[key] = v
            return v

        r = 0
        ctk.CTkLabel(wrap, text="Ollama", font=(FONT[0], 14, "bold")
                     ).grid(row=r, column=0, columnspan=4, sticky="w",
                            padx=PAD, pady=(8, 2)); r += 1
        ctk.CTkLabel(wrap, text="Host").grid(row=r, column=0, sticky="w",
                                             padx=PAD)
        ctk.CTkEntry(wrap, textvariable=var("ollama_host"),
                     placeholder_text="http://localhost:11434 (blank=default)"
                     ).grid(row=r, column=1, sticky="ew", padx=(0, 8))
        ctk.CTkButton(wrap, text="Test", width=80,
                      command=self.do_test_ollama).grid(row=r, column=2)
        self.ollama_test_lbl = ctk.CTkLabel(wrap, text="", anchor="w",
                                            text_color=("gray45", "gray65"))
        self.ollama_test_lbl.grid(row=r, column=3, sticky="w", padx=(8, PAD))
        r += 1

        ctk.CTkLabel(wrap, text="Defaults for new projects",
                     font=(FONT[0], 14, "bold")
                     ).grid(row=r, column=0, columnspan=4, sticky="w",
                            padx=PAD, pady=(10, 2)); r += 1
        ctk.CTkLabel(wrap, text="Embed model").grid(row=r, column=0,
                                                    sticky="w", padx=PAD)
        ctk.CTkEntry(wrap, textvariable=var("default_embed_model"),
                     placeholder_text="(blank = current selection)"
                     ).grid(row=r, column=1, sticky="ew", padx=(0, 8))
        ctk.CTkLabel(wrap, text="Chat model").grid(row=r, column=2, sticky="w")
        ctk.CTkEntry(wrap, textvariable=var("default_chat_model"),
                     placeholder_text="(blank = current selection)"
                     ).grid(row=r, column=3, sticky="ew", padx=(0, PAD))
        r += 1
        ctk.CTkLabel(wrap, text="Provider").grid(row=r, column=0, sticky="w",
                                                 padx=PAD)
        prov = ctk.CTkOptionMenu(wrap, variable=var("default_provider",
                                                    "ollama"),
                                 values=list(CHAT_PROVIDERS),
                                 width=140)
        prov.grid(row=r, column=1, sticky="w", padx=(0, 8))
        ctk.CTkLabel(wrap, text="Chunk / overlap / top-k").grid(
            row=r, column=2, sticky="w")
        sub = ctk.CTkFrame(wrap, fg_color="transparent")
        sub.grid(row=r, column=3, sticky="w", padx=(0, PAD))
        ctk.CTkEntry(sub, textvariable=var("chunk_size", "900"), width=64
                     ).grid(row=0, column=0, padx=(0, 4))
        ctk.CTkEntry(sub, textvariable=var("overlap", "120"), width=64
                     ).grid(row=0, column=1, padx=(0, 4))
        ctk.CTkEntry(sub, textvariable=var("top_k", "8"), width=56
                     ).grid(row=0, column=2)
        r += 1

        ctk.CTkLabel(wrap, text="API keys (stored in app.db, shown masked)",
                     font=(FONT[0], 14, "bold")
                     ).grid(row=r, column=0, columnspan=4, sticky="w",
                            padx=PAD, pady=(10, 2)); r += 1
        self._key_row(wrap, r, "Anthropic (Claude)", var("anthropic_key"),
                      "anthropic"); r += 1
        self._key_row(wrap, r, "OpenAI (ChatGPT)", var("openai_key"),
                      "openai"); r += 1
        self._key_row(wrap, r, "OpenCode", var("opencode_key"),
                      "opencode"); r += 1
        self._key_row(wrap, r, "The Grid", var("thegrid_key"),
                      "thegrid"); r += 1
        self._key_row(wrap, r, "Custom (OpenAI-compatible)", var("custom_key"),
                      "custom"); r += 1
        ctk.CTkLabel(wrap, text="Custom base URL").grid(row=r, column=0,
                                                        sticky="w", padx=PAD)
        ctk.CTkEntry(wrap, textvariable=var("custom_base_url"),
                     placeholder_text="https://host/v1  (path /chat/completions is appended)"
                     ).grid(row=r, column=1, columnspan=3, sticky="ew",
                            padx=(0, PAD))
        r += 1
        self.provider_test_lbl = ctk.CTkLabel(
            wrap, text="", anchor="w", justify="left", wraplength=760,
            text_color=("gray45", "gray65"))
        self.provider_test_lbl.grid(row=r, column=0, columnspan=6, sticky="w",
                                    padx=PAD, pady=(2, 0))
        r += 1
        self.provider_models_lbl = ctk.CTkLabel(
            wrap, text="", anchor="w", justify="left", wraplength=760,
            text_color=("gray45", "gray65"))
        self.provider_models_lbl.grid(row=r, column=0, columnspan=6,
                                      sticky="w", padx=PAD)
        r += 1
        self.provider_models_box = ctk.CTkTextbox(wrap, font=FONT_MONO,
                                                  height=120,
                                                  activate_scrollbars=True)
        self.provider_models_box.grid(row=r, column=0, columnspan=6,
                                      sticky="ew", padx=PAD, pady=(2, 0))
        self.provider_models_box.configure(state="disabled")
        r += 1

        ctk.CTkLabel(wrap, text="Data locations (fixed)",
                     font=(FONT[0], 14, "bold")
                     ).grid(row=r, column=0, columnspan=4, sticky="w",
                            padx=PAD, pady=(10, 2)); r += 1
        ctk.CTkLabel(wrap, text=f"index/  {P.INDEX_DIR}\n"
                                f"app.db  {DB_PATH}\n"
                                f"staging/  {HERE / 'staging'}",
                     anchor="w", justify="left",
                     text_color=("gray45", "gray65")
                     ).grid(row=r, column=0, columnspan=4, sticky="w",
                            padx=PAD); r += 1

        ctk.CTkLabel(wrap, text="Backup / Transfer",
                     font=(FONT[0], 14, "bold")
                     ).grid(row=r, column=0, columnspan=4, sticky="w",
                            padx=PAD, pady=(10, 2)); r += 1
        ctk.CTkLabel(wrap, text="Full bundle (.zip): app.db + index/ + "
                                "staging/ — move or restore everything",
                     text_color=("gray45", "gray65"), anchor="w"
                     ).grid(row=r, column=0, columnspan=4, sticky="w",
                            padx=PAD); r += 1
        self.btn_backup_zip = ctk.CTkButton(
            wrap, text="Backup…", width=110, command=self.do_backup_zip)
        self.btn_backup_zip.grid(row=r, column=0, sticky="w", padx=PAD)
        self.btn_restore_zip = ctk.CTkButton(
            wrap, text="Restore…", width=110, fg_color="transparent",
            border_width=1, command=self.do_restore_zip)
        self.btn_restore_zip.grid(row=r, column=1, sticky="w")
        ctk.CTkCheckBox(wrap, text="include API keys",
                        variable=var("backup_include_keys", "0"),
                        onvalue="1", offvalue="0", width=140
                        ).grid(row=r, column=2, columnspan=2, sticky="w",
                               padx=(8, PAD))
        r += 1

        ctk.CTkLabel(wrap, text="Settings only (.json): no vectors "
                                "(lightweight)",
                     text_color=("gray45", "gray65"), anchor="w"
                     ).grid(row=r, column=0, columnspan=4, sticky="w",
                            padx=PAD); r += 1
        self.btn_backup_json = ctk.CTkButton(
            wrap, text="Backup…", width=110, command=self.do_backup)
        self.btn_backup_json.grid(row=r, column=0, sticky="w", padx=PAD)
        self.btn_restore_json = ctk.CTkButton(
            wrap, text="Restore…", width=110, fg_color="transparent",
            border_width=1, command=self.do_restore)
        self.btn_restore_json.grid(row=r, column=1, sticky="w")
        ctk.CTkButton(wrap, text="Open data folder", width=140,
                      fg_color="transparent", border_width=1,
                      command=self.do_open_data_folder
                      ).grid(row=r, column=2, columnspan=2, sticky="w",
                             padx=(8, PAD))
        r += 1

        ctk.CTkLabel(wrap, text="This project only (.zip): its vectors + "
                                "staging, appended on import",
                     text_color=("gray45", "gray65"), anchor="w"
                     ).grid(row=r, column=0, columnspan=4, sticky="w",
                            padx=PAD); r += 1
        self.btn_export_project = ctk.CTkButton(
            wrap, text="Export project…", width=140,
            command=self.do_export_project)
        self.btn_export_project.grid(row=r, column=0, sticky="w", padx=PAD)
        self.btn_import_project = ctk.CTkButton(
            wrap, text="Import project…", width=140, fg_color="transparent",
            border_width=1, command=self.do_import_project)
        self.btn_import_project.grid(row=r, column=1, sticky="w")
        r += 1

        ctk.CTkLabel(wrap, text="Ollama models",
                     font=(FONT[0], 14, "bold")
                     ).grid(row=r, column=0, columnspan=4, sticky="w",
                            padx=PAD, pady=(10, 2)); r += 1
        ctk.CTkButton(wrap, text="Refresh", width=110,
                      command=self._refresh_models).grid(
            row=r, column=0, sticky="w", padx=PAD)
        self.models_lbl = ctk.CTkLabel(wrap, text="", anchor="w",
                                       text_color=("gray45", "gray65"))
        self.models_lbl.grid(row=r, column=1, columnspan=3, sticky="w")
        r += 1
        self.models_box = ctk.CTkTextbox(wrap, font=FONT_MONO, height=110,
                                         activate_scrollbars=True)
        self.models_box.grid(row=r, column=0, columnspan=4, sticky="ew",
                             padx=PAD, pady=(2, 4))
        self.models_box.configure(state="disabled")
        r += 1
        ctk.CTkLabel(wrap, text="Model name").grid(row=r, column=0, sticky="w",
                                                   padx=PAD)
        self.model_dl_var = ctk.StringVar()
        ctk.CTkEntry(wrap, textvariable=self.model_dl_var,
                     placeholder_text="e.g. qwen3:0.6b").grid(
            row=r, column=1, sticky="ew", padx=(0, 8))
        ctk.CTkButton(wrap, text="Pull", width=90,
                      command=self.do_pull_model).grid(row=r, column=2)
        ctk.CTkButton(wrap, text="Delete", width=90, fg_color="transparent",
                      border_width=1, command=self.do_delete_model).grid(
            row=r, column=3, sticky="w", padx=(0, PAD))
        r += 1

        ctk.CTkLabel(wrap, text="Chat / RAG helper",
                     font=(FONT[0], 14, "bold")
                     ).grid(row=r, column=0, columnspan=4, sticky="w",
                            padx=PAD, pady=(10, 2)); r += 1
        ctk.CTkCheckBox(wrap, text="Use corpus as helper (off = plain LLM)",
                        variable=var("rag_enabled", "1"),
                        onvalue="1", offvalue="0"
                        ).grid(row=r, column=0, columnspan=2, sticky="w",
                               padx=PAD)
        ctk.CTkLabel(wrap, text="Min similarity (0-1)").grid(
            row=r, column=2, sticky="w")
        ctk.CTkEntry(wrap, textvariable=var("rag_min_cos", "0.30"), width=64
                     ).grid(row=r, column=3, sticky="w", padx=(0, PAD))
        r += 1
        ctk.CTkLabel(wrap, text="Greetings and low-similarity questions "
                                "answer without corpus.",
                     text_color=("gray45", "gray65")
                     ).grid(row=r, column=0, columnspan=4, sticky="w",
                            padx=PAD); r += 1

        ctk.CTkCheckBox(wrap, text="Hybrid search (vector + keyword, off = vector-only)",
                        variable=var("hybrid_search", "1"),
                        onvalue="1", offvalue="0",
                        command=self._on_hybrid_setting
                        ).grid(row=r, column=0, columnspan=4, sticky="w",
                               padx=PAD); r += 1

        ctk.CTkLabel(wrap, text="Vector index",
                     font=(FONT[0], 14, "bold")
                     ).grid(row=r, column=0, columnspan=4, sticky="w",
                            padx=PAD, pady=(10, 2)); r += 1
        ctk.CTkLabel(wrap, text="nprobes").grid(row=r, column=0, sticky="w",
                                               padx=PAD)
        ctk.CTkEntry(wrap, textvariable=var("vector_nprobes", "0"), width=64
                     ).grid(row=r, column=1, sticky="w")
        ctk.CTkLabel(wrap, text="refine factor").grid(row=r, column=2,
                                                      sticky="w")
        ctk.CTkEntry(wrap, textvariable=var("vector_refine_factor", "0"),
                     width=64).grid(row=r, column=3, sticky="w", padx=(0, PAD))
        r += 1
        ctk.CTkLabel(wrap, text="0 = LanceDB default. Both are ignored until an "
                                "ANN vector index exists.",
                     text_color=("gray45", "gray65"), anchor="w"
                     ).grid(row=r, column=0, columnspan=4, sticky="w",
                            padx=PAD); r += 1
        self.ann_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(wrap, text="also build ANN (IVF_PQ)",
                        variable=self.ann_var
                        ).grid(row=r, column=0, columnspan=2, sticky="w",
                               padx=PAD)
        self.btn_index = ctk.CTkButton(
            wrap, text="Build / refresh indexes", width=180,
            fg_color="transparent", border_width=1,
            command=self.do_index_health)
        self.btn_index.grid(row=r, column=2, columnspan=2, sticky="w",
                            padx=(0, PAD))
        r += 1
        ctk.CTkLabel(wrap, text="Rebuilds the FTS + source indexes and compacts "
                                "fragments (safe and quick). ANN is lossy, needs "
                                "reseeding, and rarely beats exact search below "
                                "~20k vectors.",
                     text_color=("gray45", "gray65"), anchor="w",
                     justify="left", wraplength=760
                     ).grid(row=r, column=0, columnspan=4, sticky="w",
                            padx=PAD); r += 1

        ctk.CTkLabel(wrap, text="Terminal agents",
                     font=(FONT[0], 14, "bold")
                     ).grid(row=r, column=0, columnspan=4, sticky="w",
                            padx=PAD, pady=(10, 2)); r += 1
        ctk.CTkLabel(wrap, text="opencode command").grid(
            row=r, column=0, sticky="w", padx=PAD)
        ctk.CTkEntry(wrap, textvariable=var("opencode_cmd", "opencode")
                     ).grid(row=r, column=1, sticky="ew", padx=(0, 8))
        ctk.CTkButton(wrap, text="Test", width=80,
                      command=lambda: self.do_test_terminal("opencode_cmd")
                      ).grid(row=r, column=2)
        r += 1
        ctk.CTkLabel(wrap, text="command-code command").grid(
            row=r, column=0, sticky="w", padx=PAD)
        ctk.CTkEntry(wrap, textvariable=var("commandcode_cmd", "command-code")
                     ).grid(row=r, column=1, sticky="ew", padx=(0, 8))
        ctk.CTkButton(wrap, text="Test", width=80,
                      command=lambda: self.do_test_terminal("commandcode_cmd")
                      ).grid(row=r, column=2)
        r += 1
        self.term_test_lbl = ctk.CTkLabel(wrap, text="", anchor="w",
                                          text_color=("gray45", "gray65"))
        self.term_test_lbl.grid(row=r, column=0, columnspan=4, sticky="w",
                                padx=PAD); r += 1

        ctk.CTkButton(wrap, text="Save settings", width=140,
                      command=self.do_save_settings).grid(
            row=r, column=0, sticky="w", padx=PAD, pady=(12, PAD))

    def _key_row(self, parent, row: int, label: str,
                 var: ctk.StringVar, provider: str = "") -> None:
        ctk.CTkLabel(parent, text=label).grid(row=row, column=0, sticky="w",
                                              padx=PAD)
        entry = ctk.CTkEntry(parent, textvariable=var, show="•")
        entry.grid(row=row, column=1, sticky="ew", padx=(0, 8))
        btn = ctk.CTkButton(parent, text="Show", width=70,
                            fg_color="transparent", border_width=1)
        btn.configure(command=lambda e=entry, b=btn: self._toggle_key(e, b))
        btn.grid(row=row, column=2)
        ctk.CTkButton(parent, text="Clear", width=70, fg_color="transparent",
                      border_width=1,
                      command=lambda v=var: v.set("")
                      ).grid(row=row, column=3, sticky="w", padx=(4, PAD))
        if provider:
            ctk.CTkButton(parent, text="Test", width=64,
                          fg_color="transparent", border_width=1,
                          command=lambda p=provider: self.do_test_provider(p)
                          ).grid(row=row, column=4, sticky="w", padx=(0, 6))
            ctk.CTkButton(parent, text="Models", width=74,
                          fg_color="transparent", border_width=1,
                          command=lambda p=provider: self.do_list_models(p)
                          ).grid(row=row, column=5, sticky="w", padx=(0, PAD))

    @staticmethod
    def _toggle_key(entry, btn) -> None:
        if entry.cget("show") == "•":
            entry.configure(show="")
            btn.configure(text="Hide")
        else:
            entry.configure(show="•")
            btn.configure(text="Show")

    def _build_status(self) -> None:
        frm = ctk.CTkFrame(self, fg_color="transparent")
        frm.grid(row=2, column=0, sticky="ew", padx=PAD, pady=(0, 6))
        frm.grid_columnconfigure(1, weight=1)
        self.dot = ctk.CTkLabel(frm, text="●", width=16,
                                text_color=("gray60", "gray40"))
        self.dot.grid(row=0, column=0)
        self.ollama_lbl = ctk.CTkLabel(frm, text="checking Ollama…", anchor="w")
        self.ollama_lbl.grid(row=0, column=1, sticky="w")

    # ---------------- folder rows ----------------

    def add_folder(self, path: str = "") -> None:
        """Append a folder row. Prompts for a folder if none was given."""
        if not path:
            path = filedialog.askdirectory(parent=self,
                                           title="Pick a folder to index")
            # user cancelled: leave the list alone rather than adding a blank
            if not path:
                return
        self._new_row(path)

    def _new_row(self, path: str = "") -> ctk.CTkFrame:
        """Build one folder row. Empty path = placeholder, skipped by _sources.

        Columns: 0 name | 1 folder | 2 extensions | 3 weight | 4 browse | 5 remove.
        Never opens a dialog.
        """
        row = ctk.CTkFrame(self.folder_list)
        row.grid(row=len(self.folder_rows), column=0, sticky="ew", pady=(0, 6))
        row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(row, text="", width=90, anchor="w").grid(
            row=0, column=0, sticky="w", padx=(6, 8))
        ctk.CTkLabel(row, text="", anchor="w",
                     text_color=("gray35", "gray70")).grid(
            row=0, column=1, sticky="w", padx=(0, 8))
        ctk.CTkEntry(row, width=150,
                     placeholder_text=".md,.pdf,.py").grid(
            row=0, column=2, sticky="e", padx=(0, 8))
        menu = ctk.CTkOptionMenu(row, values=list(self.WEIGHTS), width=76)
        menu.set("1.0")
        menu.grid(row=0, column=3, sticky="e", padx=(0, 8))
        ctk.CTkButton(row, text="Browse", width=70, fg_color="transparent",
                      border_width=1,
                      command=lambda r=row: self.browse_folder(r)
                      ).grid(row=0, column=4, padx=(0, 4))
        ctk.CTkButton(row, text="✕", width=28, fg_color="transparent",
                      border_width=1, command=lambda r=row: self.remove_folder(r)
                      ).grid(row=0, column=5, padx=(0, 4))

        self.folder_rows.append(row)
        self._set_row_path(row, path)
        return row

    def _set_row_path(self, row: ctk.CTkFrame, path: str) -> None:
        """Point a row at a folder, or clear it back to a placeholder."""
        name_lbl = row.grid_slaves(row=0, column=0)[0]
        path_lbl = row.grid_slaves(row=0, column=1)[0]
        if path:
            name_lbl.configure(text=Path(path).name or path)
            path_lbl.configure(text=_short(path))
            row._ep_path = path                                # type: ignore[attr-defined]
        else:
            name_lbl.configure(text="—")
            path_lbl.configure(text="(click Add folder or Browse)")
            row._ep_path = ""                                  # type: ignore[attr-defined]

    def browse_folder(self, row: ctk.CTkFrame) -> None:
        """Re-point one row at a different folder. No-op on cancel."""
        if row not in self.folder_rows:
            return
        path = filedialog.askdirectory(parent=self,
                                       title="Pick a folder to index")
        if path:
            self._set_row_path(row, path)

    def remove_folder(self, row: ctk.CTkFrame) -> None:
        if row not in self.folder_rows:
            return
        self.folder_rows.remove(row)
        row.destroy()
        for i, r in enumerate(self.folder_rows):           # reflow rows
            r.grid(row=i, column=0)
        # keep exactly one blank-ready row when the list empties
        if not self.folder_rows:
            self._new_row("")

    def clear_folders(self) -> None:
        for row in list(self.folder_rows):
            self.remove_folder(row)

    def _row_values(self, row: ctk.CTkFrame) -> tuple[str, str, float] | None:
        """(path, ext-filter, weight) for a row, None for placeholders."""
        path = getattr(row, "_ep_path", "")
        if not path:
            return None
        try:
            exts = row.grid_slaves(row=0, column=2)[0].get().strip()
        except IndexError:
            exts = ""
        try:
            weight = float(row.grid_slaves(row=0, column=3)[0].get())
        except (ValueError, IndexError):
            weight = 1.0
        return path, exts, weight

    def _sources(self) -> list[P.SourceSpec]:
        """Build SourceSpecs from the rows + project staging, skipping gaps."""
        specs: list[P.SourceSpec] = []
        for i, row in enumerate(self.folder_rows):
            vals = self._row_values(row)
            if vals is None:
                continue
            path, exts, weight = vals
            if not Path(path).is_dir():
                continue
            label = Path(path).name or f"src{i}"
            specs.append(P.SourceSpec(path=path, only=exts, label=label,
                                      weight=weight))
        if self.project_id:
            sdir = staging_path(self.project_id)
            if sdir.is_dir() and any(sdir.iterdir()):
                specs.append(P.SourceSpec(path=str(sdir), only="",
                                          label=STAGING_LABEL, weight=1.0))
        return specs

    def _save_ui_weights(self, table: str) -> None:
        """Persist the current row weights so tweaks apply without re-ingest.

        Writes even when every weight is back to 1.0, otherwise a previously
        boosted folder would keep its old multiplier in weights.json forever.
        Skips the write when nothing changed (this runs on every query/send).
        """
        specs = self._sources()
        if not table or not specs:
            return
        weights = {s.label: s.weight for s in specs if s.label}
        if not weights:
            return
        try:
            if P.load_weights(table) == weights:
                return
            P.save_weights(table, weights)
            self._say(f"weights for '{table}': " + ", ".join(
                f"{k}={v:g}" for k, v in sorted(weights.items())))
        except Exception:
            pass

    # ---------------- actions ----------------

    def do_ingest(self) -> None:
        specs = self._sources()
        if not specs:
            messagebox.showwarning("No folders", "Add at least one existing folder.")
            return
        table = self.table_var.get().strip()
        if not table:
            messagebox.showwarning("No table", "Give the index a name.")
            return
        try:
            size = int(self.size_var.get())
            overlap = int(self.overlap_var.get())
        except ValueError:
            messagebox.showwarning("Bad numbers",
                                   "Chunk size and overlap must be integers.")
            return
        if size <= 0:
            messagebox.showwarning("Bad size", "Chunk size must be positive.")
            return

        replace = self.replace_var.get()
        append = False
        if not replace and table in P.table_names():
            try:
                n = P._db().open_table(table).count_rows()
            except Exception:                              # noqa: BLE001
                n = "?"
            ans = messagebox.askyesnocancel(
                "Table exists",
                f"Table '{table}' already holds {n} rows.\n\n"
                "Yes = append new files (already indexed are skipped).\n"
                "No = rebuild from scratch (drops everything).\n"
                "Cancel = abort.")
            if ans is None:
                return
            if ans:
                append = True
            else:
                replace = True

        self._say(f"ingest: {len(specs)} folder(s) -> table '{table}'")
        self.runner.run(
            "ingest", P.ingest_sources, specs, table=table,
            model=self.model_var.get(), size=size, overlap=overlap,
            replace=replace, append=append, dry_run=False, verbose=True,
        )

    def do_embed(self) -> None:
        table = self.table_var.get().strip()
        self._say(f"embed: table '{table}'")
        self.runner.run("embed", _call, P.cmd_embed, {
            "model": self.model_var.get(), "table": table,
        })

    def do_query(self) -> None:
        q = self.query_var.get().strip()
        if not q:
            messagebox.showinfo("Empty query", "Type something to search for.")
            return
        try:
            topk = int(self.topk_var.get())
        except ValueError:
            topk = 8
        table = self.table_var.get().strip()
        self._save_ui_weights(table)
        self._say(f"query: {q!r}")
        nprobes, refine = self._tuning_live()
        self.runner.run("query", _search, q, topk,
                        table=table,
                        model=self.model_var.get(),
                        hybrid=bool(self.hybrid_var.get()),
                        nprobes=nprobes, refine=refine)

    def do_stats(self) -> None:
        self._say("stats")
        self.runner.run("stats", _stats)

    def do_index_health(self) -> None:
        """Rebuild FTS/scalar indexes + compact, optionally adding ANN."""
        table = self.table_var.get().strip()
        ann = bool(self.ann_var.get())
        if ann and not messagebox.askyesno(
                "Build vector index",
                "Also build an ANN (IVF_PQ) vector index?\n\n"
                "It is lossy and usually slower than exact search on small "
                "tables, and building it can take a while. Exact search keeps "
                "working until it exists, and nprobes/refine only take effect "
                "afterwards."):
            ann = False
        self._say(f"index: table '{table or '(all)'}'"
                  + (" + ANN" if ann else ""))
        self.runner.run("index", _index_job, table, ann)

    def _tuning_live(self) -> tuple[int, int]:
        """(nprobes, refine) from the Settings boxes as typed, no Save needed.

        Empty or junk falls back to the saved values, then to 0. Both are
        ignored by LanceDB until an ANN vector index exists.
        """
        def _int(key: str, fallback: int) -> int:
            try:
                raw = self.sv[key].get().strip()
                return max(0, int(raw)) if raw else fallback
            except (KeyError, ValueError, AttributeError):
                return fallback
        saved = _index_tuning()
        return (_int("vector_nprobes", saved[0]),
                _int("vector_refine_factor", saved[1]))

    def do_reindex(self) -> None:
        src = self.table_var.get().strip()
        model = self.model_var.get()
        dest = f"{src}__{model.replace(':', '_').replace('/', '_')}"
        if not messagebox.askyesno(
                "Re-index",
                f"Build a new table '{dest}' from '{src}' for model\n"
                f"{model}?\n\n"
                "This copies text and metadata but not vectors. You still "
                "need to run Embed afterwards."):
            return
        self._say(f"reindex: '{src}' -> '{dest}' ({model})")
        self.runner.run("reindex", _call, P.cmd_reindex, {
            "table": src, "model": model, "dest": dest, "replace": True,
        })

    def do_export_md(self) -> None:
        src = filedialog.askdirectory(parent=self,
                                      title="Folder to convert to Markdown")
        if not src:
            return
        dest = filedialog.askdirectory(parent=self,
                                       title="Output folder for .md files")
        if not dest:
            return
        self._say(f"export-md: {src} -> {dest}")
        self.runner.run("export-md", P.export_markdown, src, dest)

    def do_sync(self) -> None:
        """Incremental: re-ingest (append, deduped by source) then embed."""
        specs = self._sources()
        if not specs:
            messagebox.showwarning("No folders", "Add at least one existing folder.")
            return
        table = self.table_var.get().strip()
        if not table:
            messagebox.showwarning("No table", "Give the index a name.")
            return
        try:
            size = int(self.size_var.get())
            overlap = int(self.overlap_var.get())
        except ValueError:
            messagebox.showwarning("Bad numbers",
                                   "Chunk size and overlap must be integers.")
            return
        if size <= 0:
            messagebox.showwarning("Bad size", "Chunk size must be positive.")
            return
        self._save_ui_weights(table)
        self._say(f"sync: {len(specs)} folder(s) -> table '{table}' "
                  f"(new/changed files only)")
        self.runner.run("sync", _sync_job, specs, table=table,
                        model=self.model_var.get(), size=size, overlap=overlap)

    def do_stop(self) -> None:
        """Stop the running job and drop anything queued behind it."""
        if not self.runner.busy:
            return
        P.request_stop()
        dropped = self.runner.clear_queue()
        self._say("stop requested — finishing the current unit…"
                  + (f" ({dropped} queued job(s) dropped)" if dropped else ""))
        try:
            self.btn_stop.configure(state="disabled")
        except Exception:                                # noqa: BLE001
            pass

    # -- saved queries (per project) ------------------------------------- #

    def _refresh_saved_queries(self) -> None:
        """Rebuild the saved-query dropdown for the current project."""
        self._saved_map = {}
        labels: list[str] = []
        if self.project_id is not None:
            try:
                items = store.list_saved_queries(DB_PATH, self.project_id)
            except Exception as e:                       # noqa: BLE001
                self._say(f"!! cannot load saved queries: {e}")
                items = []
            for it in items:
                label = _short(str(it["query"]), 48)
                while label in self._saved_map:          # keep labels unique
                    label += " "
                self._saved_map[label] = (int(it["id"]), str(it["query"]))
                labels.append(label)
        values = ["(none)"] + labels
        try:
            self.saved_menu.configure(values=values)
            self.saved_var.set("(none)")
        except Exception:                                # noqa: BLE001
            pass

    def _on_saved_picked(self, label: str) -> None:
        hit = self._saved_map.get(label)
        if hit is None:
            return
        self.query_var.set(hit[1])

    def do_save_query(self) -> None:
        if self.project_id is None:
            return
        q = self.query_var.get().strip()
        if not q:
            messagebox.showinfo("Save query", "Type a query first.")
            return
        try:
            store.add_saved_query(DB_PATH, self.project_id, q)
        except Exception as e:                           # noqa: BLE001
            messagebox.showerror("Save query", str(e))
            return
        self._say(f"saved query: {_short(q, 50)}")
        self._refresh_saved_queries()

    def do_delete_saved_query(self) -> None:
        hit = self._saved_map.get(self.saved_var.get())
        if hit is None:
            messagebox.showinfo("Saved queries", "Pick a saved query first.")
            return
        sid, q = hit
        if not messagebox.askyesno("Delete query", f"Delete saved query?\n\n{q}"):
            return
        try:
            store.delete_saved_query(DB_PATH, sid)
        except Exception as e:                           # noqa: BLE001
            messagebox.showerror("Delete query", str(e))
            return
        self._say(f"deleted saved query: {_short(q, 50)}")
        self._refresh_saved_queries()

    def do_manage_sources(self) -> None:
        """List the current table's sources and delete one (chunks + FTS)."""
        table = self.table_var.get().strip()
        if not table:
            messagebox.showwarning("No table", "Give the index a name.")
            return
        try:
            sources = _list_sources(table)
        except P.PipelineError as e:
            messagebox.showerror("Sources", str(e))
            return
        if sources is None:
            messagebox.showinfo("Sources", f"table '{table}' has no chunks yet.")
            return

        dlg = ctk.CTkToplevel(self)
        dlg.title(f"Sources — {table} ({len(sources)})")
        dlg.geometry("700x480")
        dlg.transient(self)
        dlg.grab_set()
        dlg.grid_columnconfigure(0, weight=1)
        dlg.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(dlg, text="Delete removes every chunk for that source and "
                               "rebuilds the keyword index.",
                     text_color=("gray45", "gray65"), anchor="w"
                     ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))
        scroll = ctk.CTkScrollableFrame(dlg, label_text="")
        scroll.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        scroll.grid_columnconfigure(0, weight=1)

        def _del(src: str) -> None:
            if not messagebox.askyesno(
                    "Delete source",
                    f"Delete all chunks for:\n{src}\n\nThis cannot be undone.",
                    parent=dlg):
                return
            dlg.destroy()
            self._say(f"delete source: {src}")
            self.runner.run("delete-source", _delete_sources_job, table, [src])

        for i, (src, count) in enumerate(sources):
            row = ctk.CTkFrame(scroll, fg_color="transparent")
            row.grid(row=i, column=0, sticky="ew", pady=1)
            row.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(row, text=f"{count:>5}  {_short(src, 70)}",
                         anchor="w").grid(row=0, column=0, sticky="w",
                                          padx=(4, 8))
            ctk.CTkButton(row, text="✕", width=28, fg_color="transparent",
                          border_width=1, command=lambda s=src: _del(s)
                          ).grid(row=0, column=1, padx=(0, 4))

    def do_send(self) -> None:
        q = self.query_var.get().strip()
        if not q:
            return
        provider = self.provider_var.get()
        chat_model = self._chat_model_value()
        api_key = ""
        base_url = ""
        if provider == "ollama":
            if not chat_model or chat_model.startswith("("):
                messagebox.showinfo(
                    "No chat model",
                    "No chat model available.\nPull a small one first, e.g.:\n\n"
                    "  ollama pull qwen3:0.6b")
                return
        else:
            if not chat_model:
                messagebox.showinfo("No model",
                                    "Type a model name for "
                                    f"{provider} first.")
                return
            api_key = self.sv.get(
                PROVIDER_KEY_SETTING.get(provider, ""),
                ctk.StringVar(value="")).get().strip()
            if not api_key:
                messagebox.showwarning(
                    "No API key",
                    f"No {provider} key set.\nAdd it in 4 · Settings.")
                return
            if provider == "custom":
                base_url = self.sv.get(
                    "custom_base_url",
                    ctk.StringVar(value="")).get().strip()
                if not base_url:
                    messagebox.showwarning(
                        "No base URL",
                        "Set the Custom base URL in 4 · Settings, e.g.\n"
                        "  https://host/v1\n"
                        "(the app appends /chat/completions).")
                    return
        try:
            topk = int(self.topk_var.get())
        except ValueError:
            topk = 8
        table = self.table_var.get().strip()
        self._save_ui_weights(table)
        try:
            rag_enabled = self.sv.get("rag_enabled",
                                      ctk.StringVar(value="1")).get().strip() != "0"
        except Exception:                                # noqa: BLE001
            rag_enabled = True
        try:
            min_cos = float(self.sv.get("rag_min_cos",
                                        ctk.StringVar(value="0.30")).get().strip())
        except Exception:                                # noqa: BLE001
            min_cos = 0.30
        if "0.6b" in (chat_model or "").lower():
            self._append_status("note: tiny chat model -- if the answer is "
                                "weak, try llama3.2:3b or qwen2.5:1.5b")
        self.query_var.set("")
        self._append_user(q)
        self.last_query = q
        self._say(f"chat: {q!r} [{provider}/{chat_model}]")
        self._start_pending(f"working ({chat_model})")
        nprobes, refine = self._tuning_live()
        self.runner.run("chat", _chat, q, topk, table,
                        self.model_var.get(), provider, chat_model, api_key,
                        list(self.conversation), rag_enabled, min_cos,
                        bool(self.hybrid_var.get()), base_url,
                        nprobes, refine)

    def _agent_question(self) -> str:
        """Question for terminal agents: box text, else last chat question."""
        q = self.query_var.get().strip()
        if q:
            return q
        return (self.last_query or "").strip()

    def _terminal_cmd(self, key: str, fallback: str) -> str:
        try:
            cmd = self.sv.get(key, ctk.StringVar(value=fallback)).get().strip()
        except Exception:                                # noqa: BLE001
            cmd = ""
        return cmd or fallback

    def _qa_context(self, q: str) -> str | None:
        """Question (+ answer + sources) for terminal agents.

        Copies the last exchange to the clipboard when it matches `q`.
        Always allows opening -- without context if needed.
        """
        if q and self.last_query == q and (self.last_hits or self.last_answer):
            parts = [f"Question: {q}", ""]
            if self.last_answer:
                parts += [f"Answer: {self.last_answer}", ""]
            if self.last_hits:
                parts.append("Sources:")
                for i, h in enumerate(self.last_hits, 1):
                    parts.append(
                        f"[{i}] {h.get('title', '')} > {h.get('section', '')} "
                        f"({h.get('source', '')})\n{hit_body(h)}")
            self._copy("\n\n".join(parts).strip())
            return ("question + answer + sources copied "
                    "— paste them into the terminal")
        if self.last_query and self.last_answer:
            # stale but useful: offer the last exchange instead of nothing
            if messagebox.askyesno(
                    "Copy last exchange?",
                    "No fresh results for this question.\n"
                    "Copy the last Q+A instead and open?"):
                parts = [f"Question: {self.last_query}", "",
                         f"Answer: {self.last_answer}"]
                self._copy("\n\n".join(parts))
                return "last Q+A copied — paste it into the terminal"
        if messagebox.askyesno(
                "No fresh results",
                "No search results for this exact question yet.\n"
                "Open anyway (without context)?"):
            return "opened without context (Search first for sources)"
        return None

    def _terminal_folder(self) -> str:
        specs = self._sources()
        # Fall back to the app folder, never the process cwd (which can be
        # anywhere, e.g. System32 when launched from a shortcut).
        return specs[0].path if specs else str(HERE)

    def _launch_agent_terminal(self, cmd: str, display: str,
                               install_hint: str) -> bool:
        import shutil
        import subprocess

        if not shutil.which(cmd):
            messagebox.showwarning(
                f"{display} not found",
                f"The `{cmd}` CLI is not on PATH.\n{install_hint}")
            return False
        # Pass the folder as the process working directory instead of baking it
        # into the command string: a path containing `"` or `&` can no longer
        # break out of the quoting (command injection).
        folder = self._terminal_folder()
        if not Path(folder).is_dir():
            folder = str(HERE)
        try:
            subprocess.Popen(["cmd", "/c", "start", "", "cmd", "/k", cmd],
                             cwd=folder)
        except Exception as e:                          # noqa: BLE001
            messagebox.showerror("Launch failed", str(e))
            return False
        return True

    def do_opencode(self) -> None:
        q = self._agent_question()
        if not q:
            messagebox.showinfo("Empty question",
                                "Type a question in 3 · Chat first "
                                "(or send it, then reopen).")
            return
        note = self._qa_context(q)
        if note is None:
            return
        folder = self._terminal_folder()
        cmd = self._terminal_cmd("opencode_cmd", "opencode")
        if not self._launch_agent_terminal(
                cmd, "opencode", "Install it, then try again."):
            return
        self._say(f"opencode opened in {folder} ({note})")

    def do_commandcode(self) -> None:
        q = self._agent_question()
        if not q:
            messagebox.showinfo("Empty question",
                                "Type a question in 3 · Chat first "
                                "(or send it, then reopen).")
            return
        note = self._qa_context(q)
        if note is None:
            return
        folder = self._terminal_folder()
        cmd = self._terminal_cmd("commandcode_cmd", "command-code")
        if not self._launch_agent_terminal(
                cmd, "Command Code",
                "Install it with:\n\n  npm i -g command-code\n\n"
                "and sign in, then try again."):
            return
        self._say(f"command-code opened in {folder} ({note})")

    def do_copy_answer(self) -> None:
        if not self.last_answer:
            messagebox.showinfo("Nothing to copy", "No answer yet.")
            return
        self._copy(self.last_answer)

    def do_copy_all(self) -> None:
        if not self.last_hits:
            messagebox.showinfo("Nothing to copy", "No sources yet.")
            return
        parts = []
        for i, h in enumerate(self.last_hits, 1):
            parts.append(
                f"[{i}] {h.get('title', '')} > {h.get('section', '')} "
                f"({h.get('source', '')})\n{hit_body(h)}")
        self._copy("\n\n".join(parts))

    def _copy(self, text: str) -> None:
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update()
        except Exception:                                # noqa: BLE001
            self._say("!! copy to clipboard failed")
            return
        self.status_lbl.configure(text="copied ✓")

    # ---------------- result cards (newest exchange first) ----------------

    def _clear_results(self) -> None:
        self._stop_pending()
        for block in self._history:
            try:
                block.destroy()
            except Exception:                                # noqa: BLE001
                pass
        self._history = []

    def clear_chat(self) -> None:
        self._clear_results()
        self.last_hits = []
        self.last_query = ""
        self.conversation = []
        try:
            self.results_empty.pack(expand=True, pady=40)
        except Exception:                                # noqa: BLE001
            pass
        self._say("new conversation (history cleared)")

    # ---------------- chat flow (chronological, newest at bottom) --------

    def _append_block(self):
        """Append one row at the bottom of the conversation and scroll down.

        NOTE: CTkScrollableFrame redirects pack/grid into an interior frame,
        so we track our own row widgets in self._history (oldest first).
        """
        try:
            self.results_empty.pack_forget()
        except Exception:                                # noqa: BLE001
            pass
        row = ctk.CTkFrame(self.results, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        self._history.append(row)
        self._scroll_chat_bottom()
        return row

    def _scroll_chat_bottom(self) -> None:
        try:
            self.results.update_idletasks()
            self.results._parent_canvas.yview_moveto(1.0)
        except Exception:                                # noqa: BLE001
            pass

    def _append_user(self, q: str) -> None:
        """My message -- bubble on the right."""
        row = self._append_block()
        row.grid_columnconfigure(0, weight=1)
        bubble = ctk.CTkFrame(row, corner_radius=14,
                              fg_color=("#2563eb", "#3b82f6"))
        bubble.grid(row=0, column=1, sticky="e")
        ctk.CTkLabel(bubble, text=q, justify="right", anchor="e",
                     wraplength=560, font=(FONT[0], 13),
                     text_color="white").pack(padx=14, pady=8)

    def _append_assistant(self, text: str) -> None:
        """Assistant reply -- full-width markdown message, no chrome."""
        row = self._append_block()
        row.grid_columnconfigure(0, weight=1)
        body = tk.Text(row, wrap="word", height=2, font=("Segoe UI", 12),
                       relief="flat", highlightthickness=0, padx=10, pady=4)
        body.grid(row=0, column=0, sticky="ew")
        render_markdown(body, text)
        visual = sum(max(1, len(ln) // 90 + 1) for ln in text.split("\n"))
        body.configure(state="disabled", height=min(28, max(2, visual)))
        self._scroll_chat_bottom()

    def _append_status(self, note: str) -> None:
        """Small centered gray line (errors, retrieval notes)."""
        row = self._append_block()
        ctk.CTkLabel(row, text=note, text_color=("gray55", "gray65"),
                     font=(FONT[0], 11)).pack(anchor="w", padx=12)

    # ---------------- pending reply (thinking indicator + streaming) ----

    def _start_pending(self, note: str) -> None:
        """Show a live assistant row so the user sees we are not stuck.

        Small gray status line (stage + elapsed seconds) plus a plain-text
        body that streaming tokens append to. Replaced by the final
        markdown reply in _on_done.
        """
        import time as _time

        self._stop_pending()
        self._pending_note = note
        self._pending_t0 = _time.time()
        row = self._append_block()
        row.grid_columnconfigure(0, weight=1)
        status = ctk.CTkLabel(row, text=note, text_color=("gray55", "gray65"),
                              font=(FONT[0], 11), anchor="w")
        status.grid(row=0, column=0, sticky="w", padx=12)
        body = tk.Text(row, wrap="word", height=2, font=("Segoe UI", 12),
                       relief="flat", highlightthickness=0, padx=10, pady=4)
        body.grid(row=1, column=0, sticky="ew")
        body.insert("end", "…")
        body.configure(state="disabled")
        self._pending_row = row
        self._pending_status = status
        self._pending_body = body
        self._tick_pending()

    def _tick_pending(self) -> None:
        import time as _time

        if self._pending_status is None:
            self._pending_after = None
            return
        try:
            el = max(0, int(_time.time() - (self._pending_t0 or _time.time())))
            self._pending_status.configure(
                text=f"{self._pending_note} ({el}s)")
        except Exception:                                # noqa: BLE001
            pass
        try:
            self._pending_after = self.after(1000, self._tick_pending)
        except Exception:                                # noqa: BLE001
            self._pending_after = None

    def _stop_pending(self) -> None:
        if self._pending_after is not None:
            try:
                self.after_cancel(self._pending_after)
            except Exception:                            # noqa: BLE001
                pass
            self._pending_after = None
        self._pending_status = None
        self._pending_body = None
        self._pending_note = ""

    def _drop_pending_row(self) -> None:
        """Remove the pending row widget (final reply renders separately)."""
        row, hist = self._pending_row, self._history
        self._stop_pending()
        self._pending_row = None
        if row is not None and row in hist:
            try:
                hist.remove(row)
            except ValueError:
                pass
            try:
                row.destroy()
            except Exception:                            # noqa: BLE001
                pass

    def _on_chat_token(self, delta: str) -> None:
        body = self._pending_body
        if body is None or not delta:
            return
        try:
            body.configure(state="normal")
            if body.get("1.0", "end").strip() in ("", "…"):
                body.delete("1.0", "end")
            body.insert("end", delta)
            # grow a little while streaming, cap to avoid a huge widget
            try:
                lines = int(body.index("end-1c").split(".")[0])
                body.configure(height=min(28, max(2, lines)))
            except Exception:                            # noqa: BLE001
                pass
            body.configure(state="disabled")
            body.see("end")
            self._scroll_chat_bottom()
        except Exception:                                # noqa: BLE001
            pass

    def _on_chat_stage(self, note: str) -> None:
        import time as _time

        self._pending_note = note
        # restart the clock so each stage shows its own elapsed time
        self._pending_t0 = _time.time()
        if self._pending_status is not None:
            try:
                self._pending_status.configure(text=f"{note} (0s)")
            except Exception:                            # noqa: BLE001
                pass

    def _hit_badges(self, h: dict) -> str:
        score = h.get("_distance")
        if score is None:
            badges = "bm25"
        else:
            try:
                badges = f"cos={1 - float(score):.4f}"
            except (TypeError, ValueError):
                badges = f"dist={score}"
        w = h.get("_weight")
        if isinstance(w, (int, float)) and not isinstance(w, bool) and w != 1.0:
            badges += f"  w={w:g}"
        return badges

    def _render_hits(self, hits: list[dict]) -> None:
        """Search results (from tab 2 · Query) as one compact list message."""
        row = self._append_block()
        row.grid_columnconfigure(0, weight=1)
        if not hits:
            ctk.CTkLabel(row, text="No hits.",
                         text_color=("gray50", "gray60")).grid(
                row=0, column=0, sticky="w", padx=12, pady=8)
            return
        lines = []
        for i, h in enumerate(hits, 1):
            meta, body_txt = split_frontmatter(h.get("text_raw") or "")
            chips = _fm_chips(meta)
            snippet = (body_txt or (h.get("text_raw") or "")).strip()
            snippet = " ".join(snippet.split())[:260]
            lines.append(f"**[{i}] {h.get('title', '')} › "
                         f"{h.get('section', '')}**  "
                         f"`{h.get('source', '')}`  ·  {self._hit_badges(h)}"
                         + (f"\n{chips}" if chips else "")
                         + f"\n{snippet}…")
        text_w = tk.Text(row, wrap="word", height=2,
                         font=("Segoe UI", 12), relief="flat",
                         highlightthickness=0, padx=10, pady=4)
        text_w.grid(row=0, column=0, sticky="ew")
        render_markdown(text_w, "\n\n".join(lines))
        text_w.configure(state="disabled",
                         height=min(20, 2 + 3 * len(hits)))
        self._scroll_chat_bottom()

    def _project_data(self) -> dict:
        folders = []
        for row in self.folder_rows:
            vals = self._row_values(row)
            if vals is None:
                continue
            path, exts, weight = vals
            folders.append({"path": path, "only": exts, "weight": weight})
        return {"app": "embed_process", "version": PROJECT_VERSION,
                "table": self.table_var.get(), "model": self.model_var.get(),
                "chat_model": self._chat_model_value(),
                "chat_provider": self.provider_var.get(),
                "size": self.size_var.get(), "overlap": self.overlap_var.get(),
                "replace": bool(self.replace_var.get()), "folders": folders}

    def _chat_model_value(self) -> str:
        """Effective chat model: dropdown for Ollama, free text otherwise."""
        if self.provider_var.get() == "ollama":
            return self.chat_model_var.get()
        return self.chat_entry_var.get().strip()

    def _on_provider_picked(self, _name: str = "") -> None:
        self._sync_provider_widgets()

    def _sync_provider_widgets(self) -> None:
        provider = self.provider_var.get()
        show_entry = provider != "ollama"
        if show_entry:
            self.chat_menu.grid_forget()
            try:
                self.chat_entry.configure(
                    placeholder_text=PROVIDER_MODEL_HINT.get(
                        provider, "type model name"))
            except Exception:                            # noqa: BLE001
                pass
            self.chat_entry.grid(row=0, column=1, sticky="w", padx=(0, 8))
        else:
            self.chat_entry.grid_forget()
            self.chat_menu.grid(row=0, column=1, sticky="w", padx=(0, 8))

    def do_add_note(self) -> None:
        if not self.project_id:
            messagebox.showwarning("No project", "Create a project first.")
            return
        dlg = ctk.CTkToplevel(self)
        dlg.title("New note")
        dlg.geometry("560x440")
        dlg.transient(self)
        dlg.grab_set()
        dlg.grid_columnconfigure(0, weight=1)
        dlg.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(dlg, text="Title").grid(row=0, column=0, sticky="w",
                                             padx=12, pady=(10, 0))
        title_var = ctk.StringVar()
        ctk.CTkEntry(dlg, textvariable=title_var,
                     placeholder_text="Note title").grid(
            row=0, column=1, sticky="ew", padx=(0, 12), pady=(10, 0))
        dlg.grid_columnconfigure(1, weight=1)
        body = ctk.CTkTextbox(dlg, wrap="word", activate_scrollbars=True)
        body.grid(row=1, column=0, columnspan=2, sticky="nsew",
                  padx=12, pady=8)
        saved: list = []

        def _save() -> None:
            try:
                out = save_note(staging_path(self.project_id),
                                title_var.get(), body.get("1.0", "end"))
            except Exception as e:                      # noqa: BLE001
                messagebox.showerror("Save failed", str(e), parent=dlg)
                return
            saved.append(out)
            dlg.destroy()

        ctk.CTkButton(dlg, text="Save note", command=_save).grid(
            row=2, column=0, columnspan=2, pady=(0, 12))
        self.wait_window(dlg)
        if saved:
            self._say(f"note saved: {saved[0].name}")
            self._refresh_staging()

    def do_add_link(self) -> None:
        if not self.project_id:
            messagebox.showwarning("No project", "Create a project first.")
            return
        url = simpledialog.askstring("Add link", "Page URL:",
                                     parent=self)
        if not url or not url.strip():
            return
        staging = staging_path(self.project_id)
        self._say(f"fetching: {url.strip()}")
        self.runner.run("fetch", _fetch_job, url.strip(), str(staging))

    def _apply_project(self, data: dict) -> None:
        self.clear_folders()
        for f in data.get("folders", []):
            row = self._new_row(f.get("path", ""))
            try:
                entry = row.grid_slaves(row=0, column=2)[0]
                entry.delete(0, "end")
                entry.insert(0, f.get("only", ""))
                row.grid_slaves(row=0, column=3)[0].set(
                    _weight_label(f.get("weight", 1.0)))
            except Exception:                            # noqa: BLE001
                pass
        if len(self.folder_rows) > 1:
            for row in list(self.folder_rows):
                if not getattr(row, "_ep_path", ""):
                    self.remove_folder(row)
        self.table_var.set(str(data.get("table", "corpus")))
        model = str(data.get("model", P.DEFAULT_MODEL))
        try:
            if model not in self.model_menu.cget("values"):
                self.model_menu.configure(
                    values=[model, *self.model_menu.cget("values")])
        except Exception:                                # noqa: BLE001
            pass
        self.model_var.set(model)
        chat_model = str(data.get("chat_model", ""))
        provider = str(data.get("chat_provider", "ollama") or "ollama")
        if provider not in CHAT_PROVIDERS:
            provider = "ollama"
        self.provider_var.set(provider)
        self._sync_provider_widgets()
        if chat_model:
            if provider == "ollama":
                try:
                    if chat_model not in self.chat_menu.cget("values"):
                        self.chat_menu.configure(
                            values=[chat_model, *self.chat_menu.cget("values")])
                except Exception:                            # noqa: BLE001
                    pass
                self.chat_model_var.set(chat_model)
            else:
                self.chat_entry_var.set(chat_model)
        self.size_var.set(str(data.get("size", P.CHUNK_SIZE)))
        self.overlap_var.set(str(data.get("overlap", P.CHUNK_OVERLAP)))
        self.replace_var.set(bool(data.get("replace", False)))

    # ---------------- projects (SQLite) ----------------

    def _init_projects(self) -> None:
        store.init_db(DB_PATH)
        self._import_legacy_json()
        projs = store.list_projects(DB_PATH)
        if not projs:
            pid = store.create_project(
                DB_PATH, name="My first RAG", table_name="corpus",
                embed_model=P.DEFAULT_MODEL)
            projs = store.list_projects(DB_PATH)
        self._refresh_project_menu()
        first = projs[0]["id"] if projs else None
        if first is not None:
            self._load_project(first)

    def _import_legacy_json(self) -> None:
        """One-time import of projects/*.json leftovers into SQLite."""
        legacy = HERE / "projects"
        if not legacy.is_dir():
            return
        try:
            existing = {p["name"] for p in store.list_projects(DB_PATH)}
        except Exception:                                # noqa: BLE001
            return
        for fp in sorted(legacy.glob("*.json")):
            try:
                data = load_project_file(fp)
            except Exception:                            # noqa: BLE001
                continue
            name = fp.stem
            if name in existing:
                continue
            try:
                store.create_project(
                    DB_PATH, name=name,
                    table_name=str(data.get("table", store.slugify(name))),
                    embed_model=str(data.get("model", "")),
                    chat_model=str(data.get("chat_model", "")),
                    chunk_size=data.get("size", P.CHUNK_SIZE),
                    overlap=data.get("overlap", P.CHUNK_OVERLAP),
                    replace_flag=bool(data.get("replace", False)),
                    folders=data.get("folders", []))
                self._say(f"imported legacy project: {fp.name}")
            except Exception:                            # noqa: BLE001
                continue
        self._refresh_project_menu(keep_id=self.project_id)

    def _refresh_project_menu(self, keep_id: int | None = None) -> None:
        projs = store.list_projects(DB_PATH)
        self._project_names = {p["name"]: p["id"] for p in projs}
        names = [p["name"] for p in projs]
        self.project_menu.configure(values=names or ["(none)"])
        if keep_id is not None:
            for p in projs:
                if p["id"] == keep_id:
                    self._switching_project = True
                    try:
                        self.project_var.set(p["name"])
                    finally:
                        self._switching_project = False
                    return

    def _save_current_project(self) -> None:
        if self.project_id is None:
            return
        try:
            store.save_project(DB_PATH, self.project_id, self._project_data())
        except Exception as e:                          # noqa: BLE001
            self._say(f"!! project autosave failed: {e}")

    def _load_project(self, pid: int) -> None:
        data = store.get_project(DB_PATH, pid)
        if data is None:
            return
        self.project_id = pid
        self._switching_project = True
        try:
            self.project_var.set(
                next(n for n, i in self._project_names.items() if i == pid))
        except StopIteration:
            pass
        finally:
            self._switching_project = False
        self.clear_chat()
        self._apply_project(data)
        self._refresh_staging()
        self._refresh_saved_queries()

    def _on_project_picked(self, name: str) -> None:
        if self._switching_project:
            return
        pid = self._project_names.get(name)
        if pid is None or pid == self.project_id:
            return
        self._save_current_project()
        self._load_project(pid)
        self._say(f"switched to RAG project '{name}'")

    def _on_close(self) -> None:
        self._save_current_project()
        self.destroy()

    def do_new_project(self) -> None:
        name = simpledialog.askstring("New RAG project", "Project name:",
                                      parent=self)
        if not name or not name.strip():
            return
        name = name.strip()
        if name in self._project_names:
            messagebox.showwarning("Duplicate name",
                                   f"A project called '{name}' already exists.")
            return
        self._save_current_project()
        defaults = store.get_all_settings(DB_PATH)
        cur_chat = self.chat_model_var.get()
        try:
            pid = store.create_project(
                DB_PATH, name=name, table_name=store.slugify(name),
                embed_model=defaults.get("default_embed_model", "").strip()
                or self.model_var.get(),
                chat_model=defaults.get("default_chat_model", "").strip()
                or (cur_chat if not cur_chat.startswith("(") else ""),
                chat_provider=defaults.get("default_provider", "ollama"),
                chunk_size=defaults.get("chunk_size", "900"),
                overlap=defaults.get("overlap", "120"))
        except Exception as e:                          # noqa: BLE001
            messagebox.showerror("Create failed", str(e))
            return
        self._refresh_project_menu(keep_id=pid)
        self._load_project(pid)
        self._say(f"created RAG project '{name}'")

    def do_rename_project(self) -> None:
        if self.project_id is None:
            return
        old = self.project_var.get()
        name = simpledialog.askstring("Rename RAG project", "New name:",
                                      initialvalue=old, parent=self)
        if not name or not name.strip() or name.strip() == old:
            return
        name = name.strip()
        if name in self._project_names:
            messagebox.showwarning("Duplicate name",
                                   f"A project called '{name}' already exists.")
            return
        try:
            store.rename_project(DB_PATH, self.project_id, name)
        except Exception as e:                          # noqa: BLE001
            messagebox.showerror("Rename failed", str(e))
            return
        self._refresh_project_menu(keep_id=self.project_id)
        self._say(f"renamed project '{old}' -> '{name}' (table unchanged)")

    def do_delete_project(self) -> None:
        if self.project_id is None:
            return
        name = self.project_var.get()
        data = store.get_project(DB_PATH, self.project_id)
        table = (data or {}).get("table", "")
        if not messagebox.askyesno("Delete project",
                                   f"Delete RAG project '{name}'?\n\n"
                                   "Settings are removed. The index table "
                                   "is kept unless you drop it next."):
            return
        drop = False
        if table:
            try:
                exists = table in P.table_names()
            except Exception:                            # noqa: BLE001
                exists = False
            if exists:
                drop = messagebox.askyesno(
                    "Drop index table?",
                    f"Also delete the index table '{table}' "
                    f"({table} vectors will be lost)?")
                if drop:
                    try:
                        P._db().drop_table(table)
                    except Exception as e:                # noqa: BLE001
                        messagebox.showerror("Drop failed", str(e))
                        return
        try:
            store.delete_project(DB_PATH, self.project_id)
        except Exception as e:                          # noqa: BLE001
            messagebox.showerror("Delete failed", str(e))
            return
        try:
            import shutil as _shutil
            _shutil.rmtree(staging_path(self.project_id),
                           ignore_errors=True)
        except Exception:                                # noqa: BLE001
            pass
        self.project_id = None
        self._say(f"deleted project '{name}'"
                  + (f" + table '{table}'" if drop else ""))
        projs = store.list_projects(DB_PATH)
        self._refresh_project_menu()
        if projs:
            self._load_project(projs[0]["id"])
        else:
            pid = store.create_project(DB_PATH, name="My first RAG",
                                       table_name="corpus",
                                       embed_model=P.DEFAULT_MODEL)
            self._refresh_project_menu(keep_id=pid)
            self._load_project(pid)

    # ---------------- settings ----------------

    def _load_settings(self) -> None:
        vals = store.get_all_settings(DB_PATH)
        for k, v in vals.items():
            if k in self.sv:
                self.sv[k].set(v)
        host = vals.get("ollama_host", "").strip()
        if host:
            os.environ.setdefault("OLLAMA_HOST", host)
        if vals.get("top_k", "").strip().isdigit():
            self.topk_var.set(vals["top_k"].strip())
        try:
            self.hybrid_var.set(vals.get("hybrid_search", "1").strip() != "0")
        except Exception:                                # noqa: BLE001
            pass
        if vals.get("default_provider", "") in CHAT_PROVIDERS:
            self.provider_var.set(vals["default_provider"])
            self._sync_provider_widgets()

    def _on_hybrid_setting(self) -> None:
        """Keep the Chat-tab checkbox in sync with the Settings default."""
        try:
            on = self.sv["hybrid_search"].get().strip() != "0"
        except Exception:                                # noqa: BLE001
            return
        try:
            self.hybrid_var.set(on)
        except Exception:                                # noqa: BLE001
            pass

    def do_save_settings(self) -> None:
        try:
            self.sv["hybrid_search"].set(
                "1" if self.hybrid_var.get() else "0")
        except Exception:                                # noqa: BLE001
            pass
        for k in ("chunk_size", "overlap", "top_k"):
            v = self.sv[k].get().strip()
            if not v.isdigit() or int(v) <= 0:
                messagebox.showwarning(
                    "Bad number", f"{k} must be a positive integer.")
                return
        try:
            mc = float(self.sv["rag_min_cos"].get().strip())
        except (KeyError, ValueError, AttributeError):
            mc = -1.0
        if not 0.0 <= mc <= 1.0:
            messagebox.showwarning(
                "Bad threshold",
                "Min similarity must be a number between 0 and 1.")
            return
        for k in ("opencode_cmd", "commandcode_cmd"):
            if k in self.sv and not self.sv[k].get().strip():
                messagebox.showwarning(
                    "Bad command", f"{k} must not be empty.")
                return
        store.set_all_settings(DB_PATH,
                               {k: v.get() for k, v in self.sv.items()})
        host = self.sv["ollama_host"].get().strip()
        if host:
            os.environ["OLLAMA_HOST"] = host
        elif "OLLAMA_HOST" in os.environ:
            del os.environ["OLLAMA_HOST"]
        self._say("settings saved")
        self.probe_ollama()

    def do_test_ollama(self) -> None:
        import urllib.request

        host = self.sv["ollama_host"].get().strip() or os.environ.get(
            "OLLAMA_HOST", "http://localhost:11434")
        self.ollama_test_lbl.configure(text="testing…")

        def _work() -> None:
            try:
                with urllib.request.urlopen(host.rstrip("/") + "/api/tags",
                                            timeout=5) as r:
                    import json as _json
                    n = len(_json.load(r).get("models", []))
                BUS.put("ollama_test", True, f"OK ({n} models at {host})")
            except Exception as e:                          # noqa: BLE001
                BUS.put("ollama_test", False, f"unreachable: {e}")

        threading.Thread(target=_work, daemon=True,
                         name="ollama-test").start()

    def _test_model_for(self, provider: str) -> str:
        """Which model name the Settings 'Test' button should use.

        Prefer whatever the Chat tab is pointed at for that same provider
        (that is exactly what Send would use); otherwise fall back to the
        default chat model from the Defaults section.
        """
        if self.provider_var.get() == provider:
            m = self._chat_model_value()
            if m and not m.startswith("("):
                return m
        try:
            return self.sv["default_chat_model"].get().strip()
        except (KeyError, AttributeError):
            return ""

    def do_test_provider(self, provider: str) -> None:
        """Fire one tiny completion at a hosted provider.

        Proves key + model + base URL end-to-end (including gateway
        redirects). Network runs on a worker thread; the result lands in
        `provider_test_lbl` through BUS. It reads the entry boxes as they
        are now, so no need to Save first.
        """
        key = self.sv.get(PROVIDER_KEY_SETTING.get(provider, ""),
                          ctk.StringVar(value="")).get().strip()
        base_url = ""
        if provider == "custom":
            base_url = self.sv.get("custom_base_url",
                                   ctk.StringVar(value="")).get().strip()
        model = self._test_model_for(provider)

        def _bad(msg: str) -> None:
            self.provider_test_lbl.configure(
                text=f"{provider}: {msg}",
                text_color=("#c62828", "#ef9a9a"))

        if not key:
            _bad("no API key yet — paste one above, then Test.")
            return
        if provider == "custom" and not base_url:
            _bad("no base URL yet — fill 'Custom base URL', then Test.")
            return
        if not model:
            _bad("no model to test with — set 'Chat model' in Defaults, or "
                 "pick this provider in Chat and type a model, then Test.")
            return

        self.provider_test_lbl.configure(
            text=f"{provider}: testing {model}…",
            text_color=("gray45", "gray65"))

        def _work() -> None:
            import chat_providers as C
            try:
                msg = C.test_connection(provider, model=model, api_key=key,
                                        base_url=base_url)
                BUS.put("provider_test", provider, True, msg)
            except C.ProviderError as e:
                BUS.put("provider_test", provider, False, str(e))
            except Exception as e:                          # noqa: BLE001
                BUS.put("provider_test", provider, False,
                        f"{type(e).__name__}: {e}")

        threading.Thread(target=_work, daemon=True,
                         name=f"test-{provider}").start()

    def do_list_models(self, provider: str) -> None:
        """Fetch a provider's model/instrument list into the read-only box.

        The Grid's catalog is public; the others need the key. Runs on a
        worker thread and renders through BUS (`provider_models`). OpenCode
        is listed from its gateway catalog (`/inference/v1/models`), not the
        chat surface, which has no /models.
        """
        import chat_providers as C

        key = self.sv.get(PROVIDER_KEY_SETTING.get(provider, ""),
                          ctk.StringVar(value="")).get().strip()
        base_url = ""
        if provider == "custom":
            base_url = self.sv.get("custom_base_url",
                                   ctk.StringVar(value="")).get().strip()

        def _bad(msg: str) -> None:
            self.provider_models_lbl.configure(
                text=f"{provider}: {msg}",
                text_color=("#c62828", "#ef9a9a"))

        if provider not in C.MODELS_PUBLIC and not key:
            _bad("no API key yet — paste one above, then Models.")
            return
        if provider == "custom" and not base_url:
            _bad("no base URL yet — fill 'Custom base URL', then Models.")
            return

        self.provider_models_lbl.configure(
            text=f"{provider}: loading models…",
            text_color=("gray45", "gray65"))

        def _work() -> None:
            try:
                rows = C.list_models(provider, api_key=key, base_url=base_url)
                BUS.put("provider_models", provider, True, rows)
            except C.ProviderError as e:
                BUS.put("provider_models", provider, False, str(e))
            except Exception as e:                          # noqa: BLE001
                BUS.put("provider_models", provider, False,
                        f"{type(e).__name__}: {e}")

        threading.Thread(target=_work, daemon=True,
                         name=f"models-{provider}").start()

    def _render_provider_models(self, provider: str, rows: list) -> None:
        """Fill the read-only models box -- each line is a usable model id."""
        self.provider_models_lbl.configure(
            text=f"{provider}: {len(rows)} model(s) — copy an id into the "
                 "Chat model box", text_color=("#2e7d32", "#81c784"))
        box = self.provider_models_box
        box.configure(state="normal")
        box.delete("1.0", "end")
        if provider == "opencode":
            box.insert("end", "# only models on OpenCode's "
                              "OpenAI/Chat-Completions surface work here\n")
        for mid, name in rows:
            box.insert("end", mid if (not name or name == mid)
                       else f"{mid}   # {name}")
            box.insert("end", "\n")
        box.configure(state="disabled")

    def do_test_terminal(self, key: str) -> None:
        import shutil

        cmd = (self.sv.get(key, ctk.StringVar(value="")).get().strip()
               if hasattr(self, "sv") else "")
        if not cmd:
            self.term_test_lbl.configure(text=f"{key}: empty command")
            return
        found = shutil.which(cmd)
        if found:
            self.term_test_lbl.configure(text=f"{cmd}: OK ({found})")
        else:
            self.term_test_lbl.configure(
                text=f"{cmd}: not on PATH")

    def _refresh_models(self) -> None:
        import urllib.request

        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.models_lbl.configure(text="loading…")

        def _work() -> None:
            try:
                with urllib.request.urlopen(host.rstrip("/") + "/api/tags",
                                            timeout=8) as r:
                    import json as _json
                    models = _json.load(r).get("models", [])
                rows = sorted(
                    ((m.get("name", "?"),
                      m.get("size", 0)) for m in models),
                    key=lambda t: t[0])
                BUS.put("models_list", True, rows)
            except Exception as e:                          # noqa: BLE001
                BUS.put("models_list", False, str(e))

        threading.Thread(target=_work, daemon=True,
                         name="ollama-models").start()

    def _render_models(self, rows: list) -> None:
        self.models_box.configure(state="normal")
        self.models_box.delete("1.0", "end")
        total = 0
        for name, size in rows:
            total += size or 0
            self.models_box.insert(
                "end", f"{name:<42} {size / 1e9:6.2f} GB\n")
        self.models_box.configure(state="disabled")
        self.models_lbl.configure(
            text=f"{len(rows)} model(s), {total / 1e9:.1f} GB")

    def do_pull_model(self) -> None:
        name = self.model_dl_var.get().strip()
        if not name:
            messagebox.showinfo("Empty name", "Type a model name first.")
            return
        self._say(f"pull: {name} (streams progress below)")
        self.runner.run("pull", _pull_job, name)

    def do_delete_model(self) -> None:
        name = self.model_dl_var.get().strip()
        if not name:
            messagebox.showinfo("Empty name", "Type a model name first.")
            return
        if not messagebox.askyesno("Delete model",
                                   f"Delete '{name}' from Ollama?\n\n"
                                   "Tables built with it become unusable "
                                   "until you pull it again."):
            return
        self.runner.run("delete-model", _delete_job, name)

    def _include_keys(self) -> bool:
        try:
            return self.sv.get(
                "backup_include_keys", ctk.StringVar(value="0")
            ).get().strip() == "1"
        except Exception:                                # noqa: BLE001
            return False

    def do_backup_zip(self) -> None:
        """Full bundle: app.db + index/ + staging/ in one .zip."""
        self._save_current_project()
        path = filedialog.asksaveasfilename(
            parent=self, title="Full backup bundle (app.db + index + staging)",
            initialdir=str(HERE), defaultextension=".zip",
            filetypes=[("Zip bundle", "*.zip")],
            initialfile="embed_process-backup.zip")
        if not path:
            return
        include_keys = self._include_keys()
        self._say(f"backup bundle -> {path} "
                  f"(API keys {'included' if include_keys else 'excluded'})")
        self.runner.run("backup", _backup_job, path,
                        include_staging=True, include_secrets=include_keys)

    def do_restore_zip(self) -> None:
        """Restore a full bundle, snapshotting the current data first."""
        path = filedialog.askopenfilename(
            parent=self, title="Restore from bundle",
            initialdir=str(HERE), filetypes=[("Zip bundle", "*.zip")])
        if not path:
            return
        try:
            manifest = backup.read_manifest(path)
        except backup.BackupError as e:
            messagebox.showerror("Not a bundle", str(e))
            return
        counts = manifest.get("counts") or {}
        if not messagebox.askyesno(
                "Restore bundle?",
                "Replace ALL current data (settings, projects, index, "
                "staging) with this bundle?\n\n"
                f"{counts.get('projects', 0)} project(s), "
                f"{counts.get('tables', 0)} table(s), "
                f"{counts.get('staging_files', 0)} staging file(s).\n"
                f"Created: {manifest.get('created_at_readable', '?')}\n\n"
                "A snapshot of the current data is saved to backups/ first."):
            return
        self.project_id = None
        self._say(f"restoring bundle: {path}")
        self.runner.run("restore", _restore_job, path, snapshot=True)

    def _after_restore(self, summary: dict) -> None:
        """Reload the UI against the freshly restored app.db + index."""
        try:
            store.init_db(DB_PATH)
        except Exception as e:                          # noqa: BLE001
            self._say(f"!! post-restore init failed: {e}")
        self.project_id = None
        self._load_settings()
        self._refresh_project_menu()
        projs = store.list_projects(DB_PATH)
        if projs:
            self._load_project(projs[0]["id"])
        else:
            pid = store.create_project(DB_PATH, name="My first RAG",
                                       table_name="corpus",
                                       embed_model=P.DEFAULT_MODEL)
            self._refresh_project_menu(keep_id=pid)
            self._load_project(pid)
        self.probe_ollama()
        self._refresh_models()
        snap = (summary or {}).get("snapshot")
        self._say(f"✓ restored bundle ({summary.get('projects', 0)} project(s))"
                  + (f" — snapshot: {snap}" if snap else ""))

    def do_export_project(self) -> None:
        """Write the current project (table + staging) to its own .zip."""
        if self.project_id is None:
            messagebox.showinfo("Export project", "No project selected.")
            return
        self._save_current_project()
        name = self.project_var.get().strip() or "project"
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "project"
        path = filedialog.asksaveasfilename(
            parent=self, title="Export current project",
            initialdir=str(HERE), defaultextension=".zip",
            filetypes=[("Project package", "*.zip")],
            initialfile=f"project-{safe}.zip")
        if not path:
            return
        self._say(f"export project '{name}' -> {path}")
        self.runner.run("export-project", _export_project_job, path,
                        pid=self.project_id, include_staging=True)

    def do_import_project(self) -> None:
        """Add a project package to the live data (append, never replace)."""
        path = filedialog.askopenfilename(
            parent=self, title="Import a project package",
            initialdir=str(HERE), filetypes=[("Project package", "*.zip")])
        if not path:
            return
        try:
            manifest = backup.read_manifest(path)
        except backup.BackupError as e:
            messagebox.showerror("Not a package", str(e))
            return
        counts = manifest.get("counts") or {}
        proj = manifest.get("project") or {}
        if manifest.get("kind") != backup.PROJECT_KIND:
            messagebox.showerror(
                "Wrong kind",
                "That is a full backup bundle, not a single-project package.\n"
                "Use 'Restore…' under Full bundle instead.")
            return
        if not messagebox.askyesno(
                "Import project?",
                f"Add project '{proj.get('name', '?')}' (table "
                f"'{proj.get('table', '?')}') to this installation?\n\n"
                f"{counts.get('files', 0)} file(s), "
                f"{counts.get('staging_files', 0)} staging file(s).\n"
                "Created: " + str(manifest.get("created_at_readable", "?")) +
                "\n\nNothing existing is overwritten — a clashing name gets "
                "an '(imported)' suffix. A snapshot is saved to backups/ first."):
            return
        self._say(f"importing project: {path}")
        self.runner.run("import-project", _import_project_job, path,
                        snapshot=True)

    def _after_project_import(self, summary: dict) -> None:
        """Reload menus and jump to the freshly imported project."""
        try:
            store.init_db(DB_PATH)
        except Exception as e:                          # noqa: BLE001
            self._say(f"!! post-import init failed: {e}")
        self._save_current_project()
        self._refresh_project_menu()
        pid = summary.get("pid")
        if isinstance(pid, int):
            self._load_project(pid)
        self._refresh_staging()
        note = ""
        if summary.get("table_renamed"):
            note = f" (table renamed to '{summary.get('table')}')"
        elif summary.get("renamed"):
            note = f" (renamed to '{summary.get('name')}')"
        self._say(f"✓ imported project '{summary.get('name')}'{note}"
                  + (f" — snapshot: {summary.get('snapshot')}"
                     if summary.get("snapshot") else ""))

    def do_open_data_folder(self) -> None:
        """Open the folder holding index/, app.db and staging/."""
        try:
            os.startfile(str(HERE))                      # Windows
        except Exception as e:                          # noqa: BLE001
            messagebox.showerror("Open failed", str(e))

    def do_backup(self) -> None:
        self._save_current_project()
        path = filedialog.asksaveasfilename(
            parent=self, title="Backup settings + projects",
            initialdir=str(HERE), defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="embed_process-backup.json")
        if not path:
            return
        try:
            data = {"app": "embed_process", "kind": "backup", "version": 1,
                    "settings": store.get_all_settings(DB_PATH),
                    "weights": _read_weights_file(),
                    "projects": [
                        store.get_project(DB_PATH, p["id"])
                        for p in store.list_projects(DB_PATH)]}
            Path(path).write_text(json.dumps(data, indent=2,
                                             ensure_ascii=False),
                                   encoding="utf-8")
        except Exception as e:                              # noqa: BLE001
            messagebox.showerror("Backup failed", str(e))
            return
        self._say(f"backup written: {path} "
                  f"({len(data['projects'])} projects)")

    def do_restore(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="Restore from backup",
            initialdir=str(HERE), filetypes=[("JSON files", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:                              # noqa: BLE001
            messagebox.showerror("Restore failed",
                                 f"cannot read file: {e}")
            return
        if not isinstance(data, dict) or not isinstance(
                data.get("projects"), list):
            messagebox.showerror("Restore failed", "not a backup file")
            return
        projs = data["projects"]
        if not messagebox.askyesno(
                "Restore backup?",
                f"Replace ALL current settings and {len(projs)} project(s) "
                f"from backup?\n\nCurrent projects will be deleted "
                f"(index tables are kept)."):
            return
        try:
            for p in store.list_projects(DB_PATH):
                store.delete_project(DB_PATH, p["id"])
            if isinstance(data.get("settings"), dict):
                store.set_all_settings(DB_PATH, data["settings"])
            if isinstance(data.get("weights"), dict):
                P.weights_file().parent.mkdir(parents=True, exist_ok=True)
                P.weights_file().write_text(
                    json.dumps(data["weights"], indent=2), encoding="utf-8")
            for p in projs:
                if not isinstance(p, dict) or not str(p.get("name", "")).strip():
                    continue
                store.create_project(
                    DB_PATH, name=str(p["name"]).strip(),
                    table_name=str(p.get("table", "")),
                    embed_model=str(p.get("model", "")),
                    chat_model=str(p.get("chat_model", "")),
                    chat_provider=str(p.get("chat_provider", "ollama")),
                    chunk_size=p.get("size", 900),
                    overlap=p.get("overlap", 120),
                    replace_flag=bool(p.get("replace", False)),
                    folders=p.get("folders", []))
        except Exception as e:                              # noqa: BLE001
            messagebox.showerror("Restore failed", str(e))
            return
        self.project_id = None
        self._load_settings()
        self._refresh_project_menu()
        projs = store.list_projects(DB_PATH)
        if projs:
            self._load_project(projs[0]["id"])
        self._say(f"restored {len(projs)} project(s) from backup")

    # ---------------- ollama probe ----------------

    def probe_ollama(self) -> None:
        """Check the Ollama daemon and populate the model dropdown."""
        import json
        import urllib.request

        def _work() -> None:
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            try:
                with urllib.request.urlopen(host.rstrip("/") + "/api/tags",
                                            timeout=3) as r:
                    data = json.load(r)
                models = sorted(m["name"] for m in data.get("models", []))
                BUS.put("ollama", True, models)
            except Exception:                              # noqa: BLE001
                BUS.put("ollama", False, [])

        threading.Thread(target=_work, daemon=True, name="ollama-probe").start()

    # ---------------- UI plumbing ----------------

    def _say(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def set_busy(self, busy: bool) -> None:
        """Toggle Stop while a job runs; leave everything else clickable.

        This used to disable every job button and the project switcher, because
        a second click would have raced the first job. `Runner` now *queues*
        whatever you start, so nothing runs concurrently and you can keep
        working -- even switch projects -- during a long embed. Only Stop needs
        a state.
        """
        stop = getattr(self, "btn_stop", None)
        if stop is not None:
            try:
                stop.configure(state="normal" if busy else "disabled")
            except Exception:                            # noqa: BLE001
                pass
        self._update_queue_badge()

    def _update_queue_badge(self) -> None:
        """Show how many jobs are waiting, on the Stop button.

        A queue only exists while a job runs, and Stop is the one control that
        is live then -- so it is the natural place for the count.
        """
        stop = getattr(self, "btn_stop", None)
        if stop is None:
            return
        try:
            n = self.runner.pending
            stop.configure(text=f"Stop ({n})" if n else "Stop")
        except Exception:                                # noqa: BLE001
            pass

    def set_progress(self, done: int, total: int, title: str = "") -> None:
        if total and total > 0:
            self.progress.set(max(0.0, min(1.0, done / total)))
            self.status_lbl.configure(
                text=f"{title}  {done}/{total}" if title else f"{done}/{total}")
        else:
            self.status_lbl.configure(text=title or "working…")

    def _drain(self) -> None:
        """Pull worker messages onto the UI thread.

        Everything is guarded: a single bad message must not kill the refresh
        loop, or the UI would freeze with no visible error.
        """
        try:
            while True:
                kind, payload = BUS.get_nowait()
                try:
                    self._handle(kind, payload)
                except Exception as e:                    # noqa: BLE001
                    # never let one bad message stop the pump
                    self._say(f"!! UI error handling {kind}: {e}")
        except queue.Empty:
            pass
        except Exception as e:                            # noqa: BLE001
            self._say(f"!! queue error: {e}")
        self.after(80, self._drain)

    def _handle(self, kind: str, payload: tuple) -> None:
        if kind == "log":
            self._say(payload[0])
        elif kind == "progress":
            self.set_progress(*payload)
        elif kind == "ollama":
            ok, models = payload
            if ok:
                self.dot.configure(text_color="#2e7d32")
                emb = [m for m in models if "embed" in m.lower()]
                chat = [m for m in models if "embed" not in m.lower()]
                self.chat_models = chat
                self.ollama_lbl.configure(
                    text=f"Ollama online · {len(emb)} embedding · "
                         f"{len(chat)} chat · {len(models)} total")
                if emb:
                    self.model_menu.configure(values=emb)
                    if self.model_var.get() not in emb:
                        self.model_var.set(emb[0])
                if chat:
                    self.chat_menu.configure(values=chat)
                    if self.chat_model_var.get() not in chat:
                        self.chat_model_var.set(chat[0])
                else:
                    self.chat_menu.configure(values=["(no chat model)"])
                    self.chat_model_var.set("(no chat model)")
            else:
                self.dot.configure(text_color="#c62828")
                self.chat_models = []
                self.ollama_lbl.configure(
                    text="Ollama not reachable — start it with "
                         "`ollama serve` (KV cache: q8_0)")
        elif kind == "ollama_test":
            ok, text = payload
            self.ollama_test_lbl.configure(text=text)
            self.dot.configure(text_color="#2e7d32" if ok else "#c62828")
        elif kind == "provider_test":
            provider, ok, text = payload
            lbl = getattr(self, "provider_test_lbl", None)
            if lbl is not None:
                lbl.configure(
                    text=f"{provider}: {text}",
                    text_color=("#2e7d32", "#81c784") if ok
                    else ("#c62828", "#ef9a9a"))
        elif kind == "provider_models":
            provider, ok, text = payload
            if ok:
                self._render_provider_models(provider, text)
            else:
                self.provider_models_lbl.configure(
                    text=f"{provider}: {text}",
                    text_color=("#c62828", "#ef9a9a"))
        elif kind == "models_list":
            ok, payload2 = payload
            if ok:
                self._render_models(payload2)
            else:
                self.models_lbl.configure(text=f"failed: {payload2}")
        elif kind == "done":
            self._on_done(payload[0])
        elif kind == "chat_token":
            self._on_chat_token(payload[0] if payload else "")
        elif kind == "chat_stage":
            self._on_chat_stage(payload[0] if payload else "working")
        elif kind == "stopped":
            self._say("■ stopped by user")
            if getattr(self, "_last_job", "") == "chat":
                self._drop_pending_row()
                self._append_status("stopped")
        elif kind == "failed":
            if getattr(self, "_last_job", "") == "chat":
                self._drop_pending_row()
                self._append_status(f"⚠ {payload[0]}")
            else:
                messagebox.showerror("Pipeline error", payload[0])
        elif kind == "crashed":
            self._say(f"!! {payload[0]}")
            if getattr(self, "_last_job", "") == "chat":
                self._drop_pending_row()
                self._append_status(f"⚠ {payload[0]}")
            else:
                messagebox.showerror("Unexpected error", payload[0])
        elif kind == "finished":
            # finish() either starts the next queued job or goes idle, and it
            # owns the unlock -- resetting the UI here would unlock the window
            # while the next job is already running.
            self.runner.finish()

    def _on_done(self, result) -> None:
        if isinstance(result, dict):
            # backup / restore jobs carry their own kind marker
            if result.get("kind") == backup.BUNDLE_KIND:
                counts = result.get("counts") or {}
                self._say(f"✓ backup bundle written — {counts.get('tables', 0)} "
                          f"table(s), {counts.get('staging_files', 0)} staging "
                          f"file(s), keys "
                          f"{'included' if result.get('include_secrets') else 'excluded'}")
                messagebox.showinfo(
                    "Backup complete",
                    f"Bundle written:\n{result.get('path')}\n\n"
                    f"{counts.get('projects', 0)} project(s), "
                    f"{counts.get('tables', 0)} index table(s).")
                return
            if result.get("kind") == backup.PROJECT_KIND:
                counts = result.get("counts") or {}
                proj = result.get("project") or {}
                self._say(f"✓ project package written — "
                          f"{counts.get('files', 0)} file(s), "
                          f"{counts.get('staging_files', 0)} staging file(s)")
                messagebox.showinfo(
                    "Export complete",
                    f"Project '{proj.get('name', '?')}' exported to:\n"
                    f"{result.get('path')}\n\n"
                    f"{'with' if result.get('has_table') else 'without'} its "
                    f"index table.")
                return
            if result.get("kind") == "project-import":
                self._after_project_import(result)
                messagebox.showinfo("Import complete",
                                    f"Project '{result.get('name')}' imported.")
                return
            if result.get("kind") == "restore":
                self._after_restore(result)
                messagebox.showinfo("Restore complete", "Bundle restored.")
                return
            self.last_summary = result
            ctx = getattr(self, "_job_ctx", None) or {}
            # A job queued for another project must not paint into this one:
            # the user switched projects while it ran, so log it instead of
            # dropping the answer/hits into the wrong chat or card pane.
            foreign = (ctx.get("pid") is not None
                       and ctx.get("pid") != self.project_id)
            if "answer" in result:
                if foreign:
                    self._say(f"✓ '{ctx.get('title')}' finished for another "
                              f"project — result not shown here")
                    return
                self.last_hits = result.get("hits", [])
                self.last_query = result.get("query", "")
                self.last_answer = result.get("answer", "")
                self.conversation = result.get("history", [])[-6:]
                self._drop_pending_row()
                self._append_assistant(self.last_answer)
                n = len(self.last_hits)
                if result.get("from_corpus") and n:
                    self.status_lbl.configure(
                        text=f"grounded in {n} chunk(s) ✓")
                else:
                    self.status_lbl.configure(
                        text="general answer (RAG skipped)")
            elif "hits" in result:
                if foreign:
                    self._say(f"✓ query finished for another project — "
                              f"{len(result.get('hits') or [])} hit(s) not "
                              f"shown here")
                    return
                self.last_hits = result.get("hits", [])
                self.last_query = result.get("query", "")
                self.last_answer = ""          # a raw Query is not a chat answer
                self._render_hits(result["hits"])
            elif result.get("sync"):
                self._say(f"✓ sync complete — {result.get('chunks', 0)} new "
                          f"chunk(s), {result.get('embedded', 0)} embedded")
                self.btn_embed.configure(state="normal")
            elif "index" in result:
                rows = result.get("index") or []
                self._say(f"✓ index maintenance done — {len(rows)} table(s)")
                for s in rows:
                    self._say(f"    {s.get('table')}: fragments "
                              f"{s.get('before_fragments')} -> "
                              f"{s.get('after_fragments')}, indices: "
                              f"{', '.join(s.get('indices') or []) or 'none'}")
            elif "deleted_chunks" in result:
                self._say(f"✓ deleted {result.get('deleted_chunks', 0)} chunk(s) "
                          f"({result.get('before')} -> {result.get('after')})")
            elif "chunks" in result:
                if result.get("appended"):
                    self._say(f"✓ ingest complete — {result['chunks']} new chunks "
                              f"appended ({result.get('skipped_existing', 0)} "
                              f"already indexed)")
                else:
                    self._say(f"✓ ingest complete — {result['chunks']} chunks "
                              f"from {result.get('files', '?')} files")
                # vectors are the next step, so enable it
                self.btn_embed.configure(state="normal")
            elif "embedded" in result:
                self._say(f"✓ embed complete — {result.get('embedded', 0)} rows, "
                          f"dims={result.get('dims')}")
            elif "md_file" in result:
                self._refresh_staging()
            elif "pulled" in result or "deleted" in result:
                self.probe_ollama()
                self._refresh_models()


# --------------------------------------------------------------------------- #
# helpers called on the worker thread
# --------------------------------------------------------------------------- #

def _read_weights_file() -> dict:
    try:
        return json.loads(P.weights_file().read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return {}


def _pull_job(model: str) -> dict:
    """Pull an Ollama model, streaming progress to the log."""
    client = P._client()
    try:
        client.show(model)
        P.log(f"'{model}' already present")
        return {"pulled": model}
    except Exception:
        pass
    try:
        for st in client.pull(model, stream=True):
            if isinstance(st, dict):
                status = st.get("status", "")
                total = st.get("total") or 0
                done = st.get("completed") or 0
                if total:
                    P.log(f"  {status} {done * 100 // total}%")
                elif status:
                    P.log(f"  {status}")
    except Exception as e:                              # noqa: BLE001
        raise P.PipelineError(f"pull failed: {e}")
    P.log(f"pulled '{model}'")
    return {"pulled": model}


def _delete_job(model: str) -> dict:
    client = P._client()
    try:
        client.delete(model)
    except Exception as e:                              # noqa: BLE001
        raise P.PipelineError(f"delete failed: {e}")
    P.log(f"deleted '{model}'")
    return {"deleted": model}


def _backup_job(path: str, **kw) -> dict:
    """Write a full bundle; surface BackupError as a PipelineError."""
    try:
        return backup.export_bundle(path, **kw)
    except backup.BackupError as e:
        raise P.PipelineError(str(e))


def _restore_job(path: str, **kw) -> dict:
    """Restore a full bundle; surface BackupError as a PipelineError."""
    try:
        return backup.import_bundle(path, **kw)
    except backup.BackupError as e:
        raise P.PipelineError(str(e))


def _export_project_job(path: str, **kw) -> dict:
    """Write a single-project package; surface BackupError as a PipelineError."""
    try:
        return backup.export_project(path, **kw)
    except backup.BackupError as e:
        raise P.PipelineError(str(e))


def _import_project_job(path: str, **kw) -> dict:
    """Add a project package; surface BackupError as a PipelineError."""
    try:
        return backup.import_project(path, **kw)
    except backup.BackupError as e:
        raise P.PipelineError(str(e))


def _sync_job(specs: list, *, table: str, model: str, size: int,
              overlap: int) -> dict:
    """Incremental sync: append-ingest (dedupes unchanged files by source
    hash) and then embed whatever is still pending. Safe to run repeatedly.
    """
    P.clear_stop()
    summary = P.ingest_sources(specs, table=table, model=model, size=size,
                               overlap=overlap, append=True, verbose=True)
    summary["sync"] = True
    summary["embedded"] = 0
    if int(summary.get("chunks") or 0):
        res = P.cmd_embed(_Args(model=model, table=table))
        summary["embedded"] = int(res.get("embedded") or 0)
    return summary


def _delete_sources_job(table: str, sources: list) -> dict:
    """Remove a source (file or folder prefix) from a table."""
    res = P.delete_sources(table, list(sources))
    return {"deleted_chunks": int(res.get("deleted") or 0),
            "before": res.get("before"), "after": res.get("after"),
            "sources": res.get("sources")}


def _list_sources(table: str) -> list | None:
    """Distinct sources in `table` with their chunk counts, sorted by name.

    Returns None when the table is empty. Raises PipelineError on any DB
    error so the GUI can show a clean message.
    """
    from collections import Counter
    try:
        t = P._db().open_table(table)
    except Exception as e:                               # noqa: BLE001
        raise P.PipelineError(f"cannot open table '{table}': {e}")
    try:
        rows = t.search().select(["source"]).to_list()
    except Exception as e:                               # noqa: BLE001
        raise P.PipelineError(f"cannot read table '{table}': {e}")
    if not rows:
        return None
    counts = Counter(str(r.get("source", "")) for r in rows)
    return sorted(counts.items())


class _Args:
    """Minimal stand-in for argparse.Namespace."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _call(fn, kwargs: dict):
    """Run a `cmd_*` function with a namespace."""
    return fn(_Args(**kwargs))


def _index_tuning() -> tuple[int, int]:
    """(nprobes, refine_factor) from Settings, tolerant of junk values.

    Both are 0 (off) by default; they only do anything once an ANN vector
    index exists, and LanceDB ignores them otherwise.
    """
    def _int(key: str) -> int:
        try:
            return max(0, int(store.get_setting(DB_PATH, key, "0") or 0))
        except (TypeError, ValueError):
            return 0
    return _int("vector_nprobes"), _int("vector_refine_factor")


def _retrieve(q: str, topk: int, table: str, model: str,
               hybrid: bool = True, *, nprobes: int | None = None,
               refine: int | None = None):
    """Shared retrieval for Search and Ask: guards, embed, cosine search,
    folder weights, FTS/RRF fusion. Returns (hits, meta, model_used).

    hybrid=False skips the FTS half entirely (vector-only), so exact-term
    misses and missing FTS indexes behave identically: pure cosine ranking.
    `nprobes`/`refine` default to the saved Settings values; callers that
    have live widgets pass them explicitly.
    """
    table_obj = P._open_table(table)
    meta = P._meta_of(table_obj)
    if not meta.get("model_tag"):
        raise P.PipelineError(
            f"table '{table}' has no embedded rows -- run Embed first")
    if model and model != meta["model_tag"]:
        raise P.PipelineError(
            f"table '{table}' was built with {meta['model_tag']}, but you "
            f"asked for {model}. Vectors from different models are not "
            "comparable -- use Re-index to switch.")
    model = model or meta["model_tag"]
    client = P._client()
    vec, _ = P.embed_one(client, model, P.QUERY_INSTRUCT + q)
    k = max(topk, 1)
    # P.vector_search centralizes the cosine metric (NB: `Table.search()`
    # ignores a metric= kwarg -- it must go through .metric(), or LanceDB
    # falls back to L2 and the reported similarity is wrong) plus the optional
    # ANN knobs. nprobes/refine are no-ops until `index --ann` built a vector
    # index, so exact search is unchanged by default.
    if nprobes is None and refine is None:
        nprobes, refine = _index_tuning()
    hits = P.vector_search(table_obj, vec, k * 3,
                           nprobes=nprobes or 0, refine=refine or 0)
    # folder importance, same rule as the CLI: live weights file, no re-ingest
    hits = P.apply_weights(hits, P.load_weights(table))
    # NOTE: no FTS-index rebuild here on purpose -- `embed` already builds it,
    # and rebuilding on every query is expensive. If the index is missing
    # (older table), _fts_search fails gracefully and we fall back to vectors.
    if hybrid:
        fts = P._fts_search(table_obj, q, k * 3)
        hits = P._rrf([hits, fts], k) if fts else hits[:k]
    else:
        hits = hits[:k]
    return hits, meta, model


def _fetch_job(url: str, staging: str) -> dict:
    """Worker for + Link: download a page into staging, return its filename."""
    out = fetch_link(url, Path(staging))
    P.log(f"link saved: {out.name}")
    return {"md_file": out.name}


def _search(q: str, topk: int, table: str, model: str,
             hybrid: bool = True, nprobes: int = 0,
             refine: int = 0) -> dict:
    """Hybrid search. Returns hits for card rendering (one log summary)."""
    hits, meta, model = _retrieve(q, topk, table, model, hybrid,
                                  nprobes=nprobes, refine=refine)
    mode = "hybrid" if hybrid else "vector"
    P.log(f"query: {q!r} -> {len(hits)} hits (cards below, mode={mode})")
    return {"query": q, "hits": hits, "topk": max(topk, 1),
            "model": model, "table": table, "hybrid": hybrid}


_GREETINGS = frozenset({
    "hello", "hi", "hey", "yo", "hello!", "hi!", "hey!",
    "γεια", "γεια σου", "γεια σας", "χαίρετε",
    "ok", "thanks", "thank you", "ευχαριστώ", "ευχαριστω",
})

_RAG_MAX_CTX = 4


def _is_greeting(q: str) -> bool:
    """True for greetings / tiny chit-chat that needs no corpus lookup."""
    t = (q or "").strip().lower()
    if not t:
        return True
    if t in _GREETINGS:
        return True
    words = t.split()
    if len(words) <= 2 and len(t) <= 20 and t.rstrip("!?.,") in _GREETINGS:
        return True
    return False


def _cos_of(h: dict) -> float | None:
    """Cosine similarity from a LanceDB hit, None when unavailable."""
    try:
        d = h.get("_distance")
        if d is None:
            return None
        return 1.0 - float(d)
    except (TypeError, ValueError):
        return None


def _clean_chat_text(text: str) -> str:
    """Strip Qwen3 <think> traces and tidy whitespace."""
    t = text or ""
    # remove one or more <think>...</think> blocks (case-insensitive)
    t = re.sub(r"<think>.*?</think>", "", t,
               flags=re.DOTALL | re.IGNORECASE)
    # stray opening tag without close (cut-off generation)
    t = re.sub(r"<think>.*$", "", t, flags=re.DOTALL | re.IGNORECASE)
    return t.strip()


def _ollama_delta(chunk) -> str:
    """Content delta from one Ollama stream chunk (dict or object)."""
    try:
        if isinstance(chunk, dict):
            return (chunk.get("message") or {}).get("content", "") or ""
        return getattr(getattr(chunk, "message", None), "content", "") or ""
    except Exception:                                # noqa: BLE001
        return ""


def _chat(q: str, topk: int, table: str, embed_model: str,
          provider: str, chat_model: str, api_key: str = "",
          history: list | None = None, rag_enabled: bool = True,
          min_cos: float = 0.30, hybrid: bool = True,
          base_url: str = "", nprobes: int = 0, refine: int = 0) -> dict:
    """One turn of the conversation. LLM-first, RAG is a silent helper.

    Retrieval is attempted quietly; only hits with cosine >= min_cos are
    kept (max 4). Greetings, RAG-off, empty tables and low-similarity
    questions all fall back to a plain LLM answer -- the user never sees
    a "no hits" error. provider is "ollama" or a hosted id (anthropic /
    openai / opencode / thegrid / custom). `history` is the last exchanges
    ([{role, content}, ...]). `base_url` is only used by "custom".
    Returns {"query", "answer", "hits", "from_corpus", "history", ...}.
    """
    # -- silent retrieval: best effort, never blocks the conversation -----
    hits: list[dict] = []
    used: list[dict] = []
    rag_off_reason = ""
    if not rag_enabled:
        rag_off_reason = "RAG helper off"
    elif _is_greeting(q):
        rag_off_reason = "greeting -- no lookup needed"
    else:
        BUS.put("chat_stage", "searching corpus")
        try:
            hits, _meta, embed_model = _retrieve(q, min(max(topk, 1), 8),
                                                 table, embed_model, hybrid,
                                                 nprobes=nprobes,
                                                 refine=refine)
        except Exception as e:                          # noqa: BLE001
            P.log(f"chat: no corpus context ({e}) -- answering without it")
            hits = []
        if hits:
            # Keep hits without a score (FTS-only rows) and any vector hit at
            # or above the threshold; cap how many reach the prompt.
            used = [h for h in hits
                    if _cos_of(h) is None or _cos_of(h) >= min_cos][:_RAG_MAX_CTX]
            if not used:
                rag_off_reason = (
                    f"no chunk above min similarity {min_cos:.2f}")
            else:
                P.log(f"chat: {len(used)}/{len(hits)} chunks above "
                      f"{min_cos:.2f} -- using as helper")
        else:
            rag_off_reason = "no hits"

    if used:
        ctx = []
        for i, h in enumerate(used, 1):
            ctx.append(f"[{i}] {h.get('title', '')} > {h.get('section', '')} "
                       f"({h.get('source', '')})\n{hit_body(h)}")
        system = ("You are a helpful chat assistant. "
                  "Answer in the same language as the user's last message. "
                  "Be concise and plain. "
                  "OPTIONAL CONTEXT below may help -- use it only if it is "
                  "relevant. If it is irrelevant, ignore it completely. "
                  "Do not mix languages. Do not invent source numbers.\n\n"
                  "CONTEXT:\n" + "\n\n".join(ctx))
    else:
        system = ("You are a helpful chat assistant. "
                  "Answer in the same language as the user's last message. "
                  "Be concise and plain. Do not mix languages.")

    if provider == "ollama":
        client = P._client()
        try:
            client.show(chat_model)
        except Exception:
            BUS.put("chat_stage", f"pulling {chat_model}")
            P.log(f"pulling chat model {chat_model} (one-time download)...")
            try:
                client.pull(chat_model)
            except Exception as e:                      # noqa: BLE001
                raise P.PipelineError(
                    f"cannot pull chat model '{chat_model}': {e}")
        # system FIRST, then history, then the new question
        msgs = [{"role": "system", "content": system},
                *(history or []),
                {"role": "user", "content": q}]
        BUS.put("chat_stage", f"generating ({chat_model})")
        opts = {"temperature": 0.3, "num_ctx": 4096}
        text = ""
        streamed = False
        try:
            try:
                stream = client.chat(model=chat_model, messages=msgs,
                                     options=opts, think=False, stream=True)
            except TypeError:
                # older ollama client without `think`
                stream = client.chat(model=chat_model, messages=msgs,
                                     options=opts, stream=True)
            for chunk in stream:
                d = _ollama_delta(chunk)
                if d:
                    streamed = True
                    text += d
                    BUS.put("chat_token", d)
        except P.PipelineError:
            raise
        except Exception as e:                          # noqa: BLE001
            # stream unsupported or failed mid-flight: fall back below
            if not streamed:
                P.log(f"chat: stream failed ({e}) -- retrying without stream")
        if not streamed:
            try:
                try:
                    resp = client.chat(model=chat_model, messages=msgs,
                                       options=opts, think=False)
                except TypeError:
                    resp = client.chat(model=chat_model, messages=msgs,
                                       options=opts)
            except Exception as e:                      # noqa: BLE001
                raise P.PipelineError(f"chat failed: {e}")
            if isinstance(resp, dict):
                text = (resp.get("message") or {}).get("content", "")
            else:
                text = getattr(getattr(resp, "message", None),
                               "content", "") or ""
    else:
        BUS.put("chat_stage", f"generating ({chat_model})")
        import chat_providers as C
        try:
            text = C.complete(provider, model=chat_model, api_key=api_key,
                              system=system, user=q,
                              history=list(history or []),
                              base_url=base_url)
        except C.ProviderError as e:
            raise P.PipelineError(str(e))
    text = _clean_chat_text(text)
    if not text.strip():
        raise P.PipelineError("chat model returned an empty answer")
    if rag_off_reason and not used:
        P.log(f"chat: {q!r} -> general answer ({rag_off_reason}) "
              f"[{provider}]")
    else:
        P.log(f"chat: {q!r} -> {len(used)} corpus chunks [{provider}]")
    new_hist = (list(history or []) + [
        {"role": "user", "content": q},
        {"role": "assistant", "content": text.strip()}])[-6:]
    return {"query": q, "answer": text.strip(), "hits": used,
            "from_corpus": bool(used),
            "chat_model": chat_model, "provider": provider,
            "history": new_hist}


def _stats() -> None:
    # cmd_stats logs through pipeline.log(), which the Runner already routes
    # to the UI -- no stdout capture needed.
    P.cmd_stats(_Args(verbose=False))


def _index_job(table: str, ann: bool) -> dict:
    """Worker for the Vector-index button: maintenance for one table.

    `ann` is an explicit opt-in, so it also sets `force` -- the user has been
    told it is lossy and chose it anyway.
    """
    return P.cmd_index(_Args(table=table or None, ann=bool(ann),
                             type="IVF_PQ", sub_vectors=0, force=bool(ann),
                             no_compact=False, no_fts=False, no_scalar=False))


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
