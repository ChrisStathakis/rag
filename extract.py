"""extract.py -- 5 pluggable extractors returning a common intermediate type.

Every handler returns (title, [Block, ...]). Downstream (chunking/pipeline) is
therefore format-agnostic.

    txt/md  -> direct read, MD headings become sections, fenced code is split out
    html    -> BeautifulSoup, nav/footer/header/script/style/aside stripped
    docx    -> python-docx, Heading N becomes section
    pdf     -> pypdf, one Block per page, repeated header/footer lines removed
    code    -> our own parser, one Block per function/class

Unknown extensions are skipped and logged, never raised.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

SUPPORTED_TEXT = {".txt", ".md", ".markdown", ".rst"}
SUPPORTED_HTML = {".html", ".htm", ".xhtml"}
SUPPORTED_DOCX = {".docx"}
SUPPORTED_PDF = {".pdf"}
CODE_EXTENSIONS = {
    ".py", ".pyi",
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx",
    ".go", ".rs", ".java", ".kt", ".kts", ".scala",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".hh",
    ".cs", ".rb", ".php", ".swift", ".m", ".mm",
    ".sh", ".bash", ".zsh", ".ps1",
    ".sql", ".r", ".jl", ".dart", ".lua", ".pl",
    ".vue", ".svelte",
}

# Boilerplate that carries no retrieval value.
_HTML_DROP = {"nav", "footer", "header", "script", "style", "aside",
              "noscript", "svg", "form", "iframe", "button", "template"}


@dataclass
class Block:
    """One retrievable unit of source text, before chunking."""
    text: str
    section: str = ""
    meta: dict = field(default_factory=dict)


class Skipped(Exception):
    """Raised for formats we deliberately do not handle."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp-1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _clean(text: str) -> str:
    """Collapse whitespace runs, drop blank lines, strip trailing spaces."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    out: list[str] = []
    blank = False
    for ln in lines:
        if ln.strip():
            out.append(ln)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()


# --------------------------------------------------------------------------- #
# txt / markdown / rst
# --------------------------------------------------------------------------- #

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_RST_HEADING = re.compile(r"^[-=~^\"'#*+]{3,}\s*$")
_FENCE = re.compile(r"^(```|~~~)\s*(\S*)\s*$")


def _extract_md(path: Path) -> tuple[str, list[Block]]:
    text = _read_text(path)
    title = path.stem
    blocks: list[Block] = []
    section = ""
    buf: list[str] = []
    in_fence = False
    fence_mark = ""
    fence_lang = ""
    fence_buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        body = _clean("\n".join(buf))
        if body:
            blocks.append(Block(text=body, section=section, meta={"line": len(blocks)}))
        buf = []

    for line in text.split("\n"):
        m = _FENCE.match(line.strip())
        if m:                                  # toggle code fence
            if not in_fence:
                flush()
                in_fence, fence_mark, fence_lang = True, m.group(1), m.group(2)
                fence_buf = []
            elif line.strip().startswith(fence_mark):
                body = _clean("\n".join(fence_buf))
                if body:
                    prefix = f"{fence_lang} code\n" if fence_lang else "code\n"
                    blocks.append(Block(
                        text=prefix + body,
                        section=(f"{section} > code" if section else "code"),
                        meta={"kind_hint": "code", "lang": fence_lang},
                    ))
                in_fence = False
            continue

        if in_fence:
            fence_buf.append(line)
            continue

        h = _MD_HEADING.match(line)
        if h:
            flush()
            section = h.group(2).strip()
            continue
        buf.append(line)

    flush()                                    # whatever is left
    if not blocks:
        body = _clean(text)
        if body:
            blocks.append(Block(text=body, section=section))
    # First heading is usually the document title.
    if blocks and blocks[0].section and path.stem.lower() not in blocks[0].section.lower():
        title = blocks[0].section
    return title, blocks


def _extract_txt(path: Path) -> tuple[str, list[Block]]:
    body = _clean(_read_text(path))
    blocks = [Block(text=body, section=path.stem)] if body else []
    return path.stem, blocks


def _extract_rst(path: Path) -> tuple[str, list[Block]]:
    text = _read_text(path)
    title = path.stem
    blocks: list[Block] = []
    section = ""
    buf: list[str] = []
    lines = text.split("\n")

    def flush() -> None:
        nonlocal buf
        body = _clean("\n".join(buf))
        if body:
            blocks.append(Block(text=body, section=section))
        buf = []

    i = 0
    while i < len(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if _RST_HEADING.match(nxt) and lines[i].strip():
            flush()
            section = lines[i].strip()
            if not blocks:
                title = section
            i += 2
            continue
        buf.append(lines[i])
        i += 1
    flush()
    if not blocks:
        body = _clean(text)
        if body:
            blocks.append(Block(text=body, section=section))
    return title, blocks


# --------------------------------------------------------------------------- #
# html
# --------------------------------------------------------------------------- #

def _extract_html(path: Path) -> tuple[str, list[Block]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_read_text(path), "html.parser")
    for tag in soup(_HTML_DROP):
        tag.decompose()

    title = path.stem
    t = soup.find("title")
    if t and t.get_text(strip=True):
        title = t.get_text(strip=True)

    # Walk headings + content elements in document order so `section` tracks
    # the heading each piece of text falls under.
    nodes = soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "pre",
                           "blockquote"])
    if not nodes:
        body = _clean(soup.get_text("\n"))
        return title, ([Block(text=body, section=title)] if body else [])

    blocks: list[Block] = []
    section = title
    seen: set[str] = set()
    for el in nodes:
        # skip elements nested inside another content element (avoid dupes)
        if el.name in {"p", "li", "td", "blockquote", "pre"} and \
                el.find_parent(["p", "li", "td", "blockquote", "pre"]):
            continue
        txt = _clean(el.get_text(" ", strip=True))
        if not txt or txt in seen:
            continue
        seen.add(txt)

        if el.name.startswith("h"):
            section = txt
            if el.name == "h1" and not blocks:
                title = txt
            continue
        blocks.append(Block(text=txt, section=section))

    if not blocks:
        body = _clean(soup.get_text("\n"))
        if body:
            blocks = [Block(text=body, section=title)]
    return title, blocks


# --------------------------------------------------------------------------- #
# docx
# --------------------------------------------------------------------------- #

def _extract_docx(path: Path) -> tuple[str, list[Block]]:
    from docx import Document

    doc = Document(str(path))
    title = path.stem
    blocks: list[Block] = []
    section = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        body = _clean("\n".join(buf))
        if body:
            blocks.append(Block(text=body, section=section))
        buf = []

    for para in doc.paragraphs:
        style = (para.style.name or "") if para.style else ""
        txt = para.text.strip()
        if style.startswith("Heading") or style in {"Title", "Subtitle"}:
            flush()
            section = txt
            if not blocks and style in {"Title", "Heading 1"} and txt:
                title = txt
            continue
        if txt:
            buf.append(txt)
    flush()

    # Tables carry real content in reports -- flatten them.
    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            blocks.append(Block(text=_clean("\n".join(rows)), section=section or "table"))

    if not blocks:
        body = _clean("\n".join(p.text for p in doc.paragraphs))
        if body:
            blocks = [Block(text=body, section=section)]
    return title, blocks


# --------------------------------------------------------------------------- #
# pdf
# --------------------------------------------------------------------------- #

def _extract_pdf(path: Path) -> tuple[str, list[Block]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    title = path.stem
    if reader.metadata and reader.metadata.title:
        t = str(reader.metadata.title).strip()
        if t and len(t) < 200:
            title = t

    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")

    pages = [_clean(p) for p in pages]

    # Header/footer detection: short lines repeated on >50% of pages.
    freq: dict[str, int] = {}
    for p in pages:
        if not p:
            continue
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        for ln in lines[:3] + lines[-3:]:
            if len(ln) < 120:
                freq[ln] = freq.get(ln, 0) + 1
    threshold = max(2, int(len(pages) * 0.5))
    boiler = {ln for ln, n in freq.items() if n >= threshold}

    blocks: list[Block] = []
    for idx, p in enumerate(pages, start=1):
        if not p:
            continue
        lines = [ln for ln in p.split("\n") if ln.strip() not in boiler]
        body = _clean("\n".join(lines))
        if body:
            blocks.append(Block(text=body, section=f"page {idx}", meta={"page": idx}))
    return title, blocks


# --------------------------------------------------------------------------- #
# code
# --------------------------------------------------------------------------- #

_DEF_PATTERNS = [
    # python
    re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\("),
    re.compile(r"^class\s+(\w+)"),
    # js / ts
    re.compile(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*\("),
    re.compile(r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\("),
    re.compile(r"^(?:export\s+)?(?:default\s+)?class\s+(\w+)"),
    re.compile(r"^(?:export\s+)?interface\s+(\w+)"),
    re.compile(r"^(?:export\s+)?type\s+(\w+)\s*="),
    # c-family / java / go / rust
    re.compile(r"^(?:public\s+|private\s+|protected\s+|static\s+|final\s+|"
               r"internal\s+|virtual\s+|override\s+|inline\s+|constexpr\s+)*"
               r"(?:[\w<>\[\],*&?\s]+?)\s+(\w+)\s*\([^;]*\)\s*(?:const\s*)?\{"),
    re.compile(r"^func\s+(?:\([^)]*\)\s+)?(\w+)\s*\("),
    re.compile(r"^fn\s+(\w+)\s*\("),
]

_SKIP_NAMES = {
    "if", "for", "while", "switch", "catch", "return", "else", "do",
    "try", "foreach", "with", "sizeof", "typeof", "new", "await", "yield",
    "print", "echo", "require", "import", "include", "use",
}

_IMPORT_PREFIXES = (
    "import ", "from ", "using ", "require(", 'require "', "#include",
    "package ", "use ", "include ", "extern ", "@import", "const ",
)


def _split_code(text: str, rel: str) -> list[Block]:
    lines = text.split("\n")
    blocks: list[Block] = []
    header: list[str] = []          # top-of-file context (imports, licence)
    pending: list[str] = []
    current: list[str] = []
    current_name = ""
    current_kind = ""
    indent_base = ""

    def store() -> None:
        nonlocal current, current_name, current_kind
        body = _clean("\n".join(current))
        if body and current_name:
            context = "\n".join(header).strip()
            payload = (context + "\n\n" if context else "") + body
            blocks.append(Block(
                text=payload,
                section=f"{rel}::{current_name}",
                meta={"kind": current_kind, "symbol": current_name},
            ))
        current, current_name, current_kind = [], "", ""

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        # module-level context before the first definition
        if not current and not current_name:
            if any(stripped.startswith(p) for p in _IMPORT_PREFIXES) or \
               (stripped.startswith(("#", "//", '"""', "'''", "/*", "*")) and len(header) < 40):
                header.append(line)
                continue
            if not stripped:
                continue

        matched = None
        if stripped and not stripped.startswith(("#", "//", "*", "/*", '"', "'")):
            for pat in _DEF_PATTERNS:
                m = pat.match(stripped)
                if m:
                    name = m.group(1)
                    if name.lower() in _SKIP_NAMES:
                        break
                    matched = (name, "def")
                    break

        if matched:
            store()
            if pending:
                # Lines seen since the previous block (decorators, comments,
                # stray top-level statements) belong with the definition that
                # follows them -- keep them attached instead of dropping them.
                current.extend(pending)
                pending = []
            current_name, current_kind = matched
            indent_base = line[: len(line) - len(line.lstrip())]
            current.append(line)
            continue

        if current:
            if stripped and not line.startswith(indent_base) and \
               not stripped.startswith(("}", ")", "]", "elif", "else", "except",
                                        "finally", "case", "default", "end")):
                # dedent without closing brace -> treat as new top-level block
                store()
                pending.append(line)
                continue
            current.append(line)
        else:
            pending.append(line)

    store()
    if pending:
        # Top-level statements between/around definitions (assignments,
        # decorators, module guards) -- keep them as one tail block instead
        # of silently dropping them.
        tail = _clean("\n".join(pending))
        if tail:
            blocks.append(Block(text=tail, section=rel, meta={"kind": "file"}))
    if not blocks:
        body = _clean(text)
        if body:
            blocks.append(Block(text=body, section=rel, meta={"kind": "file"}))
    return blocks


def _extract_code(path: Path, root: Path) -> tuple[str, list[Block]]:
    try:
        rel = str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        rel = path.name
    text = _read_text(path)
    return rel, _split_code(text, rel)


# --------------------------------------------------------------------------- #
# dispatcher
# --------------------------------------------------------------------------- #

def detect_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in SUPPORTED_TEXT:
        return "prose"
    if ext in SUPPORTED_HTML:
        return "html"
    if ext in SUPPORTED_DOCX:
        return "docx"
    if ext in SUPPORTED_PDF:
        return "pdf"
    if ext in CODE_EXTENSIONS:
        return "code"
    return "unknown"


_HANDLERS: dict[str, Callable[..., tuple[str, list[Block]]]] = {
    ".md": _extract_md,
    ".markdown": _extract_md,
    ".rst": _extract_rst,
    ".txt": _extract_txt,
    ".html": _extract_html,
    ".htm": _extract_html,
    ".xhtml": _extract_html,
    ".docx": _extract_docx,
    ".pdf": _extract_pdf,
}


def extract(path: Path, root: Path | None = None) -> tuple[str, list[Block], str]:
    """Return (title, blocks, kind). Raises Skipped for unsupported formats."""
    ext = path.suffix.lower()
    root = root or path.parent
    kind = detect_kind(path)
    if kind == "unknown":
        raise Skipped(f"unsupported extension: {ext}")

    if kind == "code":
        title, blocks = _extract_code(path, root)
    else:
        handler = _HANDLERS[ext]
        title, blocks = handler(path)

    for b in blocks:
        b.meta.setdefault("kind", kind)
    return title, blocks, kind
