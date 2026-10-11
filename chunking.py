"""chunking.py -- two splitters: recursive for prose, structural for code.

Prose never cuts mid-sentence and never cuts mid-word. Code is split on
function/class boundaries, with imports preserved as header context so the
retriever understands dependencies.

Both return `Chunk` objects carrying two views of the same text:

    text_raw    -> what the user sees (clean source text)
    embed_input -> what the model sees (title/section prefix + text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# English runs ~4 characters per tokenizer token. Used to convert the
# token-based chunk budget into a character budget without a tokenizer dep.
CHARS_PER_TOKEN = 4

DEFAULT_SIZE = 900          # tokens
DEFAULT_OVERLAP = 120       # tokens
DEFAULT_MIN_SIZE = 50       # tokens
CODE_MAX_SIZE = 600         # tokens

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+|\n{2,}")
_BULLET = re.compile(r"^\s*([-*+•]|\d+[.)])\s+")


@dataclass
class Chunk:
    text_raw: str
    embed_input: str
    section: str = ""
    kind: str = "prose"
    token_count: int = 0
    chunk_index: int = 0
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# token estimate
# --------------------------------------------------------------------------- #

def estimate_tokens(text: str) -> int:
    """Tokenizer-free estimate. ~4 chars/token for English prose.

    Slightly over-counts whitespace-heavy text, which is the safe direction:
    we would rather make chunks a little small than trip the model's limit.
    """
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #

def _split_units(text: str) -> list[str]:
    """Split into sentence-ish units, keeping paragraph breaks."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if _BULLET.match(para) or "\n" not in para and estimate_tokens(para) < 40:
            parts = [p for p in _SENTENCE_END.split(para) if p and p.strip()]
            units.extend(p.strip() for p in parts if p.strip())
            continue
        # long paragraph -> split into sentences, then re-join lines
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        buf = " ".join(lines)
        parts = [p for p in _SENTENCE_END.split(buf) if p and p.strip()]
        units.extend(p.strip() for p in parts if p.strip())
    return units


def _pack(units: list[str], size: int, overlap: int, min_size: int) -> list[str]:
    """Greedy pack units into chunks of ~size tokens, repeating ~`overlap`
    tokens at the start of each subsequent chunk.

    Progress is guaranteed: the next start is always > current start.
    """
    budgets = [estimate_tokens(u) for u in units]
    chunks: list[str] = []
    n = len(units)
    i = 0

    while i < n:
        total = 0
        end = i
        while end < n and (total + budgets[end] <= size or end == i):
            total += budgets[end]
            end += 1

        body = " ".join(units[i:end])
        if estimate_tokens(body) >= min_size or not chunks:
            chunks.append(body)

        if end >= n:
            break

        # back off from `end` so the tail of this chunk reappears next time
        back = end - 1
        used = 0
        while back > i and used < overlap:
            used += budgets[back]
            back -= 1
        nxt = back + 1
        i = nxt if nxt > i else end      # always advances
    return chunks


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def chunk_prose(
    text: str,
    section: str = "",
    title: str = "",
    kind: str = "prose",
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_size: int = DEFAULT_MIN_SIZE,
    start_index: int = 0,
    meta: dict | None = None,
) -> list[Chunk]:
    text = (text or "").strip()
    if not text:
        return []

    units = _split_units(text)
    if not units:
        return []

    # A single oversized unit (e.g. one dense line) gets hard-split on
    # whitespace so we still respect the budget.
    expanded: list[str] = []
    for u in units:
        if estimate_tokens(u) <= size * 2:
            expanded.append(u)
            continue
        words = u.split()
        buf: list[str] = []
        for w in words:
            buf.append(w)
            if estimate_tokens(" ".join(buf)) >= size:
                expanded.append(" ".join(buf))
                buf = []
        if buf:
            expanded.append(" ".join(buf))

    pieces = _pack(expanded, size, overlap, min_size)
    return [
        _make_chunk(p, section, title, kind, start_index + i, meta)
        for i, p in enumerate(pieces)
    ]


def _make_chunk(
    text: str,
    section: str,
    title: str,
    kind: str,
    index: int,
    meta: dict | None = None,
) -> Chunk:
    text = text.strip()
    # Code sections already carry the file path (`file.py::symbol`); do not
    # repeat the title in that case.
    if title and section.startswith(title):
        embed_input = f"Document: {section}\n{text}"
    elif section and title and section != title:
        embed_input = f"Document: {title} > {section}\n{text}"
    elif section:
        embed_input = f"Document: {section}\n{text}"
    elif title:
        embed_input = f"Document: {title}\n{text}"
    else:
        embed_input = text
    return Chunk(
        text_raw=text,
        embed_input=embed_input,
        section=section,
        kind=kind,
        token_count=estimate_tokens(text),
        chunk_index=index,
        meta=dict(meta or {}),
    )


# --------------------------------------------------------------------------- #
# code
# --------------------------------------------------------------------------- #

def chunk_code(
    text: str,
    section: str,
    title: str = "",
    max_size: int = CODE_MAX_SIZE,
    min_size: int = DEFAULT_MIN_SIZE,
    start_index: int = 0,
    meta: dict | None = None,
) -> list[Chunk]:
    """Code blocks arrive already split per function/class by extract.py.

    Oversized ones are cut on line boundaries with a one-line overlap, never
    mid-line.
    """
    text = (text or "").strip()
    if not text:
        return []

    if estimate_tokens(text) <= max_size:
        return [_make_chunk(text, section, title, "code", start_index, meta)]

    lines = text.split("\n")
    # Separate the header context (imports/comments at the top) so it can be
    # repeated on every piece.
    header: list[str] = []
    body_start = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        if s.startswith(("#", "//", "*", "/*", '"', "'", "import ", "from ",
                         "using ", "require(", "package ", "@import", "const ")):
            header.append(ln)
            body_start = i + 1
        else:
            break
    header_text = "\n".join(header).strip()
    body = lines[body_start:]

    budget = (max_size * CHARS_PER_TOKEN)
    header_cost = len(header_text) + 2
    pieces: list[str] = []
    buf: list[str] = []
    size = 0
    for ln in body:
        # one-line overlap between pieces
        if buf and size + len(ln) + 1 > budget:
            pieces.append("\n".join(buf))
            buf = [buf[-1]] if buf else []
            size = len(buf[0]) if buf else 0
        buf.append(ln)
        size += len(ln) + 1
    if buf:
        pieces.append("\n".join(buf))

    out: list[Chunk] = []
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        if header_text and header_text not in piece and header_cost < len(piece) // 2:
            piece = header_text + "\n\n" + piece
        out.append(_make_chunk(piece, section, title, "code", start_index + i, meta))
    if not out:
        out = [_make_chunk(text, section, title, "code", start_index, meta)]
    return out
