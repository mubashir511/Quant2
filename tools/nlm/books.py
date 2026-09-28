"""Book text access for the rule ledger: extract the PDFs in Books/ to plain text once, then search them.

NotebookLM answers are useful but are paraphrases with page citations that are sometimes wrong (2026-09-25 check:
two "quotes" it produced were not in the text). Nothing goes into a prompt or a rule ledger entry as a book claim
until `find_quote` finds it in the actual book text. The searcher is tolerant of what PDF extraction does to text:
hyphenated line breaks ("con-\\nsolidates"), wrapped lines, ligatures and curly quotes.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOKS_DIR = ROOT / "Books"
TEXT_DIR = BOOKS_DIR / "_text"

_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}


def normalise(text: str) -> str:
    """Lower-case, join hyphenated line breaks, unify quotes/dashes/ligatures and collapse whitespace."""
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = re.sub(r"-\s*\n\s*", "", text)  # 'con-\nsolidates' -> 'consolidates'
    text = (
        text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-")
    )
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def extract_books(books_dir: Path = BOOKS_DIR, text_dir: Path = TEXT_DIR, force: bool = False) -> list[Path]:
    """pdftotext every Books/*.pdf into Books/_text/<stem>.txt (skips ones already extracted). Returns the text files.
    Raises RuntimeError when pdftotext is not installed."""
    exe = shutil.which("pdftotext")
    if exe is None:
        raise RuntimeError("pdftotext is not installed (poppler); cannot extract the book PDFs")
    text_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for pdf in sorted(books_dir.glob("*.pdf")):
        target = text_dir / (re.sub(r"[^A-Za-z0-9]+", "_", pdf.stem).strip("_") + ".txt")
        if force or not target.exists():
            subprocess.run([exe, "-layout", str(pdf), str(target)], check=True, capture_output=True)
        out.append(target)
    return out


@dataclass
class QuoteHit:
    book: str
    snippet: str
    approx_line: int
    exact: bool  # True: the whole quote was found (as written, or with PDF-extraction spaces removed); False: loose
    match: str = "exact"  # "exact" | "squashed" (found only after ignoring all spaces - PDF word splits) | "loose"


def load_books(text_dir: Path = TEXT_DIR) -> dict[str, str]:
    """{book file stem: raw text} for every extracted book."""
    return {p.stem: p.read_text(encoding="utf-8", errors="replace") for p in sorted(text_dir.glob("*.txt"))}


def find_quote(
    quote: str, books: dict[str, str], book_filter: str | None = None, window_chars: int = 400, min_word_len: int = 4
) -> list[QuoteHit]:
    """Find `quote` in the books. First an exact match of the normalised quote; if none, a 'loose' match where every
    word of at least `min_word_len` letters occurs inside one `window_chars` window (catches a paraphrase that kept the
    author's vocabulary) - reported with exact=False so a human can judge it. Empty list = NOT in the text."""
    needle = normalise(quote)
    hits: list[QuoteHit] = []
    words = [w for w in re.findall(r"[a-z0-9']+", needle) if len(w) >= min_word_len]
    squashed_needle = re.sub(r"\s+", "", needle)
    wanted = _alnum(book_filter) if book_filter else None
    for name, raw in books.items():
        if wanted and wanted not in _alnum(name):
            continue
        hay = normalise(raw)
        pos = hay.find(needle) if needle else -1
        if pos >= 0:
            hits.append(QuoteHit(name, hay[max(pos - 80, 0):pos + len(needle) + 80], _line_of(raw, needle), True, "exact"))
            continue
        if len(squashed_needle) >= 25:
            squashed_hay = re.sub(r"\s+", "", hay)
            spos = squashed_hay.find(squashed_needle)
            if spos >= 0:
                hits.append(QuoteHit(name, "(text matches once PDF word-splits are ignored) " + squashed_needle[:120], _line_of(raw, needle), True, "squashed"))
                continue
        if len(words) >= 3:
            loose = _loose_match(hay, words, window_chars)
            if loose is not None:
                hits.append(QuoteHit(name, hay[max(loose - 80, 0):loose + window_chars // 2], _line_of(raw, words[0]), False, "loose"))
    return hits


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _loose_match(hay: str, words: list[str], window: int) -> int | None:
    anchor = min(words, key=lambda w: hay.count(w) or 10**9)  # rarest word anchors the scan
    start = 0
    while True:
        i = hay.find(anchor, start)
        if i < 0:
            return None
        chunk = hay[max(i - window // 2, 0):i + window // 2]
        if all(w in chunk for w in words):
            return max(i - window // 2, 0)
        start = i + len(anchor)


def _line_of(raw: str, needle_or_word: str) -> int:
    """Best-effort line number of the first occurrence of the first few words of the needle in the raw text."""
    first = normalise(needle_or_word).split(" ")[:3]
    pattern = r"\W+".join(re.escape(w) for w in first if w)
    m = re.search(pattern, raw, re.IGNORECASE) if pattern else None
    return raw.count("\n", 0, m.start()) + 1 if m else 0
