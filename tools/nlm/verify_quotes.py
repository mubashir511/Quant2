"""Check claimed book quotes against the extracted book text.

    .venv\\Scripts\\python.exe -m tools.nlm.verify_quotes "use market orders rather than limit orders" --book Schwager
    .venv\\Scripts\\python.exe -m tools.nlm.verify_quotes --extract            # (re)extract Books/*.pdf first

Exit code 0 when the quote is found (exact or loose), 1 when it is NOT in the text - i.e. do not cite it.
"""

from __future__ import annotations

import argparse

from tools.nlm.books import extract_books, find_quote, load_books


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("quote", nargs="?", help="the claimed quote / phrase")
    parser.add_argument("--book", help="restrict to books whose file name contains this text")
    parser.add_argument("--extract", action="store_true", help="extract the PDFs in Books/ to Books/_text first")
    args = parser.parse_args(argv)
    if args.extract:
        files = extract_books()
        print(f"extracted {len(files)} book(s)")
    if not args.quote:
        return 0
    hits = find_quote(args.quote, load_books(), args.book)
    if not hits:
        print("NOT FOUND in the book text - do not cite this as a book claim.")
        return 1
    for hit in hits:
        kind = {"exact": "EXACT", "squashed": "EXACT (PDF word-splits ignored)"}.get(hit.match, "loose (same vocabulary, not the same sentence)")
        print(f"[{kind}] {hit.book} ~line {hit.approx_line}: ...{hit.snippet}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
