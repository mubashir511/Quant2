import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.nlm import ask as nlm_ask
from tools.nlm.books import extract_books, find_quote, normalise

BOOKS = {
    "Getting_Started_by_Jack_Schwager": (
        "19. If selling into resistance or buying into support and the market con-\n"
        "     solidates instead of rev ersing, get out.\n"
        "11. In most cases, use market orders rather than limit orders. This is\n"
        "    especially important when liquidating a losing position.\n"
    ),
    "How_to_Make_Money_in_Stocks_by_William_O_Neil": "You must avoid buying once the stock is extended more than 5% or so.\n",
}


def test_normalise_joins_hyphenated_breaks_and_unifies_punctuation():
    assert normalise("con-\n   solidates “the” ﬁrst  – x") == 'consolidates "the" first - x'


def test_exact_quote_is_found_across_a_line_wrap_and_reports_the_book_and_line():
    hits = find_quote("use market orders rather than limit orders", BOOKS)
    assert len(hits) == 1 and hits[0].match == "exact" and hits[0].exact and hits[0].book.endswith("Jack_Schwager")
    assert hits[0].approx_line == 3


def test_pdf_word_splits_are_ignored_but_reported_as_squashed():
    hits = find_quote("market consolidates instead of reversing, get out", BOOKS)
    assert [h.match for h in hits] == ["squashed"] and hits[0].exact


def test_a_paraphrase_that_keeps_the_vocabulary_is_only_a_loose_match():
    hits = find_quote("orders: market rather than limit, especially when liquidating a losing position", BOOKS)
    assert hits and hits[0].match == "loose" and not hits[0].exact


def test_a_claim_that_is_not_in_the_text_returns_nothing():
    assert find_quote("do not place a limit order blindly at the strongest support", BOOKS) == []


def test_the_book_filter_ignores_case_and_punctuation():
    assert find_quote("extended more than 5%", BOOKS, "O'Neil")
    assert find_quote("extended more than 5%", BOOKS, "oneil")
    assert find_quote("extended more than 5%", BOOKS, "Schwager") == []


def test_extract_books_needs_pdftotext_and_skips_existing_text(tmp_path):
    books, text = tmp_path / "Books", tmp_path / "Books" / "_text"
    books.mkdir()
    (books / "A Book - Someone.pdf").write_bytes(b"%PDF-1.4")
    with patch("tools.nlm.books.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="pdftotext"):
            extract_books(books, text)
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_text("hello", encoding="utf-8")

    with patch("tools.nlm.books.shutil.which", return_value="pdftotext"), patch("tools.nlm.books.subprocess.run", side_effect=fake_run):
        files = extract_books(books, text)
        extract_books(books, text)  # second call: already extracted
    assert [f.name for f in files] == ["A_Book_Someone.txt"] and len(calls) == 1 and calls[0][1] == "-layout"


class _Done:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def test_ask_sends_the_prompt_through_a_file_and_never_uses_new():
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["prompt"] = Path(cmd[cmd.index("--prompt-file") + 1]).read_text(encoding="utf-8")
        return _Done(json.dumps({"answer": "A", "references": [{"n": 1}]}))

    with patch("tools.nlm.ask.subprocess.run", side_effect=fake_run):
        result = nlm_ask.ask("what is a spring?")
    assert result["answer"] == "A"
    assert "--new" not in seen["cmd"] and "--json" in seen["cmd"] and seen["cmd"][seen["cmd"].index("-n") + 1] == nlm_ask.DEFAULT_NOTEBOOK
    assert seen["prompt"].endswith("what is a spring?") and "Answer ONLY from the books" in seen["prompt"]
    assert not Path(seen["cmd"][seen["cmd"].index("--prompt-file") + 1]).exists()  # temp file cleaned up


def test_ask_raises_on_failure_empty_output_or_bad_json():
    for done in (_Done("", 1, "auth expired"), _Done("   "), _Done("not json")):
        with patch("tools.nlm.ask.subprocess.run", return_value=done):
            with pytest.raises(nlm_ask.NotebookLMError):
                nlm_ask.ask("q")


def test_save_answer_writes_question_answer_and_references(tmp_path):
    path = nlm_ask.save_answer("stops / trailing", "Q?", {"answer": "text", "references": [{"source": "x"}]}, tmp_path)
    body = path.read_text(encoding="utf-8")
    assert path.name == "stops___trailing.md" and "Q?" in body and "unverified paraphrase" in body and '"source": "x"' in body
