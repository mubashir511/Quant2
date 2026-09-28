"""Ask the books' NotebookLM notebook a question and keep the answer (with its citations) on disk.

    .venv\\Scripts\\python.exe -m tools.nlm.ask "question text" --name entries_vs_limits
    .venv\\Scripts\\python.exe -m tools.nlm.ask --file prompt.txt --name stops

Wraps the `notebooklm` CLI (notebooklm-py, already logged in under ~/.notebooklm): the question goes through
--prompt-file so quotes/newlines survive, and the answer is stored as Books/research/<name>.md. NEVER passes
--new (that deletes the notebook's server-side conversation). Answers are paraphrases with page citations that can
be wrong - run `python -m tools.nlm.verify_quotes` on anything that will become a rule.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT / "Books" / "research"
DEFAULT_NOTEBOOK = "8cd68ce5"  # "Technical Analysis Explained: Mastering Market Trends and Indicators" (the 9 books)

PREAMBLE = (
    "You are helping design deterministic rules for an intraday (M5-decision, H1/H4/D1 context) trading system. "
    "Answer ONLY from the books in this notebook. For every rule, name the book and chapter/section and quote or "
    "closely paraphrase the exact wording; say clearly when a book gives NO guidance or when authors disagree; give "
    "every number the books actually state. Prefer concrete, mechanically testable conditions. "
)


class NotebookLMError(RuntimeError):
    pass


def cli_path() -> Path:
    exe = Path(sys.executable).parent / "notebooklm.exe"
    return exe if exe.exists() else Path("notebooklm")


def ask(question: str, notebook: str = DEFAULT_NOTEBOOK, timeout: int = 600, preamble: str = PREAMBLE) -> dict:
    """Run one NotebookLM question and return the parsed JSON ({'answer': str, 'references': [...], ...}).
    Raises NotebookLMError on a non-zero exit, empty output or unparsable JSON."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        handle.write(preamble + question)
        prompt_path = handle.name
    try:
        result = subprocess.run(
            [str(cli_path()), "ask", "-n", notebook, "--prompt-file", prompt_path, "--json"],
            capture_output=True, text=True, encoding="utf-8", timeout=timeout,
        )
    finally:
        Path(prompt_path).unlink(missing_ok=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise NotebookLMError(f"notebooklm exited {result.returncode}: {(result.stderr or result.stdout)[:300]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise NotebookLMError(f"could not parse the notebooklm JSON output: {e}") from e


def save_answer(name: str, question: str, answer: dict, directory: Path = RESEARCH_DIR) -> Path:
    """Write Books/research/<name>.md with the question, the answer text and the reference list."""
    directory.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    path = directory / f"{safe}.md"
    references = answer.get("references") or []
    body = [f"# {name}", "", "## Question", "", question.strip(), "", "## Answer (NotebookLM - unverified paraphrase)", "", str(answer.get("answer", "")).strip()]
    if references:
        body += ["", "## References reported by NotebookLM", ""] + [f"- {json.dumps(r, ensure_ascii=False)[:300]}" for r in references[:40]]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("question", nargs="?", help="the question text (or use --file)")
    parser.add_argument("--file", help="read the question from a file")
    parser.add_argument("--name", required=True, help="output name (Books/research/<name>.md)")
    parser.add_argument("--notebook", default=DEFAULT_NOTEBOOK)
    args = parser.parse_args(argv)
    question = Path(args.file).read_text(encoding="utf-8") if args.file else args.question
    if not question:
        parser.error("give a question or --file")
    answer = ask(question, notebook=args.notebook)
    path = save_answer(args.name, question, answer)
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
