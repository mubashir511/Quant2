from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SessionRecord:
    summary: str
    model: str
    draft: str
    audit_block: str
    audit_available: bool
    final_answer: str


def save_portfolio_session(record: SessionRecord, records_dir: Path | None = None) -> Path | None:
    """Writes a full, human-readable Markdown transcript of one Portfolio
    Suggestion run — the input data, Claude's initial draft, both
    independent audit reports, and Claude's final revised answer — to a
    local `records/` folder, like minutes of a meeting, for future
    reference. Never raises: a failure to save the record (disk full,
    permissions) shouldn't break the actual feature, so any OSError is
    swallowed and None returned instead."""
    import config

    records_dir = Path(config.PORTFOLIO_RECORDS_DIR) if records_dir is None else records_dir
    timestamp = datetime.now()

    content = (
        f"# Portfolio Suggestion Session — {timestamp:%Y-%m-%d %H:%M:%S}\n\n"
        f"**Model:** {record.model}\n"
        f"**Independent audit available:** {'yes' if record.audit_available else 'no'}\n\n"
        "## Input data (account, market, macro summary)\n\n"
        f"{record.summary}\n\n"
        "## Stage 1 — Claude's initial draft\n\n"
        f"{record.draft}\n\n"
        "## Stage 2 — Independent audits\n\n"
        f"{record.audit_block}\n\n"
        "## Stage 3 — Claude's final revised suggestion\n\n"
        f"{record.final_answer}\n"
    )

    try:
        records_dir.mkdir(parents=True, exist_ok=True)
        path = records_dir / f"portfolio_suggestion_{timestamp:%Y-%m-%d_%H%M%S}.md"
        path.write_text(content, encoding="utf-8")
        return path
    except OSError:
        return None
