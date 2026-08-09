from unittest.mock import patch

from ai.session_record import SessionRecord, save_portfolio_session

_RECORD = SessionRecord(
    summary="Account: balance 10000.00 USD...",
    model="sonnet",
    draft="Claude's draft mix and reasoning.",
    audit_block="OpenAI gpt-oss-20b audit:\nLooks reasonable.",
    audit_available=True,
    final_answer="Claude's final revised mix.",
)


def test_save_portfolio_session_writes_file_with_all_sections(tmp_path):
    path = save_portfolio_session(_RECORD, records_dir=tmp_path)
    assert path is not None
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Account: balance 10000.00 USD..." in content
    assert "Claude's draft mix and reasoning." in content
    assert "OpenAI gpt-oss-20b audit:" in content
    assert "Claude's final revised mix." in content
    assert "sonnet" in content


def test_save_portfolio_session_states_audit_availability(tmp_path):
    path = save_portfolio_session(_RECORD, records_dir=tmp_path)
    content = path.read_text(encoding="utf-8")
    assert "audit available:** yes" in content.lower()


def test_save_portfolio_session_creates_records_dir_if_missing(tmp_path):
    nested = tmp_path / "does" / "not" / "exist" / "yet"
    path = save_portfolio_session(_RECORD, records_dir=nested)
    assert path is not None
    assert nested.exists()


def test_save_portfolio_session_filename_is_timestamped(tmp_path):
    path = save_portfolio_session(_RECORD, records_dir=tmp_path)
    assert path.name.startswith("portfolio_suggestion_")
    assert path.suffix == ".md"


@patch("pathlib.Path.mkdir", side_effect=OSError("disk full"))
def test_save_portfolio_session_returns_none_on_failure_without_raising(mock_mkdir, tmp_path):
    result = save_portfolio_session(_RECORD, records_dir=tmp_path / "unwritable")
    assert result is None


def test_save_portfolio_session_uses_configured_default_dir(tmp_path):
    import config

    original_dir = config.PORTFOLIO_RECORDS_DIR
    config.PORTFOLIO_RECORDS_DIR = str(tmp_path)
    try:
        path = save_portfolio_session(_RECORD)
        assert path is not None
        assert path.parent == tmp_path
    finally:
        config.PORTFOLIO_RECORDS_DIR = original_dir
