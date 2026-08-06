from unittest.mock import patch

from ai.narrate import build_summary, narrate
from risk.rebalance import Suggestion
from tests.test_rebalance import make_position


def test_build_summary_lists_positions_and_suggestions():
    p = make_position(symbol="XAU", sl=None, profit=-5.0)
    s = Suggestion("XAU", "no_stop", "No stop-loss is set on this position.")
    summary = build_summary([p], [s])
    assert "XAU" in summary
    assert "no_stop" in summary
    assert "No stop-loss is set" in summary


def test_build_summary_handles_empty_state():
    summary = build_summary([], [])
    assert "none" in summary.lower()


@patch("ai.narrate.run_claude")
def test_narrate_delegates_to_claude_cli_with_summary_embedded(mock_run_claude):
    mock_run_claude.return_value = "All positions look fine."
    result = narrate("some summary")
    assert result == "All positions look fine."
    prompt = mock_run_claude.call_args.args[0]
    assert "some summary" in prompt
