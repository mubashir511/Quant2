import logging
import subprocess
from unittest.mock import MagicMock, patch

from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE, run_claude


def _mock_process(returncode=0, stdout="ok\n", pid=1234):
    process = MagicMock()
    process.communicate.return_value = (stdout, "")
    process.returncode = returncode
    process.pid = pid
    return process


@patch("ai.claude_cli.subprocess.Popen")
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_returns_stdout_on_success(mock_which, mock_popen):
    mock_popen.return_value = _mock_process(stdout="Looks fine.\n")
    result = run_claude("some prompt")
    assert result == "Looks fine."
    called_args = mock_popen.call_args.args[0]
    assert called_args == [r"C:\fake\claude.CMD", "-p"]
    assert mock_popen.return_value.communicate.call_args.kwargs["input"] == "some prompt"


@patch("ai.claude_cli.subprocess.Popen")
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_omits_allowed_tools_flag_by_default(mock_which, mock_popen):
    mock_popen.return_value = _mock_process()
    run_claude("some prompt")
    called_args = mock_popen.call_args.args[0]
    assert called_args == [r"C:\fake\claude.CMD", "-p"]


@patch("ai.claude_cli.subprocess.Popen")
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_adds_allowed_tools_flag_when_given(mock_which, mock_popen):
    mock_popen.return_value = _mock_process()
    run_claude("some prompt", allowed_tools=["WebSearch", "WebFetch"])
    called_args = mock_popen.call_args.args[0]
    assert called_args == [
        r"C:\fake\claude.CMD",
        "-p",
        "--allowedTools",
        "WebSearch",
        "WebFetch",
    ]


@patch("ai.claude_cli.subprocess.Popen")
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_omits_model_flag_by_default(mock_which, mock_popen):
    mock_popen.return_value = _mock_process()
    run_claude("some prompt")
    called_args = mock_popen.call_args.args[0]
    assert "--model" not in called_args


@patch("ai.claude_cli.subprocess.Popen")
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_adds_model_flag_when_given(mock_which, mock_popen):
    mock_popen.return_value = _mock_process()
    run_claude("some prompt", model="opus")
    called_args = mock_popen.call_args.args[0]
    assert called_args == [r"C:\fake\claude.CMD", "-p", "--model", "opus"]


@patch("ai.claude_cli.subprocess.Popen")
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_falls_back_on_nonzero_exit_with_stderr_detail(mock_which, mock_popen):
    process = _mock_process(returncode=1, stdout="")
    process.communicate.return_value = ("", "Error: rate limit exceeded")
    mock_popen.return_value = process
    result = run_claude("some prompt")
    assert result.startswith(CLI_FAILED_PREFIX)
    assert "code 1" in result
    assert "Error: rate limit exceeded" in result


@patch("ai.claude_cli.subprocess.Popen")
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_falls_back_on_empty_output_with_no_stderr(mock_which, mock_popen):
    process = _mock_process(returncode=0, stdout="")
    process.communicate.return_value = ("", "")
    mock_popen.return_value = process
    result = run_claude("some prompt")
    assert result.startswith(CLI_FAILED_PREFIX)
    assert "no output" in result


@patch("ai.claude_cli.shutil.which", return_value=None)
def test_run_claude_reports_missing_cli_without_calling_subprocess(mock_which):
    assert run_claude("some prompt") == CLI_MISSING_MESSAGE


@patch("ai.claude_cli.subprocess.Popen", side_effect=FileNotFoundError("no such file"))
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_falls_back_when_resolved_cli_still_not_runnable(mock_which, mock_popen, caplog):
    with caplog.at_level(logging.WARNING):
        result = run_claude("some prompt")
    assert result.startswith(CLI_FAILED_PREFIX)
    assert "FileNotFoundError" in result
    assert "no such file" in result


@patch("ai.claude_cli.subprocess.run")  # the taskkill call inside _kill_process_tree
@patch("ai.claude_cli.subprocess.Popen")
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_falls_back_on_timeout_with_duration(mock_which, mock_popen, mock_taskkill_run):
    process = _mock_process()
    process.communicate.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=60)
    mock_popen.return_value = process
    result = run_claude("some prompt", timeout=60)
    assert result.startswith(CLI_FAILED_PREFIX)
    assert "60s" in result


@patch("ai.claude_cli.subprocess.run")  # the taskkill call inside _kill_process_tree
@patch("ai.claude_cli.subprocess.Popen")
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_kills_whole_process_tree_on_timeout(mock_which, mock_popen, mock_taskkill_run):
    # Regression test for a real hang confirmed live against the sibling
    # `gemini` CLI: Popen.kill() alone only terminates the immediate .cmd
    # shim, not any node.exe child it spawned, so a timed-out call could
    # hang far past its nominal timeout with an orphaned, near-idle
    # node.exe still holding the stdout pipe open. taskkill /F /T /PID
    # <pid> kills the whole tree instead of just the top-level shim.
    process = _mock_process(pid=4321)
    process.communicate.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=60)
    mock_popen.return_value = process
    run_claude("some prompt")
    taskkill_args = mock_taskkill_run.call_args.args[0]
    assert taskkill_args == ["taskkill", "/F", "/T", "/PID", "4321"]
