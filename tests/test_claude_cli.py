import subprocess
from unittest.mock import patch

from ai.claude_cli import CLI_FAILED_MESSAGE, CLI_MISSING_MESSAGE, run_claude


@patch("ai.claude_cli.subprocess.run")
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_returns_stdout_on_success(mock_which, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[r"C:\fake\claude.CMD", "-p"], returncode=0, stdout="Looks fine.\n"
    )
    result = run_claude("some prompt")
    assert result == "Looks fine."
    called_args = mock_run.call_args.args[0]
    assert called_args == [r"C:\fake\claude.CMD", "-p"]
    assert mock_run.call_args.kwargs["input"] == "some prompt"


@patch("ai.claude_cli.subprocess.run")
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_omits_allowed_tools_flag_by_default(mock_which, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[r"C:\fake\claude.CMD", "-p"], returncode=0, stdout="ok\n"
    )
    run_claude("some prompt")
    called_args = mock_run.call_args.args[0]
    assert called_args == [r"C:\fake\claude.CMD", "-p"]


@patch("ai.claude_cli.subprocess.run")
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_adds_allowed_tools_flag_when_given(mock_which, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[r"C:\fake\claude.CMD", "-p"], returncode=0, stdout="ok\n"
    )
    run_claude("some prompt", allowed_tools=["WebSearch", "WebFetch"])
    called_args = mock_run.call_args.args[0]
    assert called_args == [
        r"C:\fake\claude.CMD",
        "-p",
        "--allowedTools",
        "WebSearch",
        "WebFetch",
    ]


@patch("ai.claude_cli.subprocess.run")
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_omits_model_flag_by_default(mock_which, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[r"C:\fake\claude.CMD", "-p"], returncode=0, stdout="ok\n"
    )
    run_claude("some prompt")
    called_args = mock_run.call_args.args[0]
    assert "--model" not in called_args


@patch("ai.claude_cli.subprocess.run")
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_adds_model_flag_when_given(mock_which, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[r"C:\fake\claude.CMD", "-p"], returncode=0, stdout="ok\n"
    )
    run_claude("some prompt", model="opus")
    called_args = mock_run.call_args.args[0]
    assert called_args == [r"C:\fake\claude.CMD", "-p", "--model", "opus"]


@patch("ai.claude_cli.subprocess.run")
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_falls_back_on_nonzero_exit(mock_which, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[r"C:\fake\claude.CMD", "-p"], returncode=1, stdout=""
    )
    assert run_claude("some prompt") == CLI_FAILED_MESSAGE


@patch("ai.claude_cli.shutil.which", return_value=None)
def test_run_claude_reports_missing_cli_without_calling_subprocess(mock_which):
    assert run_claude("some prompt") == CLI_MISSING_MESSAGE


@patch("ai.claude_cli.subprocess.run", side_effect=FileNotFoundError)
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_falls_back_when_resolved_cli_still_not_runnable(mock_which, mock_run):
    assert run_claude("some prompt") == CLI_FAILED_MESSAGE


@patch(
    "ai.claude_cli.subprocess.run",
    side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=60),
)
@patch("ai.claude_cli.shutil.which", return_value=r"C:\fake\claude.CMD")
def test_run_claude_falls_back_on_timeout(mock_which, mock_run):
    assert run_claude("some prompt") == CLI_FAILED_MESSAGE
