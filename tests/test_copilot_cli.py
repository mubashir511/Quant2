import logging
import os
import subprocess
from unittest.mock import MagicMock, patch

from ai.copilot_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE, run_copilot


def _mock_process(returncode=0, stdout="ok\n", stderr="", pid=1234):
    process = MagicMock()
    process.communicate.return_value = (stdout, stderr)
    process.returncode = returncode
    process.pid = pid
    return process


def _extract_temp_path(wrapper_prompt: str) -> str:
    # Mirrors run_copilot's own "Read the file at {path} using..." shape.
    return wrapper_prompt.split("Read the file at ", 1)[1].split(" using your file-read tool", 1)[0]


@patch("ai.copilot_cli.subprocess.Popen")
@patch("ai.copilot_cli.shutil.which", return_value=r"C:\fake\copilot.exe")
def test_run_copilot_returns_stdout_on_success(mock_which, mock_popen):
    mock_popen.return_value = _mock_process(stdout="Looks fine.\n")
    result = run_copilot("some prompt")
    assert result == "Looks fine."
    called_args = mock_popen.call_args.args[0]
    assert called_args[0] == r"C:\fake\copilot.exe"
    assert called_args[1] == "-p"
    assert called_args[3:] == ["--allow-all-tools", "--allow-all-urls", "--silent"]
    # The real prompt text is NOT passed directly on argv (that's the
    # whole point of this design) — only a fixed, ASCII-only wrapper
    # instruction pointing at a temp file goes through -p.
    wrapper_prompt = called_args[2]
    assert "some prompt" not in wrapper_prompt
    assert "Read the file at" in wrapper_prompt


@patch("ai.copilot_cli.subprocess.Popen")
@patch("ai.copilot_cli.shutil.which", return_value=r"C:\fake\copilot.exe")
def test_run_copilot_writes_the_real_prompt_to_the_referenced_temp_file(mock_which, mock_popen):
    # The temp file is cleaned up before run_copilot returns, so its
    # content must be captured DURING the call — from inside Popen's own
    # side_effect, which fires while the file still exists on disk.
    captured = {}

    def _capture_and_respond(args, **kwargs):
        temp_path = _extract_temp_path(args[2])
        with open(temp_path, encoding="utf-8") as f:
            captured["content"] = f.read()
        return _mock_process(stdout="ok\n")

    mock_popen.side_effect = _capture_and_respond
    real_prompt = 'A prompt with "quotes", a trailing backslash \\, and an em-dash \u2014.'
    run_copilot(real_prompt)
    # The file must have already been written (and readable) at the
    # moment Popen was called, containing the REAL prompt content
    # untouched by any command-line escaping — this is exactly the
    # content a direct argv pass couldn't be guaranteed safe for.
    assert captured["content"] == real_prompt


@patch("ai.copilot_cli.subprocess.Popen")
@patch("ai.copilot_cli.shutil.which", return_value=r"C:\fake\copilot.exe")
def test_run_copilot_cleans_up_the_temp_file_after_success(mock_which, mock_popen):
    mock_popen.return_value = _mock_process(stdout="ok\n")
    run_copilot("some prompt")
    wrapper_prompt = mock_popen.call_args.args[0][2]
    temp_path = _extract_temp_path(wrapper_prompt)
    assert not os.path.exists(temp_path)


@patch("ai.copilot_cli.subprocess.run")  # the taskkill call inside _kill_process_tree
@patch("ai.copilot_cli.subprocess.Popen")
@patch("ai.copilot_cli.shutil.which", return_value=r"C:\fake\copilot.exe")
def test_run_copilot_cleans_up_the_temp_file_even_on_timeout(mock_which, mock_popen, mock_taskkill_run):
    process = _mock_process()
    process.communicate.side_effect = subprocess.TimeoutExpired(cmd="copilot", timeout=240)
    mock_popen.return_value = process
    run_copilot("some prompt")
    wrapper_prompt = mock_popen.call_args.args[0][2]
    temp_path = _extract_temp_path(wrapper_prompt)
    assert not os.path.exists(temp_path)


@patch("ai.copilot_cli.shutil.which", return_value=None)
def test_run_copilot_reports_missing_cli_without_calling_subprocess(mock_which):
    assert run_copilot("some prompt") == CLI_MISSING_MESSAGE


@patch("ai.copilot_cli.subprocess.Popen")
@patch("ai.copilot_cli.shutil.which", return_value=r"C:\fake\copilot.exe")
def test_run_copilot_reports_missing_on_no_auth_stderr(mock_which, mock_popen):
    # The exact real error confirmed live: "Error: No authentication
    # information found." — treated as CLI_MISSING_MESSAGE (a config
    # problem, not a transient failure), not the generic failed-prefix.
    process = _mock_process(returncode=1, stdout="")
    process.communicate.return_value = ("", "Error: No authentication information found.")
    mock_popen.return_value = process
    assert run_copilot("some prompt") == CLI_MISSING_MESSAGE


@patch("ai.copilot_cli.subprocess.Popen")
@patch("ai.copilot_cli.shutil.which", return_value=r"C:\fake\copilot.exe")
def test_run_copilot_falls_back_on_nonzero_exit_with_stderr_detail(mock_which, mock_popen):
    process = _mock_process(returncode=1, stdout="")
    process.communicate.return_value = ("", "Error: rate limit exceeded")
    mock_popen.return_value = process
    result = run_copilot("some prompt")
    assert result.startswith(CLI_FAILED_PREFIX)
    assert "code 1" in result
    assert "Error: rate limit exceeded" in result


@patch("ai.copilot_cli.subprocess.Popen")
@patch("ai.copilot_cli.shutil.which", return_value=r"C:\fake\copilot.exe")
def test_run_copilot_falls_back_on_empty_output_with_no_stderr(mock_which, mock_popen):
    process = _mock_process(returncode=0, stdout="")
    process.communicate.return_value = ("", "")
    mock_popen.return_value = process
    result = run_copilot("some prompt")
    assert result.startswith(CLI_FAILED_PREFIX)
    assert "no output" in result


@patch("ai.copilot_cli.subprocess.Popen", side_effect=FileNotFoundError("no such file"))
@patch("ai.copilot_cli.shutil.which", return_value=r"C:\fake\copilot.exe")
def test_run_copilot_falls_back_when_resolved_cli_still_not_runnable(mock_which, mock_popen, caplog):
    with caplog.at_level(logging.WARNING):
        result = run_copilot("some prompt")
    assert result.startswith(CLI_FAILED_PREFIX)
    assert "FileNotFoundError" in result
    assert "no such file" in result


@patch("ai.copilot_cli.subprocess.run")  # the taskkill call inside _kill_process_tree
@patch("ai.copilot_cli.subprocess.Popen")
@patch("ai.copilot_cli.shutil.which", return_value=r"C:\fake\copilot.exe")
def test_run_copilot_falls_back_on_timeout_with_duration(mock_which, mock_popen, mock_taskkill_run):
    process = _mock_process()
    process.communicate.side_effect = subprocess.TimeoutExpired(cmd="copilot", timeout=240)
    mock_popen.return_value = process
    result = run_copilot("some prompt", timeout=240)
    assert result.startswith(CLI_FAILED_PREFIX)
    assert "240s" in result


@patch("ai.copilot_cli.subprocess.run")  # the taskkill call inside _kill_process_tree
@patch("ai.copilot_cli.subprocess.Popen")
@patch("ai.copilot_cli.shutil.which", return_value=r"C:\fake\copilot.exe")
def test_run_copilot_kills_whole_process_tree_on_timeout(mock_which, mock_popen, mock_taskkill_run):
    process = _mock_process(pid=4321)
    process.communicate.side_effect = subprocess.TimeoutExpired(cmd="copilot", timeout=240)
    mock_popen.return_value = process
    run_copilot("some prompt")
    taskkill_args = mock_taskkill_run.call_args.args[0]
    assert taskkill_args == ["taskkill", "/F", "/T", "/PID", "4321"]
