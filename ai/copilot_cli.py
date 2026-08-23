import logging
import os
import shutil
import subprocess
import tempfile

logger = logging.getLogger(__name__)

CLI_MISSING_MESSAGE = (
    "Copilot CLI response unavailable: the `copilot` CLI isn't on PATH or "
    "isn't authenticated (run `copilot` and complete `/login`, or set "
    "GH_TOKEN/GITHUB_TOKEN)."
)

# Every CLI-failure message returned by run_copilot starts with this exact
# text — callers must check with .startswith(CLI_FAILED_PREFIX), not
# equality, matching ai/claude_cli.py's own convention.
CLI_FAILED_PREFIX = "Copilot CLI response unavailable right now"

_STDERR_PREVIEW_CHARS = 500


def _failed(detail: str) -> str:
    return f"{CLI_FAILED_PREFIX} ({detail})."


def _kill_process_tree(pid: int) -> None:
    # Same defensive measure as ai/claude_cli.py's sibling helper — cheap
    # insurance against an orphaned child process outliving a timeout, a
    # real bug this project has hit before with a different CLI tool.
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)


def run_copilot(prompt: str, timeout: int = 240) -> str:
    """Runs GitHub Copilot CLI non-interactively (`copilot -p ...`) with
    full tool/URL permissions.

    The actual prompt text is written to a temp file and Copilot is told
    (via a short, fixed, ASCII-only wrapper instruction) to read and
    carry it out — NOT passed directly as the `-p` argv value. Confirmed
    live in real production use that copilot.exe's own argument parser
    can reject a real prompt with "Invalid command format... your prompt
    was not quoted" even though Python's subprocess module escapes it
    according to the standard Windows/MSVCRT convention — a targeted
    battery of synthetic tests (embedded quotes, trailing backslashes,
    nested quotes, em-dashes) couldn't reproduce the exact trigger in
    isolation, so rather than guess at one specific character pattern,
    this sidesteps command-line quoting for the actual (arbitrary, not
    fully under our control) prompt content entirely. Live-verified this
    correctly round-trips real content containing embedded quotes and a
    trailing backslash that a direct argv pass could not be guaranteed
    safe for. `--allow-all-paths` is deliberately NOT requested — a temp
    file is already within what `--allow-all-tools` alone permits
    reading, confirmed live, so this doesn't grant broader filesystem
    access than the previous direct-argv approach needed.

    Never raises: returns CLI_MISSING_MESSAGE if the binary isn't found or
    reports no auth, or f"{CLI_FAILED_PREFIX} (...)." with the real reason
    on any other failure — same resilience contract as run_claude/
    run_openrouter, so callers can match on these exact sentinels."""
    executable = shutil.which("copilot")
    if executable is None:
        return CLI_MISSING_MESSAGE

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(prompt)
            tmp_path = f.name

        wrapper_prompt = (
            f"Read the file at {tmp_path} using your file-read tool — it "
            "contains your real task. Carry it out exactly as written "
            "there and respond accordingly."
        )
        args = [
            executable, "-p", wrapper_prompt,
            "--allow-all-tools", "--allow-all-urls", "--silent",
        ]

        try:
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                # Real bug found live 2026-08-23: two PARALLEL Copilot
                # verdict calls (ai.copilot_execution runs one per not-
                # yet-settled Pending Setup via ThreadPoolExecutor) both
                # died at the same instant with Windows exit code
                # 3221225786 (0xC000013A, STATUS_CONTROL_C_EXIT) — the
                # code Windows reports when a CTRL_BREAK/CTRL_C reaches a
                # console process — and the PARENT python process running
                # copilot_execution_job.py died within the same few
                # seconds too (confirmed via job_lock's own PID-liveness
                # check finding it dead on the next poll). Without this
                # flag, a spawned child shares the parent's console
                # process group by default on Windows, so any console
                # signal (or a crash inside copilot.exe severe enough to
                # affect its own process group) can cascade to everything
                # else attached to that console — including this parent.
                # Isolating the child into its OWN process group can't
                # explain what originally triggers the signal, but it
                # does stop it from taking the parent down too, so a
                # crashed copilot.exe becomes the ordinary, already-
                # handled "CLI exited non-zero" failure path below instead
                # of an unrecoverable process-level crash that orphans
                # the execution lock.
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        except (FileNotFoundError, OSError) as e:
            logger.warning("run_copilot: failed to launch %s: %s: %s", executable, type(e).__name__, e)
            return _failed(f"failed to launch the `copilot` CLI: {type(e).__name__}: {e}")

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(process.pid)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            logger.warning("run_copilot: timed out after %ds", timeout)
            return _failed(f"the `copilot` CLI did not respond within {timeout}s and was terminated")

        if process.returncode != 0 or not stdout.strip():
            stderr_text = stderr.strip() if stderr else ""
            if "no authentication information found" in stderr_text.lower():
                logger.warning("run_copilot: not authenticated")
                return CLI_MISSING_MESSAGE
            logger.warning(
                "run_copilot: exited with code %s, stderr=%r", process.returncode, stderr_text
            )
            if stderr_text:
                return _failed(
                    f"the `copilot` CLI exited with code {process.returncode}: "
                    f"{stderr_text[:_STDERR_PREVIEW_CHARS]}"
                )
            return _failed(
                f"the `copilot` CLI exited with code {process.returncode} and produced no output"
            )

        return stdout.strip()
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.warning("run_copilot: failed to clean up temp file %s: %s", tmp_path, e)
