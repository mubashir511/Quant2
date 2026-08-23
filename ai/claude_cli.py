import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

CLI_MISSING_MESSAGE = (
    "AI response unavailable: the `claude` CLI isn't on PATH. Install "
    "Claude Code and make sure `claude` is reachable from this environment."
)

# Every CLI-failure message returned by run_claude starts with this exact
# text — callers must check with .startswith(CLI_FAILED_PREFIX), not
# equality, since the rest of the message now carries the actual failure
# reason (exit code, real stderr, timeout length) instead of being a fixed
# placeholder that discarded that information.
CLI_FAILED_PREFIX = "AI response unavailable right now"

_STDERR_PREVIEW_CHARS = 500


def _failed(detail: str) -> str:
    return f"{CLI_FAILED_PREFIX} ({detail})."


def _kill_process_tree(pid: int) -> None:
    # Windows-only (this whole project is, via the MetaTrader5 dependency).
    # A bare Popen.kill() only terminates the immediate process — the npm
    # .cmd shim — not any node.exe child it spawned. Confirmed live against
    # the sibling `gemini` CLI: a timed-out call left an orphaned, near-idle
    # node.exe running for over 10 minutes past its nominal timeout, still
    # holding the stdout pipe open, which made communicate() hang
    # indefinitely waiting for EOF that would never come. taskkill /T
    # recurses the whole process tree instead of just the top-level shim.
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)


def run_claude(
    prompt: str,
    timeout: int = 60,
    allowed_tools: list[str] | None = None,
    model: str | None = None,
    session_id: str | None = None,
    resume_session_id: str | None = None,
) -> str:
    """`session_id` and `resume_session_id` let a caller chain two calls
    into one real conversation instead of two independent, fully self-
    contained ones — pass a caller-generated UUID as `session_id` on the
    first call, then the SAME UUID as `resume_session_id` on a later call
    to continue it. Live-verified (including with real WebSearch results,
    not just plain text) that a resumed call correctly retains the exact
    facts/figures found in the first call without needing them repeated —
    this is what lets a caller send a much smaller follow-up prompt (only
    the genuinely NEW content) instead of re-sending unchanging context a
    second time. `resume_session_id` takes priority if both are somehow
    given (a call is either starting a session or continuing one, not
    both)."""
    # On Windows, `claude` resolves to an npm .cmd shim that subprocess.run
    # won't launch by bare name without shell=True; resolving the full path
    # via shutil.which (which honors PATHEXT) sidesteps that.
    executable = shutil.which("claude")
    if executable is None:
        return CLI_MISSING_MESSAGE

    args = [executable, "-p"]
    if allowed_tools:
        args += ["--allowedTools", *allowed_tools]
    if model:
        args += ["--model", model]
    if resume_session_id:
        args += ["--resume", resume_session_id]
    elif session_id:
        args += ["--session-id", session_id]

    try:
        # Pass the (multi-line) prompt over stdin rather than as a CLI arg —
        # batch-file argument forwarding on Windows can mangle embedded
        # newlines in a single argv item.
        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            # Same defensive isolation as ai/copilot_cli.py's own Popen
            # call, added after a real crash found there 2026-08-23 (two
            # child CLI processes and their own parent Python process all
            # died together from a Windows console-level signal) — this
            # keeps a crash or console event on the parent's console from
            # cascading into this child, and vice versa.
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    except (FileNotFoundError, OSError) as e:
        logger.warning("run_claude: failed to launch %s: %s: %s", executable, type(e).__name__, e)
        return _failed(f"failed to launch the `claude` CLI: {type(e).__name__}: {e}")

    try:
        stdout, stderr = process.communicate(input=prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        # Uses Popen directly (not subprocess.run's timeout=) so this can
        # kill the *whole* process tree via taskkill /T on timeout, instead
        # of just the immediate .cmd shim — see _kill_process_tree.
        _kill_process_tree(process.pid)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        logger.warning("run_claude: timed out after %ds", timeout)
        return _failed(f"the `claude` CLI did not respond within {timeout}s and was terminated")

    if process.returncode != 0 or not stdout.strip():
        stderr_text = stderr.strip() if stderr else ""
        logger.warning(
            "run_claude: exited with code %s, stderr=%r", process.returncode, stderr_text
        )
        if stderr_text:
            return _failed(
                f"the `claude` CLI exited with code {process.returncode}: "
                f"{stderr_text[:_STDERR_PREVIEW_CHARS]}"
            )
        return _failed(
            f"the `claude` CLI exited with code {process.returncode} and produced no output"
        )

    return stdout.strip()
