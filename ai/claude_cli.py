import shutil
import subprocess

CLI_MISSING_MESSAGE = (
    "AI response unavailable: the `claude` CLI isn't on PATH. Install "
    "Claude Code and make sure `claude` is reachable from this environment."
)

CLI_FAILED_MESSAGE = (
    "AI response unavailable right now (the local `claude` CLI could not be "
    "reached or returned an error)."
)


def run_claude(
    prompt: str,
    timeout: int = 60,
    allowed_tools: list[str] | None = None,
    model: str | None = None,
) -> str:
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

    try:
        # Pass the (multi-line) prompt over stdin rather than as a CLI arg —
        # batch-file argument forwarding on Windows can mangle embedded
        # newlines in a single argv item.
        result = subprocess.run(
            args,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return CLI_FAILED_MESSAGE

    if result.returncode != 0 or not result.stdout.strip():
        return CLI_FAILED_MESSAGE

    return result.stdout.strip()
