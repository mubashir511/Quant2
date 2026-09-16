import logging

logger = logging.getLogger(__name__)

API_URL = "http://localhost:11434/api/chat"

FAILED_MESSAGE = (
    "Ollama response unavailable right now (the local server could not be "
    "reached, the model isn't pulled, or it returned an error)."
)


def run_ollama(prompt: str, model: str, timeout: int = 120, think: bool = False, keep_alive: str | None = None) -> str:
    """Plain chat call against a locally-running Ollama server — no API
    key, no cost, no shared rate limit, but also no built-in web search
    (unlike ai/claude_cli.py's CLI). Same "never raises, degrades to a
    clear, unchanging fallback message" resilience pattern as
    ai/openrouter_client.py's run_openrouter, so a caller can match on
    FAILED_MESSAGE the same way.

    `think` defaults to False: live-tested 2026-08-29 against Gemma 4
    (a genuine reasoning model) with the default (thinking-enabled)
    behavior — a single tactical-defense-style call took 8-13+ minutes
    and, once, never even reached a final answer within a generous
    token budget (all of it spent on the internal "thinking" trace).
    With thinking disabled the same model answered correctly in
    30-70s, in line with a non-reasoning model's own latency — "quick
    and correct" was the explicit design goal here (see
    ai/clerk_execution.py's own module docstring), not "eventually
    correct after extended deliberation." A caller that genuinely wants
    a model's reasoning trace can still pass think=True and read
    `message.thinking` itself; this function only ever returns
    `message.content`.

    `keep_alive` (Ollama's own request field — a duration string like
    "30m", or None to take Ollama's own default) controls how long the
    model stays resident in memory after this call before Ollama unloads
    it. None (Ollama's own default, ~5 minutes) is fine for a single
    isolated call, but a real, live-observed cost for a caller making
    MANY calls to the SAME model with real work (a network fetch, a
    file write) in between each one: a genuinely idle gap over that
    default window means the NEXT call pays a full cold-reload penalty —
    confirmed live 2026-09-14, a trivial "say OK" prompt took 75.5s cold
    vs 2.6s once already warm, a ~29x difference. ai.researcher passes a
    longer keep_alive across its own ~17-symbol run specifically to
    avoid paying that penalty repeatedly."""
    import requests

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": think,
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    try:
        response = requests.post(API_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        content = data["message"]["content"]
    except requests.exceptions.RequestException as e:
        logger.warning("Ollama request failed for %s: %s: %s", model, type(e).__name__, e)
        return FAILED_MESSAGE
    except (ValueError, KeyError, TypeError) as e:
        logger.warning("Ollama returned an unexpected response shape for %s: %s", model, e)
        return FAILED_MESSAGE
    except Exception as e:
        # Catch-all so "never raises" holds even for a genuinely
        # unanticipated failure — still logged with enough detail to
        # diagnose rather than silently degrading.
        logger.warning("Ollama call failed unexpectedly for %s: %s: %s", model, type(e).__name__, e)
        return FAILED_MESSAGE

    if not content or not content.strip():
        logger.warning("Ollama returned empty content for %s", model)
        return FAILED_MESSAGE

    return content.strip()
