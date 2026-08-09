import logging

logger = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"

MISSING_KEY_MESSAGE = (
    "OpenRouter response unavailable: OPENROUTER_API_KEY isn't configured."
)

FAILED_MESSAGE = (
    "OpenRouter response unavailable right now (the API could not be "
    "reached, hit a rate limit, or returned an error)."
)


def run_openrouter(prompt: str, model: str, timeout: int = 90) -> str:
    """Plain chat-completions HTTP call — unlike ai/claude_cli.py and
    ai/gemini_cli.py, OpenRouter is a raw model API, not an agentic CLI:
    no built-in web search, no subprocess involved. Same resilience
    pattern as the CLI modules: never raises, degrades to a clear,
    unchanging fallback message on any failure (callers, e.g.
    build_audit_block, match on these exact constants) — but logs the
    *actual* reason first (status code + response body, or the specific
    exception), so a future failure is diagnosable from the console/log
    instead of needing a live investigation each time. Confirmed live
    that OpenRouter's free-tier models can 429 from a shared pool
    contended by other users — a very different cause from a bad key or a
    network outage, and impossible to tell apart without this."""
    import config
    import requests

    if not config.OPENROUTER_API_KEY:
        logger.warning("OpenRouter call skipped for %s: no API key configured", model)
        return MISSING_KEY_MESSAGE

    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        body = e.response.text[:500] if e.response is not None else ""
        logger.warning(
            "OpenRouter returned an error response for %s: status=%s body=%s", model, status, body
        )
        return FAILED_MESSAGE
    except requests.exceptions.RequestException as e:
        logger.warning("OpenRouter request failed for %s: %s: %s", model, type(e).__name__, e)
        return FAILED_MESSAGE
    except (ValueError, KeyError, IndexError, TypeError) as e:
        logger.warning("OpenRouter returned an unexpected response shape for %s: %s", model, e)
        return FAILED_MESSAGE
    except Exception as e:
        # Catch-all so "never raises" holds even for a genuinely
        # unanticipated failure — still logged with enough detail to
        # diagnose rather than silently degrading.
        logger.warning("OpenRouter call failed unexpectedly for %s: %s: %s", model, type(e).__name__, e)
        return FAILED_MESSAGE

    if not content or not content.strip():
        logger.warning("OpenRouter returned empty content for %s", model)
        return FAILED_MESSAGE

    return content.strip()
