import logging
from unittest.mock import MagicMock, patch

import requests

from ai.ollama_client import FAILED_MESSAGE, run_ollama


def _fake_response(content):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"message": {"content": content}}
    return mock_response


@patch("requests.post")
def test_run_ollama_returns_content_on_success(mock_post):
    mock_post.return_value = _fake_response("Looks fine.")
    result = run_ollama("some prompt", model="gemma4:12b")
    assert result == "Looks fine."
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["model"] == "gemma4:12b"
    assert call_kwargs["json"]["messages"] == [{"role": "user", "content": "some prompt"}]
    assert call_kwargs["json"]["stream"] is False


def test_run_ollama_defaults_to_thinking_disabled():
    # Direct finding, 2026-08-29: Gemma 4's default thinking-enabled mode
    # took 8-13+ minutes per call and once never even reached a final
    # answer — a category mismatch for a "quick and correct" clerk job.
    # think=False must be the default so a caller never has to remember
    # to opt out of it.
    with patch("requests.post") as mock_post:
        mock_post.return_value = _fake_response("ok")
        run_ollama("some prompt", model="gemma4:12b")
        assert mock_post.call_args.kwargs["json"]["think"] is False


def test_run_ollama_can_opt_into_thinking():
    with patch("requests.post") as mock_post:
        mock_post.return_value = _fake_response("ok")
        run_ollama("some prompt", model="gemma4:12b", think=True)
        assert mock_post.call_args.kwargs["json"]["think"] is True


def test_run_ollama_omits_keep_alive_by_default():
    # None (the default) means "take Ollama's own default" -- must not
    # send a literal null/None value in the request body for it.
    with patch("requests.post") as mock_post:
        mock_post.return_value = _fake_response("ok")
        run_ollama("some prompt", model="gemma4:12b")
        assert "keep_alive" not in mock_post.call_args.kwargs["json"]


def test_run_ollama_passes_keep_alive_when_given():
    # Real fix, 2026-09-14: a caller making many back-to-back calls to
    # the same model (ai.researcher's ~17-symbol run) needs the model to
    # stay resident between calls -- confirmed live, a cold reload took
    # 75.5s vs 2.6s warm.
    with patch("requests.post") as mock_post:
        mock_post.return_value = _fake_response("ok")
        run_ollama("some prompt", model="gemma4:12b", keep_alive="30m")
        assert mock_post.call_args.kwargs["json"]["keep_alive"] == "30m"


@patch("requests.post", side_effect=RuntimeError("something unexpected"))
def test_run_ollama_falls_back_on_unexpected_exception(mock_post):
    # A genuinely unanticipated exception type must still never raise —
    # caught by the final catch-all, not the specific requests/JSON
    # exception branches.
    assert run_ollama("some prompt", model="gemma4:12b") == FAILED_MESSAGE


@patch("requests.post", side_effect=requests.exceptions.ConnectionError("could not connect"))
def test_run_ollama_falls_back_on_request_exception(mock_post):
    # This is the real, expected failure mode when the local Ollama
    # server isn't running at all.
    assert run_ollama("some prompt", model="gemma4:12b") == FAILED_MESSAGE


@patch("requests.post", side_effect=requests.exceptions.ConnectionError("could not connect"))
def test_run_ollama_logs_request_exception_reason_and_model(mock_post, caplog):
    with caplog.at_level(logging.WARNING):
        run_ollama("some prompt", model="qwen2.5:7b")
    assert "qwen2.5:7b" in caplog.text
    assert "could not connect" in caplog.text.lower()


def _http_error_response(status_code, body):
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.text = body
    error = requests.exceptions.HTTPError(response=mock_response)
    mock_response.raise_for_status.side_effect = error
    return mock_response


@patch("requests.post")
def test_run_ollama_falls_back_on_http_error(mock_post):
    # E.g. the model named isn't pulled on this machine (404).
    mock_post.return_value = _http_error_response(404, "model not found")
    assert run_ollama("some prompt", model="gemma4:12b") == FAILED_MESSAGE


@patch("requests.post")
def test_run_ollama_falls_back_on_malformed_response_shape(mock_post):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"unexpected": "shape"}
    mock_post.return_value = mock_response
    assert run_ollama("some prompt", model="gemma4:12b") == FAILED_MESSAGE


@patch("requests.post")
def test_run_ollama_logs_malformed_shape_reason(mock_post, caplog):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"unexpected": "shape"}
    mock_post.return_value = mock_response
    with caplog.at_level(logging.WARNING):
        run_ollama("some prompt", model="gemma4:12b")
    assert "unexpected response shape" in caplog.text.lower()
    assert "gemma4:12b" in caplog.text


@patch("requests.post")
def test_run_ollama_falls_back_on_empty_content(mock_post):
    mock_post.return_value = _fake_response("   ")
    assert run_ollama("some prompt", model="gemma4:12b") == FAILED_MESSAGE


@patch("requests.post")
def test_run_ollama_logs_empty_content_reason(mock_post, caplog):
    mock_post.return_value = _fake_response("   ")
    with caplog.at_level(logging.WARNING):
        run_ollama("some prompt", model="gemma4:12b")
    assert "empty content" in caplog.text.lower()
    assert "gemma4:12b" in caplog.text
