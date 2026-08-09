import logging
from unittest.mock import MagicMock, patch

import requests

from ai.openrouter_client import FAILED_MESSAGE, MISSING_KEY_MESSAGE, run_openrouter


def _fake_response(content):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return mock_response


@patch("requests.post")
@patch("config.OPENROUTER_API_KEY", "fake-key")
def test_run_openrouter_returns_content_on_success(mock_post):
    mock_post.return_value = _fake_response("Looks fine.")
    result = run_openrouter("some prompt", model="openai/gpt-oss-20b:free")
    assert result == "Looks fine."
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["headers"]["Authorization"] == "Bearer fake-key"
    assert call_kwargs["json"]["model"] == "openai/gpt-oss-20b:free"
    assert call_kwargs["json"]["messages"] == [{"role": "user", "content": "some prompt"}]


@patch("config.OPENROUTER_API_KEY", None)
def test_run_openrouter_reports_missing_key_without_calling_requests():
    assert run_openrouter("some prompt", model="openai/gpt-oss-20b:free") == MISSING_KEY_MESSAGE


@patch("config.OPENROUTER_API_KEY", None)
def test_run_openrouter_logs_missing_key_reason(caplog):
    with caplog.at_level(logging.WARNING):
        run_openrouter("some prompt", model="openai/gpt-oss-20b:free")
    assert "no api key" in caplog.text.lower()
    assert "openai/gpt-oss-20b:free" in caplog.text


@patch("requests.post", side_effect=RuntimeError("network down"))
@patch("config.OPENROUTER_API_KEY", "fake-key")
def test_run_openrouter_falls_back_on_unexpected_exception(mock_post):
    # A genuinely unanticipated exception type must still never raise —
    # caught by the final catch-all, not the specific requests/JSON
    # exception branches.
    assert run_openrouter("some prompt", model="openai/gpt-oss-20b:free") == FAILED_MESSAGE


@patch("requests.post", side_effect=requests.exceptions.ConnectionError("could not connect"))
@patch("config.OPENROUTER_API_KEY", "fake-key")
def test_run_openrouter_falls_back_on_request_exception(mock_post):
    assert run_openrouter("some prompt", model="openai/gpt-oss-20b:free") == FAILED_MESSAGE


@patch("requests.post", side_effect=requests.exceptions.ConnectionError("could not connect"))
@patch("config.OPENROUTER_API_KEY", "fake-key")
def test_run_openrouter_logs_request_exception_reason_and_model(mock_post, caplog):
    with caplog.at_level(logging.WARNING):
        run_openrouter("some prompt", model="nvidia/nemotron-3-ultra-550b-a55b:free")
    assert "nvidia/nemotron-3-ultra-550b-a55b:free" in caplog.text
    assert "could not connect" in caplog.text.lower()


def _http_error_response(status_code, body):
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.text = body
    error = requests.exceptions.HTTPError(response=mock_response)
    mock_response.raise_for_status.side_effect = error
    return mock_response


@patch("requests.post")
@patch("config.OPENROUTER_API_KEY", "fake-key")
def test_run_openrouter_falls_back_on_http_error(mock_post):
    mock_post.return_value = _http_error_response(429, "rate limited")
    assert run_openrouter("some prompt", model="openai/gpt-oss-20b:free") == FAILED_MESSAGE


@patch("requests.post")
@patch("config.OPENROUTER_API_KEY", "fake-key")
def test_run_openrouter_logs_http_error_status_and_body(mock_post, caplog):
    mock_post.return_value = _http_error_response(
        429, "openai/gpt-oss-20b:free is temporarily rate-limited upstream"
    )
    with caplog.at_level(logging.WARNING):
        run_openrouter("some prompt", model="openai/gpt-oss-20b:free")
    assert "429" in caplog.text
    assert "rate-limited upstream" in caplog.text
    assert "openai/gpt-oss-20b:free" in caplog.text


@patch("requests.post")
@patch("config.OPENROUTER_API_KEY", "fake-key")
def test_run_openrouter_falls_back_on_malformed_response_shape(mock_post):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"unexpected": "shape"}
    mock_post.return_value = mock_response
    assert run_openrouter("some prompt", model="openai/gpt-oss-20b:free") == FAILED_MESSAGE


@patch("requests.post")
@patch("config.OPENROUTER_API_KEY", "fake-key")
def test_run_openrouter_logs_malformed_shape_reason(mock_post, caplog):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"unexpected": "shape"}
    mock_post.return_value = mock_response
    with caplog.at_level(logging.WARNING):
        run_openrouter("some prompt", model="openai/gpt-oss-20b:free")
    assert "unexpected response shape" in caplog.text.lower()
    assert "openai/gpt-oss-20b:free" in caplog.text


@patch("requests.post")
@patch("config.OPENROUTER_API_KEY", "fake-key")
def test_run_openrouter_falls_back_on_empty_content(mock_post):
    mock_post.return_value = _fake_response("   ")
    assert run_openrouter("some prompt", model="openai/gpt-oss-20b:free") == FAILED_MESSAGE


@patch("requests.post")
@patch("config.OPENROUTER_API_KEY", "fake-key")
def test_run_openrouter_logs_empty_content_reason(mock_post, caplog):
    mock_post.return_value = _fake_response("   ")
    with caplog.at_level(logging.WARNING):
        run_openrouter("some prompt", model="openai/gpt-oss-20b:free")
    assert "empty content" in caplog.text.lower()
    assert "openai/gpt-oss-20b:free" in caplog.text
