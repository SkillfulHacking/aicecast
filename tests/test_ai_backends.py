"""Tests for the AI backend dispatch logic in helpers.py."""

import unittest
from unittest.mock import MagicMock, patch

from mac.content_generator import helpers


class RunClaudeBackendDispatchTests(unittest.TestCase):
    """Verify that run_claude() dispatches to the correct backend."""

    def _make_backend(self, return_value):
        """Return a mock backend function that returns *return_value*."""
        mock = MagicMock(return_value=return_value)
        return mock

    # ------------------------------------------------------------------
    # Default (claude) backend
    # ------------------------------------------------------------------

    def test_default_backend_uses_claude(self):
        mock_claude = self._make_backend("hello world from claude")
        import os
        env = {k: v for k, v in os.environ.items() if k != "WRIT_AI_BACKEND"}
        with patch.dict("os.environ", env, clear=True):
            with patch.dict(helpers._AI_BACKENDS, {"claude": mock_claude}):
                result = helpers.run_claude("prompt")
        self.assertEqual(result, "hello world from claude")
        mock_claude.assert_called_once()

    def test_explicit_claude_backend(self):
        mock_claude = self._make_backend("from claude")
        with patch.dict("os.environ", {"WRIT_AI_BACKEND": "claude"}):
            with patch.dict(helpers._AI_BACKENDS, {"claude": mock_claude}):
                result = helpers.run_claude("prompt")
        self.assertEqual(result, "from claude")
        mock_claude.assert_called_once()

    # ------------------------------------------------------------------
    # OpenAI backend
    # ------------------------------------------------------------------

    def test_openai_backend_is_selected(self):
        mock_openai = self._make_backend("from openai")
        with patch.dict("os.environ", {"WRIT_AI_BACKEND": "openai"}):
            with patch.dict(helpers._AI_BACKENDS, {"openai": mock_openai}):
                result = helpers.run_claude("prompt")
        self.assertEqual(result, "from openai")
        mock_openai.assert_called_once()

    def test_openai_backend_passes_model(self):
        mock_openai = self._make_backend("ok")
        with patch.dict("os.environ", {"WRIT_AI_BACKEND": "openai"}):
            with patch.dict(helpers._AI_BACKENDS, {"openai": mock_openai}):
                helpers.run_claude("prompt", model="gpt-4.1")
        _, kwargs = mock_openai.call_args
        self.assertEqual(kwargs.get("model"), "gpt-4.1")

    # ------------------------------------------------------------------
    # Codex backend
    # ------------------------------------------------------------------

    def test_codex_backend_is_selected(self):
        mock_codex = self._make_backend("from codex")
        with patch.dict("os.environ", {"WRIT_AI_BACKEND": "codex"}):
            with patch.dict(helpers._AI_BACKENDS, {"codex": mock_codex}):
                result = helpers.run_claude("prompt")
        self.assertEqual(result, "from codex")
        mock_codex.assert_called_once()

    # ------------------------------------------------------------------
    # Unknown backend falls back to claude
    # ------------------------------------------------------------------

    def test_unknown_backend_falls_back_to_claude(self):
        mock_claude = self._make_backend("fallback")
        with patch.dict("os.environ", {"WRIT_AI_BACKEND": "notabackend"}):
            with patch.dict(helpers._AI_BACKENDS, {"claude": mock_claude}):
                result = helpers.run_claude("prompt")
        self.assertEqual(result, "fallback")
        mock_claude.assert_called_once()

    # ------------------------------------------------------------------
    # min_length filtering
    # ------------------------------------------------------------------

    def test_min_length_filters_short_responses(self):
        mock_claude = self._make_backend("hi")
        with patch.dict("os.environ", {"WRIT_AI_BACKEND": "claude"}):
            with patch.dict(helpers._AI_BACKENDS, {"claude": mock_claude}):
                result = helpers.run_claude("prompt", min_length=10)
        self.assertIsNone(result)

    def test_none_from_backend_returns_none(self):
        mock_claude = self._make_backend(None)
        with patch.dict("os.environ", {"WRIT_AI_BACKEND": "claude"}):
            with patch.dict(helpers._AI_BACKENDS, {"claude": mock_claude}):
                result = helpers.run_claude("prompt")
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # _run_openai_api: missing API key
    # ------------------------------------------------------------------

    def test_openai_api_returns_none_without_api_key(self):
        import os
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
            result = helpers._run_openai_api("prompt", timeout=5, model=None)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # _run_codex_cli: missing binary
    # ------------------------------------------------------------------

    def test_codex_cli_returns_none_when_not_installed(self):
        with patch("shutil.which", return_value=None):
            result = helpers._run_codex_cli("prompt", timeout=5, model=None)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # Backend is case-insensitive
    # ------------------------------------------------------------------

    def test_backend_name_is_case_insensitive(self):
        mock_openai = self._make_backend("case insensitive")
        with patch.dict("os.environ", {"WRIT_AI_BACKEND": "OpenAI"}):
            with patch.dict(helpers._AI_BACKENDS, {"openai": mock_openai}):
                result = helpers.run_claude("prompt")
        self.assertEqual(result, "case insensitive")


if __name__ == "__main__":
    unittest.main()
