from contextlib import redirect_stdout
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import huggingface_auth as auth


class HuggingFaceAuthTests(unittest.TestCase):
    def test_existing_login_never_prompts(self):
        with patch.object(auth, "get_token", return_value="test-token"), patch.object(
            auth.getpass, "getpass"
        ) as prompt, patch.object(auth, "login") as login, redirect_stdout(io.StringIO()):
            auth.ensure_login()
        prompt.assert_not_called()
        login.assert_not_called()

    def test_missing_login_prompts_with_hidden_input_and_saves(self):
        output = io.StringIO()
        with patch.object(auth, "get_token", return_value=None), patch.object(
            auth, "has_terminal", return_value=True
        ), patch.object(auth.getpass, "getpass", return_value=" test-token ") as prompt, patch.object(
            auth, "login"
        ) as login, redirect_stdout(output):
            auth.ensure_login()
        prompt.assert_called_once()
        login.assert_called_once_with(token="test-token", add_to_git_credential=False)
        self.assertNotIn("test-token", output.getvalue())

    def test_noninteractive_session_fails_without_echoing_a_token(self):
        with patch.object(auth, "get_token", return_value=None), patch.object(
            auth, "has_terminal", return_value=False
        ), patch.object(auth.getpass, "getpass") as prompt:
            with self.assertRaisesRegex(RuntimeError, "hf auth login"):
                auth.ensure_login()
        prompt.assert_not_called()

    def test_cache_redirection_preserves_login_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            auth.preserve_login_location()
            os.environ["XDG_CACHE_HOME"] = directory
            self.assertEqual(os.environ["HF_HOME"], auth.HF_HOME)
            self.assertEqual(os.environ["HF_TOKEN_PATH"], auth.HF_TOKEN_PATH)
            os.environ["HF_HOME"] = "custom-home"
            os.environ["HF_TOKEN_PATH"] = "custom-token-path"
            auth.preserve_login_location()
            self.assertEqual(os.environ["HF_HOME"], "custom-home")
            self.assertEqual(os.environ["HF_TOKEN_PATH"], "custom-token-path")
