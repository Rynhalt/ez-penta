from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from local_server import local_server


class LocalServerTests(unittest.TestCase):
    def test_busy_port_never_launches_or_stops_an_unowned_process(self):
        with patch("local_server.chat.server_status", return_value="ready"), patch(
            "local_server.subprocess.Popen"
        ) as start, patch("local_server.chat.stop_owned_server") as stop:
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                with local_server("model", 8080, Path("unused.log")):
                    self.fail("Busy port should not become available")
        start.assert_not_called()
        stop.assert_not_called()

    def test_guard_configuration_and_cleanup_after_failure(self):
        process = Mock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory, patch(
            "local_server.chat.server_status", side_effect=["offline", "ready"]
        ), patch("local_server.shutil.which", return_value="llama-server"), patch(
            "local_server.subprocess.Popen", return_value=process
        ) as start, patch("local_server.chat.stop_owned_server") as stop:
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                with local_server("guard-model", 8081, Path(directory) / "guard.log", guard=True):
                    raise RuntimeError("generation failed")
        self.assertIn("--no-prefill-assistant", start.call_args.args[0])
        self.assertIn("--no-context-shift", start.call_args.args[0])
        self.assertIn("-hf", start.call_args.args[0])
        self.assertNotIn("--gpu-layers", start.call_args.args[0])
        stop.assert_called_once_with(process)

    def test_explicit_model_file_and_gpu_offload(self):
        process = Mock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory, patch(
            "local_server.chat.server_status", side_effect=["offline", "ready"]
        ), patch("local_server.shutil.which", return_value="llama-server"), patch(
            "local_server.subprocess.Popen", return_value=process
        ) as start, patch("local_server.chat.stop_owned_server") as stop:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()
            with local_server("model-alias", 8080, Path(directory) / "server.log",
                              model_path=model_path, gpu_layers="all"):
                command = start.call_args.args[0]
                self.assertEqual(command[1:5], ["-m", str(model_path), "--alias", "model-alias"])
                self.assertNotIn("-hf", command)
                self.assertEqual(command[-2:], ["--gpu-layers", "all"])
            stop.assert_called_once_with(process)

    def test_missing_model_file_does_not_start_a_process(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "local_server.chat.server_status", return_value="offline"
        ), patch("local_server.shutil.which", return_value="llama-server"), patch(
            "local_server.subprocess.Popen"
        ) as start:
            with self.assertRaises(FileNotFoundError):
                with local_server("model", 8080, Path(directory) / "server.log",
                                  model_path=Path(directory) / "missing.gguf"):
                    self.fail("Missing model must not launch a server")
            start.assert_not_called()

    def test_failed_startup_still_cleans_up_owned_process(self):
        process = Mock()
        process.poll.return_value = 1
        with tempfile.TemporaryDirectory() as directory, patch(
            "local_server.chat.server_status", return_value="offline"
        ), patch("local_server.shutil.which", return_value="llama-server"), patch(
            "local_server.subprocess.Popen", return_value=process
        ), patch("local_server.chat.stop_owned_server") as stop:
            with self.assertRaisesRegex(RuntimeError, "exited"):
                with local_server("model", 8080, Path(directory) / "server.log"):
                    self.fail("Exited process cannot serve requests")
        stop.assert_called_once_with(process)
