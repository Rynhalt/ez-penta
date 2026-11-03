"""Test persistent build artifacts without compiling code or accessing Drive."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from colab_build_cache import cache_key, restore_build, save_build, validate_cuda_server


class BuildCacheTests(unittest.TestCase):
    def make_binary_directory(self, directory):
        directory.mkdir(parents=True)
        (directory / "llama-server").write_bytes(b"test executable")
        (directory / "libggml.so.1").write_bytes(b"test shared library")
        (directory / "libggml.so").symlink_to("libggml.so.1")

    def test_round_trip_preserves_libraries_and_executable_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, restored = root / "source", root / "runtime" / "bin"
            self.make_binary_directory(source)
            identity = {"gpu": "T4", "llama_commit": "commit"}
            archive = root / "drive" / "build.zip"
            save_build(source, archive, identity)
            self.assertTrue(restore_build(archive, restored, identity))
            self.assertEqual((restored / "llama-server").read_bytes(), b"test executable")
            self.assertEqual((restored / "libggml.so").read_bytes(), b"test shared library")
            self.assertTrue((restored / "llama-server").stat().st_mode & 0o111)

    def test_environment_mismatch_never_replaces_local_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_binary_directory(root / "source")
            archive = root / "build.zip"
            save_build(root / "source", archive, {"gpu": "T4"})
            restored = root / "bin"
            restored.mkdir()
            (restored / "llama-server").write_bytes(b"existing")
            self.assertFalse(restore_build(archive, restored, {"gpu": "L4"}))
            self.assertEqual((restored / "llama-server").read_bytes(), b"existing")
            self.assertNotEqual(cache_key({"gpu": "T4"}), cache_key({"gpu": "L4"}))

    def test_corrupted_files_fail_before_replacing_local_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_binary_directory(root / "source")
            archive = root / "build.zip"
            identity = {"gpu": "T4"}
            save_build(root / "source", archive, identity)
            with zipfile.ZipFile(archive) as original:
                entries = {name: original.read(name) for name in original.namelist()}
            entries["libggml.so"] = b"corrupted bytes"
            with zipfile.ZipFile(archive, "w") as damaged:
                for name, data in entries.items():
                    damaged.writestr(name, data)
            restored = root / "bin"
            restored.mkdir()
            (restored / "llama-server").write_bytes(b"existing")
            with self.assertRaisesRegex(ValueError, "checksum"):
                restore_build(archive, restored, identity)
            self.assertEqual((restored / "llama-server").read_bytes(), b"existing")

    def test_interrupted_upload_preserves_existing_cache(self):
        def interrupted_copy(source, destination):
            destination.write_bytes(b"partial")
            raise OSError("connection interrupted")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_binary_directory(root / "source")
            archive = root / "drive" / "build.zip"
            save_build(root / "source", archive, {"gpu": "T4"})
            original = archive.read_bytes()
            with patch("colab_build_cache.shutil.copyfile", side_effect=interrupted_copy):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    save_build(root / "source", archive, {"gpu": "T4"})
            self.assertEqual(archive.read_bytes(), original)
            self.assertEqual(list(archive.parent.glob("*.partial")), [])

    def test_validation_checks_cuda_device_and_shared_library_search_path(self):
        with patch("colab_build_cache.subprocess.check_output", side_effect=[
            "Available devices:\n  CUDA0: Tesla T4 (15000 MiB)", "version: test",
        ]) as command:
            devices, version = validate_cuda_server(Path("/runtime/bin"))
            self.assertIn("CUDA0", devices)
            self.assertEqual(version, "version: test")
            self.assertTrue(command.call_args_list[0].kwargs["env"]["LD_LIBRARY_PATH"].startswith("/runtime/bin"))
        with patch("colab_build_cache.subprocess.check_output", return_value="No CUDA devices"):
            with self.assertRaisesRegex(RuntimeError, "did not detect"):
                validate_cuda_server(Path("/runtime/bin"))

    def test_notebook_restores_without_compiling_and_rebuilds_a_broken_cache(self):
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "answercarefully_colab.ipynb"
        notebook = json.loads(notebook_path.read_text())
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if "".join(cell["source"]).startswith("CACHE_BUILD_IN_DRIVE"))
        identity = {"gpu": "T4", "llama_commit": "commit"}
        for damaged in (False, True):
            with self.subTest(damaged=damaged), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                drive = root / "drive" / "MyDrive"
                archive = drive / "ez-penta-cache" / "llama-cuda" / f"{cache_key(identity)}.zip"
                self.make_binary_directory(root / "previous-build")
                save_build(root / "previous-build", archive, identity)
                if damaged:
                    archive.write_bytes(b"interrupted archive")
                colab = ModuleType("google.colab")
                colab.drive = SimpleNamespace(mount=lambda path: self.fail("Drive is already mounted"))
                namespace = {"Path": Path, "os": os, "shutil": shutil,
                             "subprocess": subprocess, "sys": sys}

                def fake_build(command, **kwargs):
                    if "--build" in command:
                        self.make_binary_directory(root / "llama.cpp" / "build" / "bin")

                cell = source.replace("/content", str(root))
                with patch.dict(sys.modules, {"google.colab": colab}), patch.dict(os.environ), patch(
                    "colab_build_cache.build_identity", return_value=identity
                ), patch("colab_build_cache.validate_cuda_server", return_value=("CUDA0: T4", "test")), patch(
                    "subprocess.run", side_effect=fake_build
                ) as run, redirect_stdout(io.StringIO()):
                    exec(cell, namespace)
                    self.assertEqual(namespace["llama_build_origin"], "built" if damaged else "drive")
                    if damaged:
                        self.assertTrue(any("--build" in call.args[0] for call in run.call_args_list))
                        self.assertTrue(restore_build(archive, root / "verify" / "bin", identity))
                    else:
                        run.assert_not_called()
                    # Subsequent experiments in the same runtime should also skip compilation.
                    run.reset_mock()
                    exec(cell, namespace)
                    self.assertEqual(namespace["llama_build_origin"], "runtime")
                    run.assert_not_called()
