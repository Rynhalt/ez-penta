"""Offline protocol, model-cache, cleanup and dependency tests for the GGUF backend."""

from contextlib import contextmanager
import hashlib
import io
import os
import subprocess
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from eval import dataset2_gguf as g


def small_spec(role="aya"):
    content = b"GGUF-test-weights"
    return {
        **g.MODEL_SPECS[role],
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }, content


class DownloadTests(unittest.TestCase):
    def test_interrupted_download_survives_new_runtime_directory(self):
        spec, content = small_spec()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(g.MODEL_SPECS, {"aya": spec}):
            root = Path(tmp)
            cache = root / "drive"
            def interrupted(spec, target):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.with_name(target.name + ".partial").write_bytes(content[:4])
                raise ConnectionError("disconnected")
            with self.assertRaises(ConnectionError):
                g.prepare_model("aya", cache, root / "runtime1", download=interrupted)
            def resume(spec, target):
                self.assertTrue(target.is_relative_to(cache))
                self.assertEqual(target.with_name(target.name + ".partial").read_bytes(), content[:4])
                target.write_bytes(content)
            result = g.prepare_model("aya", cache, root / "runtime2", download=resume)
            self.assertEqual(result.read_bytes(), content)
            g.prepare_model("aya", cache, root / "runtime3",
                            download=lambda *a: self.fail("Must reuse Drive cache"))

    def test_download_verify_resume_and_reuse(self):
        spec, content = small_spec()

        class Response(io.BytesIO):
            status = 206
            headers = {"Content-Range": f"bytes 4-{len(content)-1}/{len(content)}"}

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "model.gguf"
            destination.with_name("model.gguf.partial").write_bytes(content[:4])
            requests = []

            def open_url(request, timeout):
                requests.append(request)
                return Response(content[4:])

            g.download_model(spec, destination, open_url)
            self.assertEqual(destination.read_bytes(), content)
            self.assertEqual(requests[0].get_header("Range"), "bytes=4-")
            self.assertIn("/resolve/" + spec["revision"] + "/", requests[0].full_url)
            with patch.dict(g.MODEL_SPECS, {"aya": spec}):
                local = Path(tmp) / "local"
                target = local / "aya" / spec["filename"]
                target.parent.mkdir(parents=True)
                target.write_bytes(content)
                actual = g.prepare_model(
                    "aya",
                    Path(tmp) / "cache",
                    local,
                    download=lambda *a: self.fail("Should reuse local file"),
                )
                self.assertEqual(actual, target)

    def test_cache_copy_and_bad_override(self):
        spec, content = small_spec()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            g.MODEL_SPECS, {"aya": spec}
        ):
            root = Path(tmp)
            cached = root / "cache" / spec["sha256"][:16] / spec["filename"]
            cached.parent.mkdir(parents=True)
            cached.write_bytes(content)
            path = g.prepare_model(
                "aya",
                root / "cache",
                root / "local",
                download=lambda *a: self.fail("Should copy persistent cache"),
            )
            self.assertEqual(path.read_bytes(), content)
            bad = root / "bad.gguf"
            bad.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "does not match"):
                g.prepare_model("aya", None, root / "other", bad)

    def test_wrong_range_and_hash_rejected(self):
        spec, content = small_spec()

        class Response(io.BytesIO):
            status = 206
            headers = {"Content-Range": "bytes 9-19/20"}

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "m"
            destination.with_name("m.partial").write_bytes(content[:4])
            with self.assertRaisesRegex(ValueError, "byte range"):
                g.download_model(
                    spec, destination, lambda *a, **k: Response(content[4:])
                )
            destination.with_name("m.partial").write_bytes(b"X" * len(content))
            with self.assertRaisesRegex(ValueError, "checksum"):
                g.download_model(spec, destination)
            self.assertFalse(destination.exists())


class NativeProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model.gguf"
        self.spec, content = small_spec()
        self.path.write_bytes(content)
        self.build = {
            "llama_commit": g.LLAMA_COMMIT,
            "cmake_options": ["-DGGML_CUDA=ON"],
            "version": "test " + g.LLAMA_COMMIT[:7],
            "identity": {"gpu": "fixture"},
        }
        self.calls = []
        self.closed = []
        self.stop = "eos"
        self.truncated = False

        @contextmanager
        def launcher(*args, **kwargs):
            self.launch_options = kwargs
            self.launch_port = args[1]
            try:
                yield "http://fixture"
            finally:
                self.closed.append(True)

        self.launcher = launcher

    def read(self, url, data=None, timeout=10):
        self.calls.append((url, data))
        if url.endswith("/props"):
            return {
                "chat_template": "<BOS>{{ messages }}",
                "default_generation_settings": {"n_ctx": 8192},
            }
        if url.endswith("/apply-template"):
            return {"prompt": "<BOS>user: Hi\nassistant:"}
        if url.endswith("/tokenize"):
            self.assertEqual(data["add_special"], False)
            self.assertTrue(data["parse_special"])
            return {"tokens": [1, 3, 5]}
        if url.endswith("/completion"):
            self.assertEqual(data["prompt"], [1, 3, 5])
            self.assertEqual(data["temperature"], 0.0)
            self.assertFalse(data["cache_prompt"])
            self.assertTrue(data["return_tokens"])
            return {
                "content": "Hello",
                "tokens": [7, 2],
                "stop_type": self.stop,
                "truncated": self.truncated,
                "generation_settings": {"n_predict": data["n_predict"]},
            }
        raise AssertionError(url)

    def runner(self, role="aya"):
        spec = {
            **self.spec,
            **{
                k: g.MODEL_SPECS[role][k]
                for k in ["model_id", "filename", "quantization", "revision"]
            },
        }
        with patch.dict(g.MODEL_SPECS, {role: spec}):
            return g.GGUFRunner(
                role,
                self.path,
                self.build,
                self.tmp.name,
                read_json=self.read,
                launcher=self.launcher,
            )

    def test_native_template_tokens_greedy_eos_and_cleanup(self):
        r = self.runner()
        answer = r.answer({"prompt": "Hi"})
        self.assertEqual(answer["input_ids"], [1, 3, 5])
        self.assertEqual(answer["output_ids"], [7, 2])
        self.assertEqual(answer["status"], "completed")
        self.assertEqual(answer["output_tokens"], 2)
        self.assertEqual(r.configuration["backend"], "llama.cpp-gguf")
        self.assertEqual(r.configuration["quantization"], "Q4_K_M")
        self.assertEqual(self.launch_options["context_size"], 8192)
        r.close()
        self.assertEqual(self.closed, [True])

    def test_automatic_port_retries_collision_without_changing_config(self):
        from local_server import PortInUseError

        attempts = []

        @contextmanager
        def launcher(*args, **kwargs):
            attempts.append(args[1])
            if len(attempts) == 1:
                raise PortInUseError("occupied")
            yield "http://fixture"

        with patch.dict(g.MODEL_SPECS, {"aya": self.spec}), patch.object(
            g, "available_port", side_effect=[32101, 32102]
        ):
            r = g.GGUFRunner(
                "aya",
                self.path,
                self.build,
                self.tmp.name,
                read_json=self.read,
                launcher=launcher,
            )
        self.assertEqual(attempts, [32101, 32102])
        self.assertEqual(r.port, 32102)
        config = r.configuration
        r.close()
        with patch.dict(g.MODEL_SPECS, {"aya": self.spec}):
            other = g.GGUFRunner(
                "aya",
                self.path,
                self.build,
                self.tmp.name,
                port=32103,
                read_json=self.read,
                launcher=self.launcher,
            )
        self.assertEqual(other.configuration, config)
        other.close()

    def test_explicit_busy_port_does_not_silently_change(self):
        from local_server import PortInUseError

        @contextmanager
        def busy(*args, **kwargs):
            raise PortInUseError("occupied")
            yield

        with patch.dict(g.MODEL_SPECS, {"aya": self.spec}):
            with self.assertRaises(PortInUseError):
                g.GGUFRunner(
                    "aya",
                    self.path,
                    self.build,
                    self.tmp.name,
                    port=32100,
                    read_json=self.read,
                    launcher=busy,
                )

    def test_limit_and_context_shift_not_safe_completion(self):
        r = self.runner()
        self.stop = "limit"
        self.assertEqual(r.generate([])["status"], "truncated")
        self.truncated = True
        with self.assertRaisesRegex(ValueError, "truncated context"):
            r.generate([])
        r.close()

    def test_guard_launch_disables_prefill(self):
        r = self.runner("guard")
        self.assertTrue(self.launch_options["guard"])
        self.assertEqual(r.configuration["generation"]["n_predict"], 128)
        r.close()

    def test_partial_constructor_failure_releases_server(self):
        with patch.dict(g.MODEL_SPECS, {"aya": self.spec}):
            with self.assertRaises(ValueError):
                g.GGUFRunner(
                    "aya",
                    self.path,
                    self.build,
                    self.tmp.name,
                    read_json=lambda url: {},
                    launcher=self.launcher,
                )
        self.assertEqual(self.closed, [True])

    def test_context_overflow_rejected_before_generation(self):
        r = self.runner()
        r.context_size = 3
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            r.generate([])
        self.assertFalse(any(url.endswith("/completion") for url, data in self.calls))
        r.close()

    def test_auditkit_scores_precomputed_using_native_transport(self):
        from eval.dataset2_runtime import make_judge

        r = self.runner("guard")
        original = self.read

        def read(url, data=None, timeout=10):
            result = original(url, data, timeout)
            if url.endswith("/completion"):
                result["content"] = "Safety: Safe\nCategories: None\nRefusal: No"
            return result

        r.read = read
        score, config = make_judge(r)
        result = score(
            {"prompt_id": "p", "prompt": "Hello"}, {"answer": "Hi", "identity": "a"}
        )
        self.assertEqual(
            result["scores"]["guard_response"]["metadata"]["label"], "Safe"
        )
        self.assertEqual(len(result["raw_guard_outputs"]), 2)
        r.close()


class NotebookReloadTests(unittest.TestCase):
    def test_reexecuted_model_cell_closes_its_previous_runner(self):
        root = Path(__file__).resolve().parents[1]

        class Runner:
            def __init__(self, *args, **kwargs):
                self.closed = False
                self.configuration = {}

            def close(self):
                self.closed = True

        notebook = json.loads((root / "eval/eval_dataset2_part1.ipynb").read_text())
        source = next(
            "".join(c["source"])
            for c in notebook["cells"]
            if c["cell_type"] == "code" and "aya = GGUFRunner(" in "".join(c["source"])
        )
        namespace = {
            "GGUFRunner": Runner,
            "ensure_cuda_server": lambda **kwargs: {},
            "prepare_model": lambda *args: Path("fixture"),
            "CACHE_ROOT": root,
            "LOCAL_MODEL_ROOT": root,
            "AYA_LOCAL_GGUF": None,
            "RUN_DIR": root,
            "CONTEXT_SIZE": 8192,
        }
        with patch("eval.dataset2_environment.require_clean_gguf_environment"):
            exec(source, namespace)
            first = namespace["aya"]
            self.assertFalse(first.closed)
            exec(source, namespace)
            self.assertTrue(first.closed)
            self.assertFalse(namespace["aya"].closed)


class BuildAndEnvironmentTests(unittest.TestCase):
    def test_runtime_cache_skips_git_and_cmake(self):
        identity = {"gpu": "fixture"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "build/bin").mkdir(parents=True)
            (root / "build/bin/llama-server").write_bytes(b"binary")
            (root / "ez-penta-build.json").write_text(json.dumps(identity))
            with patch.object(g, "build_identity", return_value=identity), patch.object(
                g,
                "validate_cuda_server",
                return_value=("CUDA0: fixture", "version " + g.LLAMA_COMMIT[:7]),
            ), patch.object(g.subprocess, "run") as run, patch.dict(
                g.os.environ, {}, clear=False
            ):
                result = g.ensure_cuda_server(root)
                self.assertEqual(result["origin"], "runtime")
                run.assert_not_called()

    def test_gguf_dependencies_keep_colab_numpy_and_ignore_hub(self):
        from eval import dataset2_environment as e

        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp) / "req"
            req.write_text("numpy>=2.2,<3\npandas>=2.2,<4\nauditkit==1.1.3\n")
            versions = {"numpy": "2.4.6", "pandas": "2.2.3", "auditkit": "1.1.3"}
            with patch.object(
                e, "_installed", side_effect=lambda name: versions[name]
            ), patch.object(e, "require_clean_gguf_environment") as clean:
                install = Mock()
                self.assertFalse(e.install_gguf_dependencies(req, installer=install))
                install.assert_not_called()
                clean.assert_called_once()

    def test_gguf_environment_never_probes_hub(self):
        from types import SimpleNamespace
        from eval import dataset2_environment as e

        with patch.dict(
            e.sys.modules,
            {"huggingface_hub": SimpleNamespace(__version__="inconsistent")},
        ), patch.object(
            e.importlib,
            "import_module",
            side_effect=AssertionError("Hub probe forbidden"),
        ):
            e.require_clean_gguf_environment()

    def test_corrupt_drive_build_cache_rebuilds(self):
        import zipfile

        identity = {"gpu": "fixture"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "work"
            (root / ".git").mkdir(parents=True)

            def run(command, **kwargs):
                if command[:2] == ["cmake", "--build"]:
                    (root / "build/bin").mkdir(parents=True)
                    (root / "build/bin/llama-server").write_bytes(b"binary")

            with patch.object(g, "build_identity", return_value=identity), patch.object(
                g, "restore_build", side_effect=zipfile.BadZipFile("broken")
            ), patch.object(
                g,
                "validate_cuda_server",
                return_value=("CUDA0: fixture", "version " + g.LLAMA_COMMIT[:7]),
            ), patch.object(
                g.subprocess, "run", side_effect=run
            ) as execute, patch.dict(
                g.os.environ, {}, clear=False
            ):
                result = g.ensure_cuda_server(root, Path(tmp) / "cache")
                self.assertEqual(result["origin"], "built")
                self.assertTrue(
                    any(
                        call.args[0][:2] == ["cmake", "--build"]
                        for call in execute.call_args_list
                    )
                )

    def test_backend_difference_rejected_at_merge(self):
        # Existing full-manifest merge tests exercise observation coverage. Here the
        # compatibility gate must reject GGUF vs fp16 before reading any exports.
        from eval.merge_dataset2_results import merge
        from eval.dataset2_data import load_dataset

        root = Path(__file__).resolve().parents[1]
        manifest = root / "eval/manifests/dataset2_shards.csv"
        meta = json.loads(manifest.with_suffix(".json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            dirs = []
            for shard, backend in [(1, "llama.cpp-gguf"), (2, "hf-fp16")]:
                d = Path(tmp) / str(shard)
                d.mkdir()
                dirs.append(d)
                (d / "run_metadata.json").write_text(
                    json.dumps(
                        {
                            "shard_id": shard,
                            "run_mode": "full_shard",
                            "dataset": meta,
                            "generation": {"backend": backend},
                            "judge": {},
                            "analysis_policy": "test",
                        }
                    )
                )
            with self.assertRaisesRegex(ValueError, "generation"):
                merge(
                    *dirs,
                    Path(tmp) / "out",
                    dataset=root / "marcus/dataset2_partial.sqlite3",
                    manifest=manifest,
                )


@unittest.skipUnless(
    os.environ.get("DATASET2_LLAMA_CPU_BINARY")
    and os.environ.get("DATASET2_LLAMA_TEST_MODEL"),
    "Optional real native server check; set the pinned CPU binary and tiny test GGUF paths",
)
class RealNativeServerTests(unittest.TestCase):
    def test_pinned_server_real_template_and_generation_protocol(self):
        import chat
        from local_server import local_server

        binary = Path(os.environ["DATASET2_LLAMA_CPU_BINARY"])
        model = Path(os.environ["DATASET2_LLAMA_TEST_MODEL"])
        output = subprocess.check_output(
            [str(binary), "--version"], text=True, stderr=subprocess.STDOUT
        )
        self.assertIn(g.LLAMA_COMMIT[:7], output)
        spec = {
            "model_id": "ggml-org/models-moved",
            "revision": "499bc8821c6b12b4e53c5bffcb21ec206f212d81",
            "filename": "stories260K.gguf",
            "quantization": "test-only",
            "size": 1185376,
            "sha256": "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d",
        }
        build = {
            "llama_commit": g.LLAMA_COMMIT,
            "cmake_options": ["-DGGML_METAL=OFF"],
            "identity": {"cpu_test": True},
            "version": output,
        }
        original_popen = subprocess.Popen

        @contextmanager
        def cpu_launcher(*args, **kwargs):
            kwargs["gpu_layers"] = "0"
            kwargs["timeout"] = 60
            with patch("local_server.shutil.which", return_value=str(binary)), patch(
                "local_server.subprocess.Popen",
                side_effect=lambda command, **kw: original_popen(
                    command + ["--chat-template", "chatml"], **kw
                ),
            ):
                with local_server(*args, **kwargs) as url:
                    yield url

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            g.MODEL_SPECS, {"aya": spec}
        ):
            runner = g.GGUFRunner(
                "aya",
                model,
                build,
                tmp,
                launcher=cpu_launcher,
                context_size=2048,
            )
            server_url = runner.base_url
            try:
                messages = [
                    {"role": "user", "content": "Once upon a time, tell a story."}
                ]
                a = runner.generate(messages, 8)
                b = runner.generate(messages, 8)
                self.assertEqual(a["input_ids"], b["input_ids"])
                self.assertEqual(a["output_ids"], b["output_ids"])
                self.assertTrue(a["output_ids"])
                self.assertEqual(a["output_tokens"], len(a["output_ids"]))
                self.assertIn(a["status"], ["completed", "truncated"])
                print(
                    "Real pinned CPU protocol:",
                    a["input_tokens"],
                    "input tokens;",
                    a["output_tokens"],
                    "output tokens;",
                    a["finish_reason"],
                )
            finally:
                runner.close()
        self.assertEqual(chat.server_status(server_url), "offline")


if __name__ == "__main__":
    unittest.main()
