"""Behavioral tests for the local, browser-facing trace service.

Run with: python3 -m unittest discover -s tests -p 'test_live_service.py'
These use a small injected capture engine; the real model capture is validated
separately, so this suite does not download model weights or import torch.
"""

from __future__ import annotations

import http.client
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import uuid


SERVICE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "serve_traces.py"
SPEC = importlib.util.spec_from_file_location("moe_trace_service", SERVICE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TraceService = MODULE.TraceService
TraceHTTPServer = MODULE.TraceHTTPServer

ALLOWED_ORIGIN = "https://moe-trace-explainer.foamygenet33.chatgpt.site"


def wait_until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for trace service")


class FakeEngine:
    """Writes real temporary files while exposing concurrency and error behavior."""

    def __init__(self, block=False):
        self.release = threading.Event()
        if not block:
            self.release.set()
        self.started = threading.Event()
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def capture(self, prompt, trace_id, output_dir):
        with self.lock:
            self.calls.append((prompt, trace_id))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if not self.release.wait(5):
                raise RuntimeError("Test capture was not released")
            if prompt == "fail this capture":
                raise ValueError("Capture rejected the prompt")
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            for name in (
                "main.json",
                f"attention-operations-{trace_id}.json",
                f"expert-operations-{trace_id}.json",
            ):
                (output / name).write_text(json.dumps({"prompt": prompt, "id": trace_id}))
            return {
                "traceId": trace_id,
                "prompt": prompt,
                "tokenCount": 5,
                "files": {
                    "main": "main.json",
                    "attention": f"attention-operations-{trace_id}.json",
                    "expert": f"expert-operations-{trace_id}.json",
                },
            }
        finally:
            with self.lock:
                self.active -= 1


class ServiceCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.services = []
        self.engines = []
        self.initializers = []

    def tearDown(self):
        for event in self.initializers:
            event.set()
        for engine in self.engines:
            engine.release.set()
        for service in self.services:
            service.close()
        self.temp.cleanup()

    def new_service(self, engine=None, factory=None, **kwargs):
        engine = engine or FakeEngine()
        self.engines.append(engine)
        service = TraceService(
            engine_factory=factory or (lambda: engine),
            output_root=Path(self.temp.name),
            **kwargs,
        )
        self.services.append(service)
        return service, engine

    def ready_service(self, **kwargs):
        service, engine = self.new_service(**kwargs)
        service.start()
        wait_until(lambda: service.health()["status"] == "ready")
        return service, engine

    def terminal_job(self, service, job_id):
        def terminal():
            job = service.get_job(job_id)
            return job if job and job["status"] in {"complete", "failed"} else None

        return wait_until(terminal)


class TraceServiceTests(ServiceCase):
    def test_model_initializes_in_background_and_rejects_work_until_ready(self):
        entered = threading.Event()
        release = threading.Event()
        self.initializers.append(release)
        engine = FakeEngine()

        def factory():
            entered.set()
            release.wait(5)
            return engine

        service, _ = self.new_service(engine=engine, factory=factory)
        self.assertFalse(entered.is_set(), "Construction must not load model weights")
        service.start()
        self.assertTrue(entered.wait(1))
        self.assertEqual(service.health()["status"], "loading")
        with self.assertRaises(RuntimeError):
            service.submit("The capital of France is")
        release.set()
        wait_until(lambda: service.health()["status"] == "ready")
        job = service.submit("The capital of France is")
        self.assertEqual(self.terminal_job(service, job["id"])["status"], "complete")

    def test_failed_initialization_exposes_error_without_accepting_jobs(self):
        def fail_factory():
            raise RuntimeError("Model weights unavailable")

        service, _ = self.new_service(factory=fail_factory)
        service.start()
        wait_until(lambda: service.health()["status"] == "error")
        health = service.health()
        self.assertIn("Model weights unavailable", health["error"])
        with self.assertRaises(RuntimeError):
            service.submit("A new prompt")

    def test_capture_is_serial_and_pending_queue_is_bounded(self):
        service, engine = self.ready_service(engine=FakeEngine(block=True), max_pending=2)
        first = service.submit("First prompt")
        self.assertTrue(engine.started.wait(1))
        self.assertEqual(service.get_job(first["id"])["status"], "running")
        second = service.submit("Second prompt")
        self.assertEqual(service.get_job(second["id"])["status"], "queued")
        with self.assertRaises(RuntimeError):
            service.submit("Queue is full")
        self.assertEqual(len(engine.calls), 1)
        engine.release.set()
        self.assertEqual(self.terminal_job(service, first["id"])["status"], "complete")
        self.assertEqual(self.terminal_job(service, second["id"])["status"], "complete")
        self.assertEqual(engine.max_active, 1)
        self.assertEqual([prompt for prompt, _ in engine.calls], ["First prompt", "Second prompt"])
        self.assertNotEqual(first["id"], second["id"])
        self.assertTrue(first["id"].startswith("live-"))
        self.assertEqual(uuid.UUID(first["id"][5:]).hex, first["id"][5:])

    def test_failed_capture_does_not_stop_next_job(self):
        service, engine = self.ready_service()
        failed = service.submit("fail this capture")
        result = self.terminal_job(service, failed["id"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("Capture rejected the prompt", result["error"])
        good = service.submit("A different prompt")
        self.assertEqual(self.terminal_job(service, good["id"])["status"], "complete")
        self.assertEqual(len(engine.calls), 2)

    def test_completed_retention_is_bounded_without_deleting_unowned_files(self):
        service, _ = self.ready_service(max_jobs=2)
        first = service.submit("First retained prompt")
        self.terminal_job(service, first["id"])
        first_folder = Path(self.temp.name) / first["id"]
        unowned = first_folder / "leave-me.txt"
        unowned.write_text("This file is not owned by the capture service")
        second = service.submit("Second retained prompt")
        self.terminal_job(service, second["id"])
        third = service.submit("Third retained prompt")
        self.terminal_job(service, third["id"])
        self.assertIsNone(service.get_job(first["id"]))
        self.assertIsNone(service.trace_file(first["id"], "main.json"))
        self.assertFalse((first_folder / "main.json").exists())
        self.assertTrue(unowned.is_file())
        self.assertEqual(service.get_job(second["id"])["status"], "complete")
        self.assertEqual(service.get_job(third["id"])["status"], "complete")

    def test_partial_capture_is_failed_and_its_file_is_not_published(self):
        class PartialEngine(FakeEngine):
            def capture(self, prompt, trace_id, output_dir):
                Path(output_dir, "main.json").write_text('{"incomplete": true}')
                return {"tokenCount": 2}

        service, _ = self.ready_service(engine=PartialEngine())
        job = service.submit("A partially recorded prompt")
        self.assertEqual(self.terminal_job(service, job["id"])["status"], "failed")
        self.assertIsNone(service.trace_file(job["id"], "main.json"))

    def test_invalid_prompts_never_enter_capture_queue(self):
        service, engine = self.ready_service()
        for prompt in (None, 123, [], "", " \n\t ", "x" * 100_000, "\ud800"):
            with self.subTest(prompt_type=type(prompt).__name__, length=len(str(prompt))):
                with self.assertRaises(ValueError):
                    service.submit(prompt)
        self.assertEqual(engine.calls, [])


class TraceHTTPTests(ServiceCase):
    def setUp(self):
        super().setUp()
        self.service, self.engine = self.ready_service()
        self.web_root = Path(self.temp.name) / "web"
        self.web_root.mkdir()
        (self.web_root / "index.html").write_text('<!doctype html><title>Local explainer</title>')
        (self.web_root / "dense").mkdir()
        (self.web_root / "dense" / "index.html").write_text('<!doctype html><title>Dense explainer</title>')
        (self.web_root / "_app").mkdir()
        (self.web_root / "_app" / "app.js").write_text('window.app = "MoE";')
        (self.web_root / "_app" / "app.css").write_text('body { color: purple; }')
        (self.web_root / "traces").mkdir()
        (self.web_root / "traces" / "granite-moe.json").write_text('{"recorded": true}')
        self.server = TraceHTTPServer(
            ("127.0.0.1", 0), self.service, allowed_origins={ALLOWED_ORIGIN},
            web_root=self.web_root,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)
        super().tearDown()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def post(self, payload, **header_overrides):
        headers = {
            "Origin": ALLOWED_ORIGIN,
            "X-MoE-Client": "explainer",
            "Content-Type": "application/json",
        }
        headers.update(header_overrides)
        headers = {key: value for key, value in headers.items() if value is not None}
        return self.request("POST", "/api/traces", json.dumps(payload), headers)

    def complete_job(self):
        status, _, data = self.post({"prompt": "An arbitrary custom prompt"})
        self.assertIn(status, {200, 202})
        job = json.loads(data)
        self.terminal_job(self.service, job["id"])
        return job

    def test_local_app_serves_root_assets_and_bundled_traces_with_head(self):
        for path, filename, content_type in (
            ("/", "index.html", "text/html"),
            ("/_app/app.js?v=1", "_app/app.js", "text/javascript"),
            ("/_app/app.css", "_app/app.css", "text/css"),
            ("/traces/granite-moe.json", "traces/granite-moe.json", "application/json"),
        ):
            with self.subTest(path=path):
                expected = (self.web_root / filename).read_bytes()
                status, headers, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertEqual(body, expected)
                self.assertTrue(headers["Content-Type"].startswith(content_type))
                self.assertEqual(int(headers["Content-Length"]), len(expected))
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
                self.assertEqual(headers["Cache-Control"], "no-store")
                head_status, head_headers, head_body = self.request("HEAD", path)
                self.assertEqual(head_status, 200)
                self.assertEqual(head_body, b"")
                self.assertEqual(head_headers["Content-Length"], headers["Content-Length"])
                self.assertEqual(head_headers["Content-Type"], headers["Content-Type"])

    def test_large_static_file_arrives_completely(self):
        expected = bytes(range(256)) * 3000
        (self.web_root / "asset.bin").write_bytes(expected)
        status, headers, body = self.request("GET", "/asset.bin")
        self.assertEqual(status, 200)
        self.assertEqual(int(headers["Content-Length"]), len(expected))
        self.assertEqual(body, expected)

    def test_dense_entry_serves_its_index_without_general_directory_fallback(self):
        expected = (self.web_root / "dense" / "index.html").read_bytes()
        for path in ("/dense/", "/dense/?example=1"):
            with self.subTest(path=path):
                status, headers, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertEqual(body, expected)
                self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
                self.assertEqual(int(headers["Content-Length"]), len(expected))
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
                head_status, head_headers, head_body = self.request("HEAD", path)
                self.assertEqual(head_status, 200)
                self.assertEqual(head_headers["Content-Length"], headers["Content-Length"])
                self.assertEqual(head_body, b"")
        (self.web_root / "_app" / "index.html").write_text("UNREQUESTED INDEX")
        for path in ("/dense", "/dense//", "/dense/missing", "/dense/missing/", "/_app/"):
            with self.subTest(path=path):
                status, _, body = self.request("GET", path)
                self.assertEqual(status, 404)
                self.assertNotIn(expected, body)
                self.assertNotIn(b"UNREQUESTED INDEX", body)

    def test_dense_index_still_rejects_traversal_and_external_symlinks(self):
        for path in ("/dense/../index.html", "/dense/%2e%2e/index.html"):
            with self.subTest(path=path):
                status, _, _ = self.request("GET", path)
                self.assertEqual(status, 404)
        outside = Path(self.temp.name) / "private.html"
        outside.write_text("PRIVATE CONTENT")
        index = self.web_root / "dense" / "index.html"
        index.unlink()
        index.symlink_to(outside)
        for method in ("GET", "HEAD"):
            with self.subTest(method=method):
                status, _, body = self.request(method, "/dense/")
                self.assertEqual(status, 404)
                self.assertNotIn(b"PRIVATE CONTENT", body)

    def test_static_paths_cannot_escape_build_root_or_list_directories(self):
        outside = Path(self.temp.name) / "secret.txt"
        outside.write_text("PRIVATE CONTENT")
        (self.web_root / "escape.txt").symlink_to(outside)
        (self.web_root / "outside").symlink_to(Path(self.temp.name), target_is_directory=True)
        (self.web_root / ".env").write_text("PRIVATE CONTENT")
        for path in (
            "/../secret.txt", "/%2e%2e/secret.txt", "/_app/../../secret.txt",
            "/%2e%2e%2fsecret.txt", "/%5c..%5csecret.txt", "/%00", "/%ff",
            "/escape.txt", "/outside/secret.txt", "/.env", "/_app", "/_app/", "/missing.js",
        ):
            with self.subTest(path=path):
                status, _, data = self.request("GET", path)
                self.assertEqual(status, 404)
                self.assertNotIn(b"PRIVATE CONTENT", data)
                head_status, _, head_body = self.request("HEAD", path)
                self.assertEqual(head_status, 404)
                self.assertEqual(head_body, b"")

    def test_static_files_cannot_shadow_health_api_or_live_trace_routes(self):
        (self.web_root / "health").write_text("STATIC SHADOW")
        unknown_id = "live-" + uuid.uuid4().hex
        for name in ("api/unknown", "api/traces/not-a-uuid", f"traces/{unknown_id}/main.json"):
            file = self.web_root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("STATIC SHADOW")
            status, _, data = self.request("GET", "/" + name)
            self.assertEqual(status, 404)
            self.assertNotIn(b"STATIC SHADOW", data)
        status, _, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ready")
        status, _, body = self.request("HEAD", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"")

    def test_local_app_origin_can_generate_on_its_actual_ephemeral_port(self):
        status, headers, body = self.post(
            {"prompt": "Typed into the local app"}, Origin=f"http://127.0.0.1:{self.port}"
        )
        self.assertEqual(status, 202)
        self.assertEqual(headers["Access-Control-Allow-Origin"], f"http://127.0.0.1:{self.port}")
        job = json.loads(body)
        self.assertEqual(self.terminal_job(self.service, job["id"])["status"], "complete")

    def test_health_and_cors_accept_only_exact_allowed_origin(self):
        status, headers, data = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["status"], "ready")
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        status, headers, _ = self.request("GET", "/health", headers={"Origin": ALLOWED_ORIGIN})
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), ALLOWED_ORIGIN)
        for origin in ("https://unrelated.example", ALLOWED_ORIGIN + ".evil.example", "null"):
            with self.subTest(origin=origin):
                status, headers, _ = self.request("GET", "/health", headers={"Origin": origin})
                self.assertEqual(status, 403)
                self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_host_validation_blocks_dns_rebinding(self):
        for host in ("evil.example", "localhost.evil.example", "127.0.0.1.evil.example", "192.168.1.2"):
            with self.subTest(host=host):
                status, _, _ = self.request("GET", "/health", headers={"Host": host})
                self.assertEqual(status, 403)
        for host in (f"127.0.0.1:{self.port}", f"localhost:{self.port}"):
            with self.subTest(host=host):
                status, _, _ = self.request("GET", "/health", headers={"Host": host})
                self.assertEqual(status, 200)

    def test_host_cannot_smuggle_a_url_suffix(self):
        for suffix in ("/other", "?query=1", "#fragment"):
            with self.subTest(suffix=suffix):
                status, _, _ = self.request(
                    "GET", "/health", headers={"Host": f"127.0.0.1:{self.port}{suffix}"}
                )
                self.assertEqual(status, 403)

    def test_post_requires_trusted_origin_client_header_and_json(self):
        for overrides in (
            {"Origin": None},
            {"Origin": "https://unrelated.example"},
            {"Origin": ALLOWED_ORIGIN + ".evil.example"},
            {"X-MoE-Client": None},
            {"X-MoE-Client": "other-client"},
            {"Content-Type": "text/plain"},
        ):
            with self.subTest(overrides=overrides):
                status, _, data = self.post({"prompt": "Do not run"}, **overrides)
                self.assertIn(status, {400, 403, 415})
                self.assertIn("error", json.loads(data))
        self.assertEqual(self.engine.calls, [])

    def test_malformed_payloads_are_rejected_before_capture(self):
        for payload in ({}, {"prompt": ""}, {"prompt": "   "}, {"prompt": 42}, [], None):
            with self.subTest(payload=payload):
                status, _, data = self.post(payload)
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(data))
        status, _, _ = self.request(
            "POST", "/api/traces", "{broken json",
            {"Origin": ALLOWED_ORIGIN, "X-MoE-Client": "explainer", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)
        self.assertEqual(self.engine.calls, [])

    def test_preflight_only_grants_the_configured_origin(self):
        headers = {
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-moe-client",
        }
        status, response_headers, _ = self.request("OPTIONS", "/api/traces", headers=headers)
        self.assertIn(status, {200, 204})
        self.assertEqual(response_headers.get("Access-Control-Allow-Origin"), ALLOWED_ORIGIN)
        self.assertIn("POST", response_headers.get("Access-Control-Allow-Methods", ""))
        self.assertIn("x-moe-client", response_headers.get("Access-Control-Allow-Headers", "").lower())
        headers["Origin"] = "https://unrelated.example"
        status, response_headers, _ = self.request("OPTIONS", "/api/traces", headers=headers)
        self.assertEqual(status, 403)
        self.assertNotIn("Access-Control-Allow-Origin", response_headers)

    def test_trace_file_symlink_is_not_served(self):
        job = self.complete_job()
        trace_path = Path(self.temp.name) / job["id"] / "main.json"
        private_path = Path(self.temp.name) / "private.json"
        private_path.write_text('{"secret": true}')
        trace_path.unlink()
        trace_path.symlink_to(private_path)
        status, _, data = self.request("GET", f"/traces/{job['id']}/main.json")
        self.assertEqual(status, 404)
        self.assertNotIn(b'"secret": true', data)

    def test_job_status_and_only_its_known_trace_files_are_readable(self):
        job = self.complete_job()
        job_id = job["id"]
        status, _, data = self.request("GET", f"/api/traces/{job_id}")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["status"], "complete")
        for name in ("main.json", f"attention-operations-{job_id}.json", f"expert-operations-{job_id}.json"):
            with self.subTest(name=name):
                status, headers, data = self.request("GET", f"/traces/{job_id}/{name}")
                self.assertEqual(status, 200)
                self.assertTrue(headers.get("Content-Type", "").startswith("application/json"))
                self.assertIn(f'filename="{job_id}-{name}"', headers.get("Content-Disposition", ""))
                self.assertTrue(headers["Content-Disposition"].startswith("attachment;"))
                self.assertEqual(json.loads(data)["prompt"], "An arbitrary custom prompt")
                head_status, head_headers, head_body = self.request("HEAD", f"/traces/{job_id}/{name}")
                self.assertEqual(head_status, 200)
                self.assertEqual(head_headers["Content-Length"], headers["Content-Length"])
                self.assertEqual(head_body, b"")
        # Place data nearby to prove the route cannot serve arbitrary filenames,
        # even if files exist inside a successful capture directory.
        (Path(self.temp.name) / job_id / "private.json").write_text('{"secret": true}')
        (Path(self.temp.name) / "secret.json").write_text('{"secret": true}')
        for path in (
            f"/traces/{job_id}/private.json",
            f"/traces/{job_id}/../secret.json",
            f"/traces/{job_id}/%2e%2e/secret.json",
            f"/traces/{job_id}/%2e%2e%2fsecret.json",
            "/traces/../secret.json",
            f"/traces/{uuid.uuid4()}/main.json",
            "/api/traces/not-a-uuid",
        ):
            with self.subTest(path=path):
                status, _, data = self.request("GET", path)
                self.assertIn(status, {400, 404})
                self.assertNotIn(b'"secret": true', data)


if __name__ == "__main__":
    unittest.main()
