#!/usr/bin/env python3
"""Loopback-only Granite inference service for the MoE explainer.

Run with the trace Python environment, then open the local app or hosted page.
The model stays loaded; requests execute serially. Only explicitly allowed
browser origins may submit prompts. No model or remote Python code is downloaded.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import queue
import re
import signal
import threading
import time
from urllib.parse import unquote, urlsplit
import uuid

MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-base"
MODEL_REVISION = "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
SITE_ORIGIN = "https://moe-trace-explainer.foamygenet33.chatgpt.site"
TRACE_ID = re.compile(r"live-[0-9a-f]{32}\Z")
MAX_TOKENS = 32
MAX_CHARS = 4096


class TraceService:
    def __init__(self, engine_factory, output_root, max_jobs=12, max_pending=2):
        self.engine_factory = engine_factory
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.max_jobs = max_jobs
        self.max_pending = max_pending
        self._jobs = OrderedDict()
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        self._state = "loading"
        self._error = ""
        self._thread = None
        self._closed = False

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="granite-inference")
            self._thread.start()

    def close(self):
        with self._lock:
            self._closed = True
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=180)

    def health(self):
        with self._lock:
            return {"status": self._state, "modelId": MODEL_ID, "revision": MODEL_REVISION,
                    "maxTokens": MAX_TOKENS, "maxCharacters": MAX_CHARS,
                    **({"error": self._error} if self._error else {})}

    def submit(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter a prompt containing text.")
        if len(prompt) > MAX_CHARS or len(prompt.encode("utf-8")) > 8192:
            raise ValueError("Keep the prompt within 4,096 characters and 8 KB of UTF-8 text.")
        with self._lock:
            if self._closed or self._state != "ready":
                raise RuntimeError(self._error or "The local model is still loading. Try again shortly.")
            pending = sum(job["status"] in ("queued", "running") for job in self._jobs.values())
            if pending >= self.max_pending:
                raise RuntimeError("The model is busy. Wait for the current trace and try again.")
            while len(self._jobs) >= self.max_jobs:
                oldest = next((key for key, job in self._jobs.items()
                               if job["status"] in ("complete", "failed")), None)
                if oldest is None:
                    raise RuntimeError("The trace queue is full. Try again shortly.")
                self._jobs.pop(oldest)
                # Only the three files owned by this process are removed.
                folder = self.output_root / oldest
                for name in self._filenames(oldest):
                    (folder / name).unlink(missing_ok=True)
                try:
                    folder.rmdir()
                except OSError:
                    pass
            trace_id = "live-" + uuid.uuid4().hex
            job = {"id": trace_id, "status": "queued", "statusUrl": f"/api/traces/{trace_id}"}
            self._jobs[trace_id] = job
            self._queue.put((trace_id, prompt))
            return dict(job)

    @staticmethod
    def _filenames(trace_id):
        return ("main.json", f"attention-operations-{trace_id}.json",
                f"expert-operations-{trace_id}.json")

    def get_job(self, trace_id):
        with self._lock:
            job = self._jobs.get(trace_id)
            return dict(job) if job else None

    def trace_file(self, trace_id, name):
        if not TRACE_ID.fullmatch(trace_id) or name not in self._filenames(trace_id):
            return None
        with self._lock:
            job = self._jobs.get(trace_id)
            if not job or job["status"] != "complete":
                return None
        path = self.output_root / trace_id / name
        if path.is_symlink() or not path.is_file():
            return None
        return path

    def _run(self):
        try:
            engine = self.engine_factory()
            with self._lock:
                self._state = "ready"
            print("Granite ready for real prompt traces.", flush=True)
        except Exception as error:
            with self._lock:
                self._state = "error"
                self._error = f"Could not load the local model: {error}"
            print(self._error, flush=True)
            return
        while True:
            item = self._queue.get()
            if item is None:
                return
            trace_id, prompt = item
            with self._lock:
                self._jobs[trace_id]["status"] = "running"
            try:
                folder = self.output_root / trace_id
                folder.mkdir(mode=0o700)
                started = time.monotonic()
                metadata = engine.capture(prompt, trace_id, folder)
                if not all((folder / name).is_file() for name in self._filenames(trace_id)):
                    raise RuntimeError("Inference did not produce all required trace files.")
                with self._lock:
                    self._jobs[trace_id].update({
                        "status": "complete", "seconds": round(time.monotonic() - started, 3),
                        "tokenCount": metadata.get("tokenCount"),
                        "mainUrl": f"/traces/{trace_id}/main.json",
                        "attentionUrl": f"/traces/{trace_id}/attention-operations-{trace_id}.json",
                        "expertUrl": f"/traces/{trace_id}/expert-operations-{trace_id}.json",
                    })
            except Exception as error:
                with self._lock:
                    self._jobs[trace_id].update({"status": "failed", "error": str(error)})
                print(f"Trace {trace_id} failed: {error}", flush=True)


class TraceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, service, allowed_origins=None, web_root=None):
        self.service = service
        self.allowed_origins = set(allowed_origins or {SITE_ORIGIN})
        self.web_root = Path(web_root).resolve() if web_root is not None else None
        if self.web_root is not None and not self.web_root.is_dir():
            raise ValueError("The local app build directory does not exist.")
        super().__init__(address, TraceHandler)
        self.allowed_origins.update({
            f"http://127.0.0.1:{self.server_port}",
            f"http://localhost:{self.server_port}",
            f"http://[::1]:{self.server_port}",
        })

    def static_file(self, path):
        if self.web_root is None:
            return None
        # Only the architecture entry points have directory indexes. Every other
        # static route must name a file explicitly, including dense assets.
        if path == "/":
            parts = ["index.html"]
        elif path == "/dense/":
            parts = ["dense", "index.html"]
        elif path in {f"/architectures/{name}/" for name in ("mla", "encoder-decoder", "hybrid", "vision-language", "vit")}:
            parts = ["architectures", path.split("/")[2], "index.html"]
        else:
            parts = path.removeprefix("/").split("/")
        if any(not part or part.startswith(".") or "\\" in part or "\0" in part for part in parts):
            return None
        try:
            candidate = self.web_root.joinpath(*parts).resolve(strict=True)
            if not candidate.is_relative_to(self.web_root) or not candidate.is_file():
                return None
        except (OSError, RuntimeError, ValueError):
            return None
        return candidate


class TraceHandler(BaseHTTPRequestHandler):
    server_version = "MoETrace/1"

    def log_message(self, fmt, *args):
        # Never log the submitted prompt or response tensors.
        super().log_message(fmt, *args)

    def _allowed(self, require_origin=False):
        try:
            host = urlsplit("http://" + self.headers.get("Host", ""))
            valid_host = (host.hostname in {"127.0.0.1", "localhost", "::1"}
                          and host.port == self.server.server_port
                          and not host.username and not host.password
                          and not host.path and not host.query and not host.fragment)
        except ValueError:
            valid_host = False
        origin = self.headers.get("Origin")
        if not valid_host or (require_origin and not origin) or (
                origin and origin not in self.server.allowed_origins):
            self._json(403, {"error": "This request is not from an allowed explainer origin."})
            return False
        return True

    def _headers(self, status, content_type, length=0, filename=None):
        self.send_response(status)
        origin = self.headers.get("Origin")
        if origin in self.server.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-MoE-Client")
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()

    def _json(self, status, data):
        body = json.dumps(data, allow_nan=False).encode()
        self._headers(status, "application/json; charset=utf-8", len(body))
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_OPTIONS(self):
        if not self._allowed(require_origin=True):
            return
        method = self.headers.get("Access-Control-Request-Method", "")
        headers = {x.strip().lower() for x in self.headers.get("Access-Control-Request-Headers", "").split(",") if x.strip()}
        if method not in {"GET", "HEAD", "POST"} or headers - {"content-type", "x-moe-client"}:
            self._json(403, {"error": "Unsupported cross-origin request."})
            return
        self._headers(204, "application/json")

    def _file(self, file, content_type, filename=None):
        try:
            handle = file.open("rb")
        except OSError:
            self._json(404, {"error": "Resource is no longer available."})
            return
        with handle:
            self._headers(200, content_type, os.fstat(handle.fileno()).st_size, filename=filename)
            if self.command == "HEAD":
                return
            try:
                while block := handle.read(256 * 1024):
                    self.wfile.write(block)
            except (BrokenPipeError, ConnectionResetError):
                # A closed tab need not produce a server-side traceback.
                return

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._allowed():
            return
        try:
            target = urlsplit(self.path)
            path = unquote(target.path, errors="strict")
            if (target.scheme or target.netloc or target.fragment or not path.startswith("/")
                    or "\\" in path or "\0" in path
                    or any(part in {".", ".."} for part in path.split("/"))):
                raise ValueError("Invalid request path")
        except (ValueError, UnicodeError):
            self._json(404, {"error": "Resource not found."})
            return
        if path == "/health":
            self._json(200, self.server.service.health())
            return
        match = re.fullmatch(r"/api/traces/(live-[0-9a-f]{32})", path)
        if match:
            job = self.server.service.get_job(match[1])
            self._json(200 if job else 404, job or {"error": "Trace not found or no longer retained."})
            return
        match = re.fullmatch(r"/traces/(live-[0-9a-f]{32})/([^/]+)", path)
        if match:
            file = self.server.service.trace_file(match[1], match[2])
            if file:
                self._file(file, "application/json; charset=utf-8", filename=f"{match[1]}-{match[2]}")
            else:
                self._json(404, {"error": "Trace resource not found."})
            return
        # API and live-recording routes never fall through to static files. Only
        # single-file bundled recordings are served beneath the /traces prefix.
        if (path == "/api" or path.startswith("/api/") or path.startswith("/health/")
                or (path == "/traces" or path.startswith("/traces/"))
                and not re.fullmatch(r"/traces/(?!live-)[^/]+\.json", path)):
            self._json(404, {"error": "Trace resource not found."})
            return
        file = self.server.static_file(path)
        if file:
            content_type = {
                ".html": "text/html; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".mjs": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".svg": "image/svg+xml",
                ".wasm": "application/wasm",
            }.get(file.suffix.lower()) or mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            self._file(file, content_type)
            return
        self._json(404, {"error": "Resource not found."})

    def do_POST(self):
        if not self._allowed(require_origin=True):
            return
        if self.path != "/api/traces":
            self._json(404, {"error": "Unknown endpoint."})
            return
        if (self.headers.get("X-MoE-Client") != "explainer"
                or self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json"):
            self._json(415, {"error": "Send JSON from the explainer prompt bar."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError("Prompt request must be between 1 byte and 16 KB.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict) or set(data) != {"prompt"}:
                raise ValueError("The request must contain only a prompt.")
            job = self.server.service.submit(data["prompt"])
        except (ValueError, UnicodeError) as error:
            self._json(400, {"error": str(error)})
            return
        except RuntimeError as error:
            self._json(409, {"error": str(error)})
            return
        self._json(202, job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--cache-dir", default="/private/tmp/moe-hf-cache")
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/moe-live-traces"))
    parser.add_argument("--threads", type=int, default=4)
    default_build = Path(__file__).resolve().parents[1] / "build"
    parser.add_argument("--web-root", type=Path,
                        default=default_build if default_build.is_dir() else None,
                        help="Built app directory to serve (defaults to this project's build directory when present)")
    parser.add_argument("--allow-origin", action="append", default=[])
    args = parser.parse_args()

    def factory():
        from live_capture import LiveTraceEngine
        return LiveTraceEngine(cache_dir=args.cache_dir, threads=args.threads)

    origins = {SITE_ORIGIN, "http://127.0.0.1:4178", "http://127.0.0.1:5178", *args.allow_origin}
    service = TraceService(factory, args.output_dir)
    server = TraceHTTPServer(("127.0.0.1", args.port), service, allowed_origins=origins,
                             web_root=args.web_root)
    service.start()

    def stop(_signal, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"MoE inference listening at http://127.0.0.1:{server.server_port}; loading pinned local Granite.", flush=True)
    if server.web_root:
        print(f"Local explainer: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
