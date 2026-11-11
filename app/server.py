"""Tiny local server: serves the generated page and powers its "Update" button.

  GET  /              the page            GET /api/status   {running, ok, message, log[], built_at}
  POST /api/refresh   starts the update flow (same command you'd run by hand) in the background

Security: binds to 127.0.0.1 only, POST needs a custom header (blocks cross-site requests) and a
local Host header (blocks DNS rebinding). Nothing outside the output folder is served.
"""
import http.server, json, os, subprocess, sys, threading, time, webbrowser
from functools import partial


class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False; self.ok = None; self.message = ""; self.log = []; self.finished_at = None


def serve(root, html_path, refresh_cmd, port, open_browser):
    job = Job()
    out_dir, page = os.path.dirname(html_path), os.path.basename(html_path)
    state_file = os.path.join(root, "data", "state.json")

    def built_at():
        try: return json.load(open(state_file, encoding="utf-8")).get("built_at") or json.load(open(state_file, encoding="utf-8")).get("checked_at")
        except Exception: return None

    def run():
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        try:
            p = subprocess.Popen(refresh_cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", env=env)
            for line in p.stdout:
                line = line.rstrip()
                if line: job.log.append(line)
            p.wait()
            lines = [l for l in job.log if l and not l.startswith("Page:")]
            job.ok = p.returncode == 0
            job.message = lines[-1] if lines else ("Done" if job.ok else "Failed")
        except Exception as e:                       # noqa: BLE001
            job.ok = False; job.message = str(e)
        finally:
            job.running = False; job.finished_at = time.time()

    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a): pass
        def end_headers(self):
            self.send_header("Cache-Control", "no-store"); super().end_headers()
        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        def _local(self):
            host = (self.headers.get("Host") or "").split(":")[0]
            return host in ("127.0.0.1", "localhost")
        def do_GET(self):
            if not self._local(): return self._json(403, {"error": "forbidden"})
            if self.path.split("?")[0] == "/api/status":
                return self._json(200, {"running": job.running, "ok": job.ok, "message": job.message,
                                        "log": job.log[-12:], "built_at": built_at()})
            if self.path in ("/", ""): self.path = "/" + page
            return super().do_GET()
        def do_POST(self):
            if not self._local() or self.headers.get("X-Requested-With") != "update-button":
                return self._json(403, {"error": "forbidden"})
            if self.path.split("?")[0] != "/api/refresh": return self._json(404, {"error": "not found"})
            with job.lock:
                if job.running: return self._json(409, {"error": "already running"})
                job.running = True; job.ok = None; job.message = "Starting..."; job.log = []
            threading.Thread(target=run, daemon=True).start()
            return self._json(202, {"started": True})

    handler = partial(H, directory=out_dir)
    srv = None
    for p in range(port, port + 10):
        try:
            srv = http.server.ThreadingHTTPServer(("127.0.0.1", p), handler); port = p; break
        except OSError:
            continue
    if srv is None:
        sys.exit(f"Could not bind a local port near {port}.")
    url = f"http://127.0.0.1:{port}/"
    print(f"Serving {page} at {url}  (Ctrl+C or close this window to stop)", flush=True)
    if open_browser: webbrowser.open(url)
    try: srv.serve_forever()
    except KeyboardInterrupt: print("Stopped.")
