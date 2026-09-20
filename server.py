"""chunklaya as an HTTP service, for running inside a Tinfoil Container.

The Tinfoil shim terminates the attested connection and forwards to
`upstream-port` over loopback inside the measured CVM, so this is a plain HTTP
server. Attestation is the platform's job here, not this process's: the
enclave measurement covers the image digest and the dm-verity-protected root
filesystem, and clients verify it with `@tinfoilsh/verifier` before any
plaintext is sent.

That is the whole reason this file is simpler than the Nitro Enclave version
it replaces. There, the enclave had no network stack, so the service spoke
vsock with a length-prefixed frame protocol and a separate proxy on the parent
translated HTTP into it -- and the parent saw every request in the clear.

    POST /ask       {"state", "questions", "detectors"?, "agg"?, "max_len"?, "prefilter"?, "top_k"?, ...}
    GET  /health

`detectors` arrives as [question, option_key] because JSON has no tuples.
Errors are returned as JSON rather than raised, so one bad request cannot take
the container down.
"""
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# transformers probes for TensorFlow at import and its abseil runtime can
# deadlock model construction; laya is torch-only. Offline is deliberate: the
# weights are baked into the image, and a Hub call would hang until it timed
# out rather than failing outright.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

import laya  # noqa: E402
from chunklaya import ChunkLaya  # noqa: E402

MODEL_DIR = os.environ.get("LAYA_MODEL_DIR", "/opt/models/laya")
PORT = int(os.environ.get("PORT", "8080"))
MAX_BODY = 64 * 1024 * 1024
# Set only to override torch's own choice. It sizes its pool from the physical
# core count, which is already what this container owns.
THREADS = int(os.environ.get("LAYA_THREADS", "0"))
# Optional. When set, /ask requires `Authorization: Bearer <it>`. Tinfoil
# injects it as an enclave secret; it is never in the image or this repo.
API_KEY = os.environ.get("API_KEY")

AGENT = None


def log(msg):
    print("[chunklaya] %s" % msg, flush=True)


def answer(req):
    """One request against the shared agent.

    A ChunkLaya is built per call because the chunking knobs are per-request;
    the agent, which holds the weights, is shared and never rebuilt.
    """
    cj = ChunkLaya(
        AGENT,
        chunk_tokens=int(req.get("chunk_tokens", 750)),
        stride=req.get("stride"),
        batch_size=int(req.get("batch_size", 16)),
        mode=req.get("mode", "paragraphs"),
        gate=req.get("gate", "auto"),
        prefilter=req.get("prefilter"),          # "bm25" to locate first; Laya then scores top_k chunks
        top_k=int(req.get("top_k", 1)),
        cache=None,  # the container filesystem is ephemeral and read-only
    )

    detectors = None
    if req.get("detectors"):
        detectors = {k: (v[0], v[1]) for k, v in req["detectors"].items()}

    return cj.ask(
        req["state"],
        req["questions"],
        agg=req.get("agg"),
        max_len=req.get("max_len"),
        detectors=detectors,
    )


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _reply(self, code, obj, close=False):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if close:
            self.close_connection = True
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self):
        if not API_KEY:
            return True
        return self.headers.get("Authorization") == "Bearer " + API_KEY

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/health"):
            if STATE["ready"]:
                self._reply(200, {"status": "ok"})
            else:
                self._reply(503, {"status": STATE["phase"], "error": STATE["error"]})
        else:
            self._reply(404, {"error": "GET /health or POST /ask"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            # Draining megabytes only to discard them is worse than ending the
            # connection, so say so and close.
            self._reply(413, {"error": "body over %d bytes" % MAX_BODY}, close=True)
            return

        # Read the body before any early return. This is a keep-alive server
        # behind a shim that pools its upstream connection, so a reply that
        # leaves unread bytes in the socket makes the next request on that
        # connection start mid-JSON.
        raw = self.rfile.read(length) if length else b""

        if self.path.rstrip("/") not in ("", "/ask"):
            self._reply(404, {"error": "GET /health or POST /ask"})
            return
        if not STATE["ready"]:
            self._reply(503, {"error": "model not ready", "status": STATE["phase"]})
            return
        if not self._authorised():
            self._reply(401, {"error": "bad or missing bearer token"})
            return
        if not raw:
            self._reply(400, {"error": "empty body"})
            return

        try:
            req = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._reply(400, {"error": "invalid JSON: %s" % exc})
            return

        try:
            self._reply(200, answer(req))
        except (KeyError, TypeError, ValueError) as exc:
            self._reply(400, {"error": "%s: %s" % (type(exc).__name__, exc)})
        except Exception as exc:  # one bad request must not kill the server
            traceback.print_exc()
            self._reply(500, {"error": "%s: %s" % (type(exc).__name__, exc)})

    def log_message(self, fmt, *args):
        # Never log paths or bodies: a `state` is customer text, and the whole
        # point of the enclave is that it does not leave.
        return


STATE = {"phase": "starting", "error": None, "ready": False}


def _serve():
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


def main():
    global AGENT

    # Bind before loading. A model load that hangs rather than crashes would
    # otherwise leave nothing on the port: the healthcheck fails with curl's
    # exit 7, the process never exits so `restart` never fires, and the
    # deployment sits in `healthy: pending` forever with no way to see why.
    # Serving /health from the first second turns that into a readable phase.
    threading.Thread(target=_serve, daemon=True).start()
    log("listening on :%d" % PORT)

    try:
        if THREADS > 0:
            torch.set_num_threads(THREADS)
        STATE["phase"] = "torch"
        log("torch %s, %d threads" % (torch.__version__, torch.get_num_threads()))

        if not os.path.isdir(MODEL_DIR):
            raise RuntimeError("no model at %s -- it must be baked into the image" % MODEL_DIR)

        STATE["phase"] = "loading"
        log("loading %s" % MODEL_DIR)
        AGENT = laya.load(MODEL_DIR, device="cpu")
        log("loaded on %s, max_len=%s" % (AGENT.device, AGENT.cfg.get("max_len")))

        # A health probe should not pass until the model has actually run once.
        STATE["phase"] = "warmup"
        warm = AGENT.predict("ready", {"q": {"type": "noul", "instructions": "Is this English?"}})
        log("warmup ok, noul=%.3f" % warm["answers"]["q"]["noul"])
    except BaseException as exc:
        log("FATAL during %s" % STATE["phase"])
        STATE["error"] = "%s: %s" % (type(exc).__name__, exc)
        STATE["phase"] = "failed"
        traceback.print_exc()
        # Stay up. The traceback is unreachable from outside a debug enclave,
        # so /health reporting the error is the only way to see it.
        threading.Event().wait()

    STATE["phase"] = "ready"
    STATE["ready"] = True
    log("auth: %s" % ("bearer token required" if API_KEY else "OPEN (no API_KEY set)"))
    threading.Event().wait()


if __name__ == "__main__":
    sys.exit(main() or 0)
