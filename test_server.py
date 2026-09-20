"""Drive the real Handler with real HTTP, stubbing only the model imports."""
import json, os, sys, threading, types
from http.client import HTTPConnection

for name, attrs in (("torch", {"__version__": "0", "set_num_threads": lambda n: None,
                               "get_num_threads": lambda: 1}),
                    ("laya", {"load": lambda *a, **k: None}),
                    ("chunklaya", {"ChunkLaya": object})):
    m = types.ModuleType(name)
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server

server.STATE.update(phase="ready", ready=True)
server.API_KEY = None
server.answer = lambda req: {"echo_keys": sorted(req.keys()), "state_len": len(req.get("state", ""))}

from http.server import ThreadingHTTPServer
srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

BODY = json.dumps({"state": "x" * 5000, "questions": {"q": {"type": "noul"}}}).encode()

def chunked(body, size=1400):
    out = b""
    for i in range(0, len(body), size):
        part = body[i:i+size]
        out += b"%x\r\n" % len(part) + part + b"\r\n"
    return out + b"0\r\n\r\n"

def raw_request(headers, payload, path="/ask", method="POST"):
    c = HTTPConnection("127.0.0.1", port, timeout=10)
    c.putrequest(method, path, skip_accept_encoding=True, skip_host=True)
    c.putheader("Host", "127.0.0.1")
    for k, v in headers.items(): c.putheader(k, v)
    c.endheaders()
    if payload: c.send(payload)
    r = c.getresponse(); data = r.read()
    return r.status, data, c

ok = True
def check(label, got, want):
    global ok
    good = got == want
    ok &= good
    print(("PASS " if good else "FAIL ") + label + f"  got={got!r} want={want!r}")

# 1. Content-Length still works
s, d, _ = raw_request({"Content-Type": "application/json", "Content-Length": str(len(BODY))}, BODY)
check("content-length", (s, json.loads(d)["state_len"]), (200, 5000))

# 2. chunked now works
s, d, _ = raw_request({"Content-Type": "application/json", "Transfer-Encoding": "chunked"}, chunked(BODY))
check("chunked", (s, json.loads(d)["state_len"]), (200, 5000))

# 3. chunked, single chunk
s, d, _ = raw_request({"Content-Type": "application/json", "Transfer-Encoding": "chunked"}, chunked(BODY, len(BODY)))
check("chunked single", (s, json.loads(d)["state_len"]), (200, 5000))

# 4. chunked with a size extension and a trailer
payload = b"%x;ext=1\r\n" % len(BODY) + BODY + b"\r\n0\r\nX-Trailer: v\r\n\r\n"
s, d, _ = raw_request({"Content-Type": "application/json", "Transfer-Encoding": "chunked"}, payload)
check("chunked ext+trailer", (s, json.loads(d)["state_len"]), (200, 5000))

# 5. empty chunked body -> 400 empty body, not a hang
s, d, _ = raw_request({"Content-Type": "application/json", "Transfer-Encoding": "chunked"}, b"0\r\n\r\n")
check("chunked empty", (s, json.loads(d).get("error")), (400, "empty body"))

# 6. malformed chunk size -> 400 malformed, connection closed
s, d, _ = raw_request({"Content-Type": "application/json", "Transfer-Encoding": "chunked"}, b"zz\r\nabc\r\n0\r\n\r\n")
check("chunked malformed", (s, json.loads(d)["error"].startswith("malformed body")), (400, True))

# 7. over-size chunked -> 413 without buffering it all
server.MAX_BODY = 4096
s, d, _ = raw_request({"Content-Type": "application/json", "Transfer-Encoding": "chunked"}, chunked(BODY))
check("chunked too large", (s, json.loads(d)["error"].startswith("body over")), (413, True))
server.MAX_BODY = 64 * 1024 * 1024

# 8. keep-alive integrity: a 401 must not poison the next request on the same connection
server.API_KEY = "secret"
c = HTTPConnection("127.0.0.1", port, timeout=10)
c.request("POST", "/ask", body=chunked(BODY), headers={"Content-Type": "application/json", "Transfer-Encoding": "chunked"})
r1 = c.getresponse(); b1 = r1.read()
c.request("GET", "/health")
r2 = c.getresponse(); b2 = r2.read()
check("401 then health, one connection", (r1.status, r2.status, json.loads(b2).get("status")), (401, 200, "ok"))
server.API_KEY = None

print("\nALL PASS" if ok else "\nFAILURES")
sys.exit(0 if ok else 1)
