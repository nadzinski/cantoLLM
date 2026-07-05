"""Live tokenizer API for the viz Tokenizer tab.

Serves the REAL Qwen3Tokenizer (tokenizer.json only — no model weights) over
a tiny stdlib HTTP server so viz/index.html can tokenize arbitrary text.
CORS is wide open so the page works whether opened via file:// or via this
server (it also serves the viz/ directory statically as a convenience).

Run from the repo root:  .venv/bin/python viz/tokenizer_server.py [port]
Then open viz/index.html (file://) or http://127.0.0.1:8765/
"""

import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ_DIR = Path(__file__).resolve().parent
TOKENIZER_JSON = (
    REPO_ROOT / "src/cantollm/models/model_data/Qwen3-0.6B/tokenizer.json"
)
DEFAULT_PORT = 8765

sys.path.insert(0, str(REPO_ROOT / "src"))
from cantollm.models.qwen3.tokenizer import Qwen3Tokenizer  # noqa: E402

ROLE_NOTES = {
    "eos": "eos_token_id — in stop_token_ids; ends generation (never emitted to the client)",
    "pad": "pad_token_id — also in stop_token_ids",
    "think_start": "thinking_start_id — phase.py flips to the thinking phase; StreamingDecoder emits ThinkingStartEvent",
    "think_end": "thinking_end_id — back to visible text; StreamingDecoder emits ThinkingEndEvent",
}


def build_meta(tokenizer: Qwen3Tokenizer) -> dict:
    added = json.load(open(TOKENIZER_JSON))["added_tokens"]
    recognized = tokenizer._special_to_id  # the project's _SPECIAL_TOKENS map
    role_by_id = {
        tokenizer.eos_token_id: "eos",
        tokenizer.pad_token_id: "pad",
        tokenizer.thinking_start_id: "think_start",
        tokenizer.thinking_end_id: "think_end",
    }
    specials = [
        {
            "id": a["id"],
            "token": a["content"],
            "hf_special": a["special"],
            "recognized": a["content"] in recognized,
            "role": role_by_id.get(a["id"], ""),
            "note": ROLE_NOTES.get(role_by_id.get(a["id"], ""), ""),
        }
        for a in sorted(added, key=lambda a: a["id"])
    ]
    return {
        "model": "Qwen3 (tokenizer.json from model_data/Qwen3-0.6B)",
        "vocab_size": tokenizer._tok.get_vocab_size(with_added_tokens=True),
        "stop_token_ids": sorted(tokenizer.stop_token_ids),
        "specials": specials,
    }


class Handler(SimpleHTTPRequestHandler):
    tokenizer: Qwen3Tokenizer = None
    meta: dict = None

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/meta":
            self._json(self.meta)
        else:
            super().do_GET()  # static files from viz/

    def do_POST(self):
        if self.path != "/api/tokenize":
            self._json({"error": "unknown endpoint"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            text = req.get("text", "")
            chat_wrapped = bool(req.get("chat_wrapped", False))
            tok = self.tokenizer
            ids = tok.encode(text, chat_wrapped=chat_wrapped) if text else []
            self._json(
                {
                    "ids": ids,
                    "pieces": [tok.decode([i]) for i in ids],
                    "vocab_tokens": [tok._tok.id_to_token(i) for i in ids],
                    "text_decoded": tok.decode(ids) if ids else "",
                }
            )
        except Exception as e:  # report parse/encode errors to the UI
            self._json({"error": str(e)}, 400)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    tokenizer = Qwen3Tokenizer(str(TOKENIZER_JSON))
    Handler.tokenizer = tokenizer
    Handler.meta = build_meta(tokenizer)
    handler = partial(Handler, directory=str(VIZ_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"tokenizer API on http://127.0.0.1:{port}/api/tokenize")
    print(f"(also serving viz/ at http://127.0.0.1:{port}/ — Ctrl-C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
