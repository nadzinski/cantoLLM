"""HTTP server with Anthropic-compatible Messages API."""

import json
import time
import uuid
from collections.abc import Callable, Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer

from qwen3.api_types import MessagesRequest, make_message_response, make_sse_event
from qwen3.decoder import StreamingDecoder
from qwen3.kv_cache import KVCache
from qwen3.stats import StatsCollector
from qwen3.stream_events import TextChunk, ThinkingEndEvent, ThinkingStartEvent


class InferenceServer:
    """Handles inference requests, decoupled from HTTP layer.

    Each request gets a fresh KVCache and TokenGenerator (via factory),
    matching the stateless Anthropic API contract.
    """

    def __init__(self, model, tokenizer, generator_factory, config, device, model_name="qwen3"):
        """
        Args:
            model: The loaded Qwen3 model.
            tokenizer: Qwen3Tokenizer instance.
            generator_factory: Callable(temperature, top_p) -> generator with .generate().
            config: Model config dict (needs num_transformers).
            device: torch device.
            model_name: Name to report in responses.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.generator_factory = generator_factory
        self.config = config
        self.device = device
        self.model_name = model_name

    def _prepare(self, req: MessagesRequest):
        """Common setup: encode input, create cache and generator."""
        input_ids = self.tokenizer.encode_conversation(req.messages, system=req.system)
        cache = KVCache(self.config["num_transformers"])
        generator = self.generator_factory(req.temperature, req.top_p)
        return input_ids, cache, generator

    def handle_request(self, body: dict) -> dict:
        """Handle a non-streaming Messages API request.

        Returns an Anthropic-format Message response dict.
        Raises ValueError for invalid requests.
        """
        req = MessagesRequest.from_dict(body)
        input_ids, cache, generator = self._prepare(req)

        # Generate all tokens
        stats = StatsCollector()
        token_stream = generator.generate(
            input_ids=input_ids,
            cache=cache,
            stop_token_ids=self.tokenizer.stop_token_ids,
            max_tokens=req.max_tokens,
        )
        token_stream = stats.wrap(token_stream)

        # Decode to events, collect content blocks
        decoder = StreamingDecoder(self.tokenizer)
        content_blocks = []
        current_text = []
        current_thinking = []
        in_thinking = False

        for event in decoder.wrap(token_stream):
            match event:
                case ThinkingStartEvent():
                    in_thinking = True
                case ThinkingEndEvent():
                    thinking_text = "".join(current_thinking)
                    if thinking_text:
                        content_blocks.append({"type": "thinking", "thinking": thinking_text})
                    current_thinking = []
                    in_thinking = False
                case TextChunk(text=t):
                    if in_thinking:
                        current_thinking.append(t)
                    else:
                        current_text.append(t)

        # Finalize text block
        text = "".join(current_text)
        if text:
            content_blocks.append({"type": "text", "text": text})

        gen_stats = stats.get_stats()
        output_tokens = gen_stats.total_tokens
        stop_reason = "max_tokens" if output_tokens >= req.max_tokens else "end_turn"

        return make_message_response(
            msg_id=f"msg_{uuid.uuid4().hex[:24]}",
            content_blocks=content_blocks,
            model=self.model_name,
            stop_reason=stop_reason,
            input_tokens=len(input_ids),
            output_tokens=output_tokens,
        )

    def handle_request_stream(self, body: dict) -> Iterator[str]:
        """Handle a streaming Messages API request.

        Yields SSE event strings following Anthropic's streaming format.
        Raises ValueError for invalid requests.
        """
        req = MessagesRequest.from_dict(body)
        input_ids, cache, generator = self._prepare(req)

        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        # message_start
        yield make_sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model_name,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": len(input_ids), "output_tokens": 0},
            },
        })

        # Generate tokens
        stats = StatsCollector()
        token_stream = generator.generate(
            input_ids=input_ids,
            cache=cache,
            stop_token_ids=self.tokenizer.stop_token_ids,
            max_tokens=req.max_tokens,
        )
        token_stream = stats.wrap(token_stream)

        # Count tokens by phase (thinking vs text) at the raw token level
        think_start_id = self.tokenizer._special_to_id.get("<think>")
        think_end_id = self.tokenizer._special_to_id.get("</think>")
        thinking_tokens = 0
        text_tokens = 0
        phase_is_thinking = False

        def phase_counted(stream):
            nonlocal thinking_tokens, text_tokens, phase_is_thinking
            for token_id in stream:
                if token_id == think_start_id:
                    phase_is_thinking = True
                    thinking_tokens += 1
                elif token_id == think_end_id:
                    thinking_tokens += 1
                    phase_is_thinking = False
                elif phase_is_thinking:
                    thinking_tokens += 1
                else:
                    text_tokens += 1
                yield token_id

        decoder = StreamingDecoder(self.tokenizer)
        block_index = 0
        in_thinking = False
        started_text_block = False

        for event in decoder.wrap(phase_counted(token_stream)):
            match event:
                case ThinkingStartEvent():
                    in_thinking = True
                    yield make_sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    })

                case ThinkingEndEvent():
                    in_thinking = False
                    yield make_sse_event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": block_index,
                    })
                    block_index += 1

                case TextChunk(text=t):
                    if in_thinking:
                        yield make_sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "thinking_delta", "thinking": t},
                        })
                    else:
                        if not started_text_block:
                            yield make_sse_event("content_block_start", {
                                "type": "content_block_start",
                                "index": block_index,
                                "content_block": {"type": "text", "text": ""},
                            })
                            started_text_block = True
                        yield make_sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "text_delta", "text": t},
                        })

        # Close text block if opened
        if started_text_block:
            yield make_sse_event("content_block_stop", {
                "type": "content_block_stop",
                "index": block_index,
            })

        gen_stats = stats.get_stats()
        output_tokens = gen_stats.total_tokens
        stop_reason = "max_tokens" if output_tokens >= req.max_tokens else "end_turn"

        # message_delta with final usage
        yield make_sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
                "text_tokens": text_tokens,
            },
        })

        yield make_sse_event("message_stop", {"type": "message_stop"})


class MessageHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Messages API."""

    # Disable output buffering for SSE
    wbufsize = 0

    def log_message(self, format, *args):
        """Suppress default request logging."""
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400):
        self._send_json(
            {"type": "error", "error": {"type": "invalid_request_error", "message": message}},
            status=status,
        )

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path != "/v1/messages":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        # Read body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_error("empty request body")
            return

        try:
            raw = self.rfile.read(content_length)
            body = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as e:
            self._send_error(f"invalid JSON: {e}")
            return

        server: InferenceServer = self.server.inference_server

        if body.get("stream"):
            self._handle_stream(server, body)
        else:
            self._handle_sync(server, body)

    def _handle_sync(self, server: InferenceServer, body: dict):
        try:
            response = server.handle_request(body)
            self._send_json(response)
        except ValueError as e:
            self._send_error(str(e))

    def _handle_stream(self, server: InferenceServer, body: dict):
        try:
            event_iter = server.handle_request_stream(body)
        except ValueError as e:
            self._send_error(str(e))
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            for event_str in event_iter:
                self.wfile.write(event_str.encode())
                self.wfile.flush()
        except BrokenPipeError:
            pass  # Client disconnected


def run_server(host: str, port: int, inference_server: InferenceServer):
    """Start the HTTP server."""
    httpd = HTTPServer((host, port), MessageHandler)
    httpd.inference_server = inference_server

    print(f"\nCantoLLM server running on http://{host}:{port}")
    print(f"  POST /v1/messages  — Anthropic-compatible Messages API")
    print(f"  GET  /health       — Health check")
    print(f"\nModel: {inference_server.model_name}")
    print("Press Ctrl+C to stop.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        httpd.shutdown()
