"""Streaming client and REPL for the CantoLLM server.

Supports both the Anthropic Messages API (`/v1/messages`) and the OpenAI
Chat Completions API (`/v1/chat/completions`) against the same server. The
dialect is selected via `run_client(api=...)`.
"""

import http.client
import json
import shutil
import threading
import time
import urllib.parse
import urllib.request


class Colors:
    """ANSI color codes."""

    BOLD_CYAN = "\033[1;36m"
    GREEN = "\033[32m"
    GRAY = "\033[90m"
    RESET = "\033[0m"
    SEPARATOR = "\033[90m"
    TITLE = "\033[1;38;5;214m"


class WordWrapper:
    """Buffers streaming text and prints with clean word wrapping."""

    def __init__(self, initial_col: int = 0):
        self._col = initial_col
        self._width = shutil.get_terminal_size().columns
        self._buf = ""

    def write(self, text: str):
        """Accept a chunk of streaming text, printing complete words with wrapping."""
        self._buf += text
        while self._buf:
            # Find the next word boundary (space or newline)
            sp = -1
            for i, ch in enumerate(self._buf):
                if ch in (" ", "\n"):
                    sp = i
                    break

            if sp == -1:
                break

            word = self._buf[:sp]
            sep = self._buf[sp]
            self._buf = self._buf[sp + 1:]

            self._emit_word(word)

            if sep == "\n":
                print(flush=True)
                self._col = 0
            else:
                if self._col > 0:
                    print(" ", end="", flush=True)
                    self._col += 1

    def flush(self):
        if self._buf:
            self._emit_word(self._buf)
            self._buf = ""

    def _emit_word(self, word: str):
        if not word:
            return
        needed = len(word) + (1 if self._col > 0 else 0)
        if self._col > 0 and self._col + needed > self._width:
            print(flush=True)
            self._col = 0
        print(word, end="", flush=True)
        self._col += len(word)


# ── Base client ──────────────────────────────────────────────────────


class _BaseChatClient:
    """Shared scaffolding for dialect-specific chat clients.

    Subclasses supply the endpoint path, request-body shape, and SSE parser.
    """

    # Subclass overrides:
    PATH: str = ""

    def __init__(self, base_url: str, temperature: float = 0.7, top_p: float = 0.9,
                 max_tokens: int = 2048, model: str | None = None,
                 show_thinking: bool = False, quiet: bool = False):
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.model = model
        self.show_thinking = show_thinking
        self.quiet = quiet
        self.messages: list[dict] = []

    def fetch_model(self) -> str:
        """GET /v1/models and set self.model to the first entry."""
        url = f"{self.base_url}/v1/models"
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = json.loads(resp.read())
        data = body.get("data", [])
        if not data:
            raise RuntimeError(f"Server at {self.base_url} has no registered models")
        self.model = data[0]["id"]
        return self.model

    def reset(self):
        self.messages = []

    # ── Subclass hooks ──

    def _build_body(self, stream: bool) -> dict:
        raise NotImplementedError

    def _parse_stream(self, resp, spinner_stop, thinking_count) -> dict | None:
        raise NotImplementedError

    # ── Send ──

    def send_message(self, text: str, stream: bool = True) -> dict | None:
        self.messages.append({"role": "user", "content": text})
        body = self._build_body(stream)
        if stream:
            return self._send_streaming(body)
        return self._send_sync(body)

    def _send_sync(self, body: dict) -> dict:
        url = f"{self.base_url}{self.PATH}"
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            response = json.loads(resp.read())
        self._append_assistant_from_sync(response)
        return response

    def _append_assistant_from_sync(self, response: dict) -> None:
        raise NotImplementedError

    def _send_streaming(self, body: dict) -> dict | None:
        parsed = urllib.parse.urlparse(self.base_url)
        host = parsed.hostname
        if parsed.scheme == "https":
            port = parsed.port or 443
            conn = http.client.HTTPSConnection(host, port)
        else:
            port = parsed.port or 80
            conn = http.client.HTTPConnection(host, port)
        spinner_stop = threading.Event()
        thinking_count = [0]

        def spin():
            frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            i = 0
            while not spinner_stop.is_set():
                n = thinking_count[0]
                count_str = f" ({n})" if n > 0 else ""
                print(f"\r{Colors.RESET}{frames[i % len(frames)]} {Colors.GRAY}thinking...{count_str}{Colors.RESET}  ",
                      end="", flush=True)
                i += 1
                spinner_stop.wait(0.08)

        spinner_thread = None
        if not self.quiet:
            spinner_thread = threading.Thread(target=spin, daemon=True)
            spinner_thread.start()

        start_time = time.perf_counter()

        try:
            data = json.dumps(body).encode()
            conn.request("POST", self.PATH, body=data,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()

            if resp.status != 200:
                spinner_stop.set()
                if spinner_thread:
                    spinner_thread.join()
                error_body = resp.read().decode()
                try:
                    error = json.loads(error_body)
                    # Anthropic 4xx shape: {"detail": "..."}; OpenAI shape:
                    # {"error": {"message": "..."}}. Accept either.
                    msg = (error.get("error", {}).get("message")
                           or error.get("detail")
                           or error_body)
                except json.JSONDecodeError:
                    msg = error_body
                self.messages.pop()
                if self.quiet:
                    return {"error": f"HTTP {resp.status}: {msg}",
                            "start_time": start_time,
                            "end_time": time.perf_counter(),
                            "first_token_time": None,
                            "usage": {}, "stop_reason": None}
                print(f"\r{' ' * 40}\r", end="", flush=True)
                print(f"\n{Colors.GRAY}Error: {msg}{Colors.RESET}")
                return None

            result = self._parse_stream(resp, spinner_stop, thinking_count)
            if result is not None:
                result.setdefault("start_time", start_time)
                result.setdefault("end_time", time.perf_counter())
                result.setdefault("error", None)
            return result
        finally:
            spinner_stop.set()
            if spinner_thread:
                spinner_thread.join()
            conn.close()

    # ── Spinner/label coordination (shared) ──

    def _make_stop_spinner(self, spinner_stop, state):
        """Produce a closure that stops the spinner and prints the assistant
        label the first time it's called."""
        prefix = "Assistant: "

        def stop():
            if state["spinner_stopped"]:
                return
            spinner_stop.set()
            if not self.quiet:
                print(f"\r{' ' * 40}\r{Colors.GREEN}{prefix}{Colors.RESET}",
                      end="", flush=True)
                state["wrapper"] = WordWrapper(initial_col=len(prefix))
            state["spinner_stopped"] = True

        return stop


# ── Anthropic ────────────────────────────────────────────────────────


class AnthropicChatClient(_BaseChatClient):
    PATH = "/v1/messages"

    def _build_body(self, stream: bool) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": stream,
        }

    def _append_assistant_from_sync(self, response: dict) -> None:
        text_parts = [b["text"] for b in response.get("content", [])
                      if b.get("type") == "text"]
        if text_parts:
            self.messages.append(
                {"role": "assistant", "content": "\n".join(text_parts)}
            )

    def _parse_stream(self, resp, spinner_stop, thinking_count) -> dict | None:
        assistant_text: list[str] = []
        in_thinking = False
        usage: dict = {}
        stop_reason = None
        first_token_time: float | None = None
        error_message: str | None = None
        state = {"spinner_stopped": False, "wrapper": None}
        stop = self._make_stop_spinner(spinner_stop, state)

        current_event = None
        current_data = None

        for line_bytes in resp:
            line = line_bytes.decode("utf-8").rstrip("\n").rstrip("\r")

            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: "):
                current_data = line[6:]
            elif line == "" and current_event and current_data:
                try:
                    data = json.loads(current_data)
                except json.JSONDecodeError:
                    current_event = None
                    current_data = None
                    continue

                match current_event:
                    case "message_start":
                        usage.update(data.get("message", {}).get("usage", {}))
                    case "content_block_start":
                        if data.get("content_block", {}).get("type") == "thinking":
                            in_thinking = True
                            if self.show_thinking and not self.quiet:
                                stop()
                                print(f"{Colors.GRAY}<think>", end="", flush=True)
                    case "content_block_delta":
                        delta = data.get("delta", {})
                        dtype = delta.get("type")
                        if dtype == "thinking_delta":
                            thinking_count[0] += 1
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            if self.show_thinking and not self.quiet:
                                stop()
                                print(delta.get("thinking", ""), end="", flush=True)
                        elif dtype == "text_delta":
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            stop()
                            text = delta.get("text", "")
                            assistant_text.append(text)
                            if not self.quiet:
                                state["wrapper"].write(text)
                    case "content_block_stop":
                        if in_thinking:
                            in_thinking = False
                            if self.show_thinking and not self.quiet:
                                print(f"</think>{Colors.GREEN}", end="", flush=True)
                    case "message_delta":
                        stop_reason = data.get("delta", {}).get("stop_reason")
                        usage.update(data.get("usage", {}))
                    case "error":
                        error_message = data.get("error", {}).get("message", "unknown error")
                        current_event = None
                        current_data = None
                        break
                    case "message_stop":
                        current_event = None
                        current_data = None
                        break

                current_event = None
                current_data = None

        stop()
        if state["wrapper"]:
            state["wrapper"].flush()

        if error_message is not None:
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            if not self.quiet:
                print(f"\n{Colors.GRAY}Error: {error_message}{Colors.RESET}")
            return {"usage": usage, "stop_reason": None,
                    "first_token_time": first_token_time,
                    "error": error_message}

        text = "".join(assistant_text)
        if text:
            self.messages.append({"role": "assistant", "content": text})

        return {"usage": usage, "stop_reason": stop_reason,
                "first_token_time": first_token_time}


# ── OpenAI ───────────────────────────────────────────────────────────


class OpenAIChatClient(_BaseChatClient):
    PATH = "/v1/chat/completions"

    def _build_body(self, stream: bool) -> dict:
        body = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "messages": self.messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": stream,
        }
        if stream:
            # Populates the stats line with completion/prompt/reasoning counts.
            body["stream_options"] = {"include_usage": True}
        return body

    def _append_assistant_from_sync(self, response: dict) -> None:
        choices = response.get("choices", [])
        if not choices:
            return
        content = choices[0].get("message", {}).get("content")
        if content:
            self.messages.append({"role": "assistant", "content": content})

    def _parse_stream(self, resp, spinner_stop, thinking_count) -> dict | None:
        assistant_text: list[str] = []
        usage: dict = {}
        stop_reason = None
        first_token_time: float | None = None
        error_message: str | None = None
        state = {"spinner_stopped": False, "wrapper": None}
        stop = self._make_stop_spinner(spinner_stop, state)
        think_open = False

        def close_think():
            nonlocal think_open
            if think_open and self.show_thinking and not self.quiet:
                print(f"</think>{Colors.GREEN}", end="", flush=True)
            think_open = False

        # Accumulate raw body line-by-line; OpenAI SSE has no `event:` lines.
        for line_bytes in resp:
            line = line_bytes.decode("utf-8").rstrip("\n").rstrip("\r")
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if "error" in data:
                error_message = data["error"].get("message", "unknown error")
                break

            # Usage chunk: empty choices, populated usage.
            if not data.get("choices"):
                if data.get("usage"):
                    usage = data["usage"]
                continue

            choice = data["choices"][0]
            delta = choice.get("delta", {})

            reasoning = delta.get("reasoning_content")
            content = delta.get("content")

            if reasoning:
                thinking_count[0] += 1
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                if self.show_thinking and not self.quiet:
                    stop()
                    if not think_open:
                        print(f"{Colors.GRAY}<think>", end="", flush=True)
                        think_open = True
                    print(reasoning, end="", flush=True)

            if content:
                close_think()
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                stop()
                assistant_text.append(content)
                if not self.quiet:
                    state["wrapper"].write(content)

            if choice.get("finish_reason"):
                stop_reason = choice["finish_reason"]

        stop()
        close_think()
        if state["wrapper"]:
            state["wrapper"].flush()

        if error_message is not None:
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            if not self.quiet:
                print(f"\n{Colors.GRAY}Error: {error_message}{Colors.RESET}")
            return {"usage": {}, "stop_reason": None,
                    "first_token_time": first_token_time,
                    "error": error_message}

        text = "".join(assistant_text)
        if text:
            self.messages.append({"role": "assistant", "content": text})

        # Normalize OpenAI usage fields into the shape the stats formatter
        # expects (input_tokens / output_tokens / thinking_tokens / text_tokens).
        normalized: dict = {}
        if usage:
            details = usage.get("completion_tokens_details") or {}
            reasoning_tokens = details.get("reasoning_tokens", 0)
            completion = usage.get("completion_tokens", 0)
            normalized = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": completion + reasoning_tokens,
                "thinking_tokens": reasoning_tokens,
                "text_tokens": completion,
            }

        return {"usage": normalized, "stop_reason": stop_reason,
                "first_token_time": first_token_time}


# ── REPL ─────────────────────────────────────────────────────────────


_CLIENTS: dict[str, type[_BaseChatClient]] = {
    "anthropic": AnthropicChatClient,
    "openai": OpenAIChatClient,
}


def run_client(base_url: str, *, api: str = "anthropic",
               temperature: float = 0.7, top_p: float = 0.9,
               max_tokens: int = 2048, show_thinking: bool = False):
    """Run the interactive chat REPL against the chosen dialect."""
    client_cls = _CLIENTS[api]
    client = client_cls(base_url, temperature=temperature, top_p=top_p,
                        max_tokens=max_tokens, show_thinking=show_thinking)

    try:
        url = f"{base_url.rstrip('/')}/health"
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status != 200:
                print(f"Server at {base_url} returned status {resp.status}")
                return
        client.fetch_model()
    except Exception as e:
        print(f"Cannot connect to server at {base_url}: {e}")
        return

    sep = f"{Colors.SEPARATOR}{'─' * 60}{Colors.RESET}"

    print(f"\n{sep}")
    print(f"  {Colors.TITLE}CantoLLM Chat Client{Colors.RESET}")
    print(f"  API:         {api}")
    print(f"  Server:      {base_url}")
    print(f"  Model:       {client.model}")
    print(f"  Temperature: {temperature}  Top-p: {top_p}  Max tokens: {max_tokens}")
    print(f"{sep}")
    print("  Type 'quit' or 'exit' to end, 'reset' to clear history.\n")

    while True:
        try:
            prompt = input(f"{Colors.BOLD_CYAN}You: {Colors.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"{Colors.RESET}\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if prompt.lower() == "reset":
            client.reset()
            print("Conversation reset.\n")
            continue

        print()
        start_time = time.perf_counter()

        try:
            result = client.send_message(prompt, stream=True)
        except Exception as e:
            print(f"\n{Colors.GRAY}Connection error: {e}{Colors.RESET}")
            if client.messages and client.messages[-1]["role"] == "user":
                client.messages.pop()
            print()
            continue

        elapsed = time.perf_counter() - start_time
        print(f"{Colors.RESET}")

        if result and result.get("usage"):
            usage = result["usage"]
            input_tok = usage.get("input_tokens", 0)
            output_tok = usage.get("output_tokens", 0)
            thinking_tok = usage.get("thinking_tokens", 0)
            text_tok = usage.get("text_tokens", 0)
            tps = output_tok / elapsed if elapsed > 0 else 0
            print(f"{Colors.GRAY}[{input_tok} in / {output_tok} out "
                  f"({thinking_tok} thinking + {text_tok} text) | "
                  f"{tps:.1f} tok/s | {elapsed:.1f}s]{Colors.RESET}")

        print(f"{sep}\n")


# ── Back-compat re-export ────────────────────────────────────────────

# bench.py imports ChatClient. Keep the name pointed at the Anthropic
# implementation so existing benchmarks don't need updating.
ChatClient = AnthropicChatClient
