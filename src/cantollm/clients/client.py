"""Streaming client and REPL for the CantoLLM Messages API."""

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
                # No boundary yet — hold the partial word in the buffer
                break

            word = self._buf[:sp]
            sep = self._buf[sp]
            self._buf = self._buf[sp + 1:]

            self._emit_word(word)

            if sep == "\n":
                print(flush=True)
                self._col = 0
            else:
                # Space — only print it if we're not at column 0
                if self._col > 0:
                    print(" ", end="", flush=True)
                    self._col += 1

    def flush(self):
        """Print any remaining buffered text."""
        if self._buf:
            self._emit_word(self._buf)
            self._buf = ""

    def _emit_word(self, word: str):
        if not word:
            return
        # Would this word overflow the line?
        needed = len(word) + (1 if self._col > 0 else 0)
        if self._col > 0 and self._col + needed > self._width:
            print(flush=True)  # Wrap to next line
            self._col = 0
        print(word, end="", flush=True)
        self._col += len(word)


class ChatClient:
    """Client for the Anthropic-compatible Messages API with streaming SSE support."""

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
        """GET /v1/models and set self.model to the first entry. Returns the id."""
        url = f"{self.base_url}/v1/models"
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = json.loads(resp.read())
        data = body.get("data", [])
        if not data:
            raise RuntimeError(f"Server at {self.base_url} has no registered models")
        self.model = data[0]["id"]
        return self.model

    def reset(self):
        """Clear conversation history."""
        self.messages = []

    def send_message(self, text: str, stream: bool = True) -> dict | None:
        """Send a message and handle the response.

        For streaming: prints tokens as they arrive, returns final usage stats.
        For non-streaming: returns the full response dict.
        """
        self.messages.append({"role": "user", "content": text})

        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": stream,
        }

        if stream:
            return self._send_streaming(body)
        else:
            return self._send_sync(body)

    def _send_sync(self, body: dict) -> dict:
        """Non-streaming request via urllib."""
        url = f"{self.base_url}/v1/messages"
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            response = json.loads(resp.read())

        # Extract assistant text and add to history
        text_parts = []
        for block in response.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block["text"])
        if text_parts:
            self.messages.append({"role": "assistant", "content": "\n".join(text_parts)})

        return response

    def _send_streaming(self, body: dict) -> dict | None:
        """Streaming request via http.client for line-by-line SSE control."""
        parsed = urllib.parse.urlparse(self.base_url)
        host = parsed.hostname
        if parsed.scheme == "https":
            port = parsed.port or 443
            conn = http.client.HTTPSConnection(host, port)
        else:
            port = parsed.port or 80
            conn = http.client.HTTPConnection(host, port)
        spinner_stop = threading.Event()
        thinking_count = [0]  # Mutable container shared with spinner thread

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
            conn.request("POST", "/v1/messages", body=data,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()

            if resp.status != 200:
                spinner_stop.set()
                if spinner_thread:
                    spinner_thread.join()
                error_body = resp.read().decode()
                try:
                    error = json.loads(error_body)
                    msg = error.get("error", {}).get("message", error_body)
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

            result = self._parse_sse_stream(resp, spinner_stop, thinking_count)
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

    def _parse_sse_stream(self, resp, spinner_stop, thinking_count) -> dict | None:
        """Parse SSE events from an HTTP response, printing content as it arrives."""
        assistant_text = []
        in_thinking = False
        usage = {}
        stop_reason = None
        spinner_stopped = False
        wrapper = None
        first_token_time = None
        error_message: str | None = None

        current_event = None
        current_data = None

        prefix = "Assistant: "

        def stop_spinner():
            nonlocal spinner_stopped, wrapper
            if not spinner_stopped:
                spinner_stop.set()
                if not self.quiet:
                    # Clear spinner line and print the assistant label
                    print(f"\r{' ' * 40}\r{Colors.GREEN}{prefix}{Colors.RESET}",
                          end="", flush=True)
                    wrapper = WordWrapper(initial_col=len(prefix))
                spinner_stopped = True

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
                        msg = data.get("message", {})
                        usage.update(msg.get("usage", {}))

                    case "content_block_start":
                        block = data.get("content_block", {})
                        if block.get("type") == "thinking":
                            in_thinking = True
                            if self.show_thinking and not self.quiet:
                                stop_spinner()
                                print(f"{Colors.GRAY}<think>", end="", flush=True)

                    case "content_block_delta":
                        delta = data.get("delta", {})
                        delta_type = delta.get("type")
                        if delta_type == "thinking_delta":
                            thinking_count[0] += 1
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            if self.show_thinking and not self.quiet:
                                stop_spinner()
                                print(delta.get("thinking", ""), end="", flush=True)
                        elif delta_type == "text_delta":
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            stop_spinner()
                            text = delta.get("text", "")
                            assistant_text.append(text)
                            if not self.quiet:
                                wrapper.write(text)

                    case "content_block_stop":
                        if in_thinking:
                            in_thinking = False
                            if self.show_thinking and not self.quiet:
                                print(f"</think>{Colors.GREEN}", end="", flush=True)

                    case "message_delta":
                        delta = data.get("delta", {})
                        stop_reason = delta.get("stop_reason")
                        usage.update(data.get("usage", {}))

                    case "error":
                        err = data.get("error", {})
                        error_message = err.get("message", "unknown error")
                        current_event = None
                        current_data = None
                        break

                    case "message_stop":
                        current_event = None
                        current_data = None
                        break

                current_event = None
                current_data = None

        stop_spinner()
        if wrapper:
            wrapper.flush()

        if error_message is not None:
            # Mid-stream server error: drop the user message we optimistically
            # appended so history doesn't get poisoned with a turn the server
            # never completed.
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            if not self.quiet:
                print(f"\n{Colors.GRAY}Error: {error_message}{Colors.RESET}")
            return {"usage": usage, "stop_reason": None,
                    "first_token_time": first_token_time,
                    "error": error_message}

        # Store assistant response in history (text only, not thinking)
        text = "".join(assistant_text)
        if text:
            self.messages.append({"role": "assistant", "content": text})

        return {"usage": usage, "stop_reason": stop_reason,
                "first_token_time": first_token_time}


def run_client(base_url: str, temperature: float = 0.7, top_p: float = 0.9,
               max_tokens: int = 2048, show_thinking: bool = False):
    """Run the interactive chat REPL."""
    client = ChatClient(base_url, temperature=temperature, top_p=top_p,
                        max_tokens=max_tokens, show_thinking=show_thinking)

    # Check server health and discover the model
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

        print()  # Blank line before response
        start_time = time.perf_counter()

        try:
            result = client.send_message(prompt, stream=True)
        except Exception as e:
            print(f"\n{Colors.GRAY}Connection error: {e}{Colors.RESET}")
            # Remove the message we tried to send
            if client.messages and client.messages[-1]["role"] == "user":
                client.messages.pop()
            print()
            continue

        elapsed = time.perf_counter() - start_time
        print(f"{Colors.RESET}")

        # Stats line
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
