"""Terminal presentation layer for chat interface."""

from qwen3.stats import GenerationStats, SpeculativeStats
from qwen3.stream_events import StreamEvent, TextChunk, ThinkingEndEvent, ThinkingStartEvent


class Colors:
    """ANSI color codes for terminal output."""

    USER = "\033[36m"  # Cyan
    ASSISTANT = "\033[32m"  # Green
    THINK = "\033[90m"  # Gray
    RESET = "\033[0m"


class TerminalPresenter:
    """Handles terminal-specific display of chat responses.

    Processes StreamEvents and renders them with appropriate colors
    and formatting for terminal output.
    """

    def __init__(self, tokenizer=None):
        """Initialize presenter.

        Args:
            tokenizer: Optional tokenizer for decoding token windows in stats.
                       If not provided, stats won't show token text.
        """
        self._tokenizer = tokenizer
        self._in_thinking = False

    def print_user_prompt(self, prompt: str):
        """Display the user's input prompt."""
        print(f"{Colors.USER}You: {Colors.RESET}{prompt}")

    def start_response(self):
        """Mark the start of an assistant response."""
        print(f"\n{Colors.ASSISTANT}Assistant: {Colors.RESET}", end="")

    def handle_event(self, event: StreamEvent):
        """Process a single stream event.

        Args:
            event: A StreamEvent (ThinkingStartEvent, ThinkingEndEvent, or TextChunk)
        """
        match event:
            case ThinkingStartEvent():
                print(f"{Colors.THINK}<think>", end="", flush=True)
                self._in_thinking = True
            case ThinkingEndEvent():
                print(f"</think>{Colors.ASSISTANT}", end="", flush=True)
                self._in_thinking = False
            case TextChunk(text=t):
                print(t, end="", flush=True)

    def end_response(self):
        """Mark the end of an assistant response."""
        print(f"{Colors.RESET}\n")

    def print_stats(self, stats: GenerationStats, spec_stats: SpeculativeStats | None = None):
        """Display generation statistics.

        Args:
            stats: GenerationStats from StatsCollector
            spec_stats: Optional SpeculativeStats from SpeculativeGenerator
        """
        if stats.total_tokens < 2:
            print(f"{Colors.THINK}[Stats: {stats.total_tokens} tokens generated]{Colors.RESET}\n")
            return

        print(
            f"{Colors.THINK}[Stats: {stats.total_tokens} tokens | "
            f"{stats.avg_tokens_per_sec:.1f} tok/s avg | "
            f"{stats.avg_tokens_per_sec_ex_swap:.1f} tok/s ex-swap",
            end="",
        )

        if stats.min_window_tps is not None and stats.max_window_tps is not None:
            print(f" | {stats.min_window_tps:.1f}-{stats.max_window_tps:.1f} tok/s range]")

            # Decode and show token windows if tokenizer available
            if self._tokenizer is not None:
                if stats.min_window_tokens:
                    min_text = self._tokenizer.decode(stats.min_window_tokens)
                    print(f"  Min window: {repr(min_text)}")
                if stats.max_window_tokens:
                    max_text = self._tokenizer.decode(stats.max_window_tokens)
                    print(f"  Max window: {repr(max_text)}", end="")
        else:
            print("]", end="")

        # Display speculative decoding stats if available
        if spec_stats is not None and spec_stats.iterations > 0:
            print(
                f"\n[Spec: {spec_stats.acceptance_rate:.0%} accept rate | "
                f"{spec_stats.tokens_per_iteration:.1f} tok/iter | "
                f"{spec_stats.draft_tokens_accepted}/{spec_stats.draft_tokens_proposed} accepted]",
                end="",
            )

        print(f"{Colors.RESET}\n")
