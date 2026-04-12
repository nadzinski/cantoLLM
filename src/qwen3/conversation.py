"""Conversation session management."""

from qwen3.decoder import StreamingDecoder
from qwen3.generator import TokenGenerator
from qwen3.kv_cache import KVCache
from qwen3.presenter import TerminalPresenter
from qwen3.stats import StatsCollector


class Conversation:
    """Orchestrates multi-turn conversation with KV cache persistence.

    Composes TokenGenerator, StreamingDecoder, and TerminalPresenter
    to handle the full generation pipeline while maintaining conversation
    state across turns.
    """

    def __init__(
        self,
        generator: TokenGenerator,
        decoder: StreamingDecoder,
        presenter: TerminalPresenter,
        tokenizer,
        config: dict,
        max_new_tokens: int = 2048,
    ):
        """Initialize conversation.

        Args:
            generator: TokenGenerator for producing tokens
            decoder: StreamingDecoder for converting tokens to events
            presenter: TerminalPresenter for display
            tokenizer: Qwen3Tokenizer for encoding messages
            config: Model configuration dict (needs num_transformers)
            max_new_tokens: Maximum tokens to generate per response
        """
        self.generator = generator
        self.decoder = decoder
        self.presenter = presenter
        self.tokenizer = tokenizer
        self.config = config
        self.max_new_tokens = max_new_tokens
        self.reset()

    def reset(self):
        """Reset conversation state for a fresh start."""
        self.cache = KVCache(self.config["num_transformers"])
        self.is_first_turn = True
        self.generator.reset()

    def generate_response(self, user_message: str):
        """Generate and display a response to a user message.

        Handles encoding, token generation, decoding, and presentation.
        Updates conversation state (KV cache position) for multi-turn.

        Args:
            user_message: The user's input message
        """
        # Encode input based on conversation state
        if self.is_first_turn:
            input_ids = self.tokenizer.encode(user_message)
            self.is_first_turn = False
        else:
            # Continue existing conversation
            think_suppression = ""
            if not self.tokenizer.enable_thinking:
                think_suppression = "<think>\n\n</think>\n\n"
            continuation = (
                f"<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n"
                f"<|im_start|>assistant\n{think_suppression}"
            )
            input_ids = self.tokenizer.encode(continuation, chat_wrapped=False)

        self.generator.reset_stats()

        # Build the pipeline (unified interface for both TokenGenerator and SpeculativeGenerator)
        token_stream = self.generator.generate(
            input_ids=input_ids,
            cache=self.cache,
            stop_token_ids=self.tokenizer.stop_token_ids,
            max_tokens=self.max_new_tokens,
        )

        # Wrap with stats collection
        stats = StatsCollector()
        token_stream = stats.wrap(token_stream)

        # Decode to events
        event_stream = self.decoder.wrap(token_stream)

        # Present
        self.presenter.start_response()
        for event in event_stream:
            self.presenter.handle_event(event)
        self.presenter.end_response()

        # Display stats
        generation_stats = stats.get_stats()
        spec_stats = self.generator.get_stats()
        self.presenter.print_stats(generation_stats, spec_stats)
