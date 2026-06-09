"""Pluggable tokenizer interface for accurate token estimation."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Abstract tokenizer for token count estimation.

    Plugins can register custom tokenizers (e.g. tiktoken, sentencepiece)
    via PluginContext.register_tokenizer().
    """

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the given text."""

    def estimate_message_tokens(self, content: str, role: str = "user") -> int:
        """Estimate tokens for a message including overhead.

        Override for models with per-message overhead (e.g. ChatML format).
        """
        return self.estimate_tokens(content) + 4  # default overhead


class SimpleTokenizer(Tokenizer):
    """Default tokenizer using character-based estimation (len // 4).

    Reasonable approximation for English text on most models.
    """

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4


class TiktokenTokenizer(Tokenizer):
    """Tokenizer using OpenAI's tiktoken library for accurate counting.

    Falls back to SimpleTokenizer if tiktoken is not installed.
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        self._model = model
        self._encoder: Any = None
        try:
            import tiktoken
            self._encoder = tiktoken.encoding_for_model(model)
        except (ImportError, ModuleNotFoundError, KeyError):
            self._encoder = None

    def estimate_tokens(self, text: str) -> int:
        if self._encoder:
            return len(self._encoder.encode(text))
        return len(text) // 4
