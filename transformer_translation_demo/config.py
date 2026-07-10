"""Application configuration for the Transformer Machine Translation Demo."""

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME: Final[str] = "facebook/m2m100_418M"
MODEL_DISPLAY_NAME: Final[str] = "M2M100 418M (Facebook AI)"

# ---------------------------------------------------------------------------
# Supported languages — M2M100 uses ISO 639-1 codes
# ---------------------------------------------------------------------------
LANGUAGES: Final[dict[str, str]] = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Hindi": "hi",
}

# Reverse lookup: code → display name
LANGUAGE_NAMES: Final[dict[str, str]] = {v: k for k, v in LANGUAGES.items()}

# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------
MAX_INPUT_LENGTH: Final[int] = 512
MAX_NEW_TOKENS: Final[int] = 256
NUM_BEAMS: Final[int] = 5

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
APP_TITLE: Final[str] = "Transformer Machine Translation Demo"
APP_SUBTITLE: Final[str] = (
    "Demonstrating encoder-decoder Transformer translation "
    "— inspired by *Attention Is All You Need* (Vaswani et al., 2017)"
)
PAGE_ICON: Final[str] = "🌐"


@dataclass(frozen=True)
class TranslationResult:
    """Structured output returned by the translation pipeline."""

    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    source_lang_code: str
    target_lang_code: str
    model_name: str
    inference_time_ms: float
