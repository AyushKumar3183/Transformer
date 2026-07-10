"""Shared utilities: logging setup and formatting helpers."""

import html
import logging
import sys
from typing import Final

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the application root logger."""
    logger = logging.getLogger("transformer_translation_demo")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


def format_inference_time(seconds: float) -> str:
    """Format inference duration for display (milliseconds with 1 decimal)."""
    return f"{seconds * 1000:.1f} ms"


def escape_html(text: str) -> str:
    """Escape user-provided text before embedding in HTML blocks."""
    return html.escape(text)


def truncate_text(text: str, max_chars: int = 120) -> str:
    """Truncate long text for log messages."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
