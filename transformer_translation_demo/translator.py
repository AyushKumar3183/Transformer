"""Core translation logic using the M2M100 Transformer model.

Pipeline overview (maps to Vaswani et al., 2017):
  1. Tokenizer  — converts source text into subword token IDs
  2. Encoder    — self-attention layers build contextual representations
  3. Decoder    — cross-attention attends to encoder output while generating
  4. Output     — token IDs are decoded back into human-readable text
"""

import time
from functools import lru_cache

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from config import (
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS,
    MODEL_DISPLAY_NAME,
    MODEL_NAME,
    NUM_BEAMS,
    TranslationResult,
)
from utils import setup_logging, truncate_text

logger = setup_logging()


class TranslationError(Exception):
    """Raised when translation fails due to model or input issues."""


@lru_cache(maxsize=1)
def load_model_and_tokenizer() -> tuple[M2M100ForConditionalGeneration, M2M100Tokenizer]:
    """Load the M2M100 model and tokenizer exactly once (cached in-process).

  The @lru_cache decorator ensures the ~1.6 GB weights are downloaded and
  loaded only on the first translation request, not on every call.
    """
    logger.info("Loading model '%s' — this may take a moment on first run…", MODEL_NAME)
    try:
        tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
        model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
    except OSError as exc:
        logger.error("Failed to download or load model: %s", exc)
        raise TranslationError(
            f"Could not load model '{MODEL_NAME}'. "
            "Check your internet connection and Hugging Face access."
        ) from exc

    model.eval()
    logger.info("Model loaded successfully on device: %s", _get_device(model))
    return model, tokenizer


def _get_device(model: M2M100ForConditionalGeneration) -> str:
    """Return a human-readable device string for the model."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "unknown"


def is_model_loaded() -> bool:
    """Return True if the model has already been cached in memory."""
    return load_model_and_tokenizer.cache_info().currsize > 0


def translate(
    text: str,
    source_lang_code: str,
    target_lang_code: str,
    source_language: str,
    target_language: str,
) -> TranslationResult:
    """Translate *text* from source language to target language.

    Args:
        text: Raw input sentence or paragraph.
        source_lang_code: ISO 639-1 code (e.g. ``"en"``).
        target_lang_code: ISO 639-1 code (e.g. ``"fr"``).
        source_language: Human-readable source language name.
        target_language: Human-readable target language name.

    Returns:
        A :class:`TranslationResult` with the translation and metadata.

    Raises:
        TranslationError: On empty input, identical languages, or model failure.
        ValueError: On invalid language codes.
    """
    text = text.strip()
    if not text:
        raise TranslationError("Please enter text to translate.")

    if source_lang_code == target_lang_code:
        raise TranslationError(
            "Source and target languages must be different. "
            "Please select two distinct languages."
        )

    logger.info(
        "Translating [%s → %s]: %s",
        source_lang_code,
        target_lang_code,
        truncate_text(text),
    )

    model, tokenizer = load_model_and_tokenizer()

    # Step 1 — Tokenize: convert text to subword token IDs with source-language tag
    tokenizer.src_lang = source_lang_code
    try:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )
    except Exception as exc:
        logger.error("Tokenization failed: %s", exc)
        raise TranslationError(f"Tokenization failed: {exc}") from exc

    # Move tensors to the same device as the model
    device = next(model.parameters()).device
    model_inputs = {k: v.to(device) for k, v in encoded.items()}

    # forced_bos_token_id tells the decoder which language to generate in
    forced_bos_token_id = tokenizer.get_lang_id(target_lang_code)

    # Step 2 & 3 — Encoder + Decoder: self-attention + cross-attention generation
    start = time.perf_counter()
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                early_stopping=True,
            )
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        raise TranslationError(f"Translation generation failed: {exc}") from exc

    elapsed = time.perf_counter() - start

    # Step 4 — Decode: convert token IDs back to text
    translated_text: str = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    logger.info(
        "Translation complete in %.1f ms: %s",
        elapsed * 1000,
        truncate_text(translated_text),
    )

    return TranslationResult(
        original_text=text,
        translated_text=translated_text,
        source_language=source_language,
        target_language=target_language,
        source_lang_code=source_lang_code,
        target_lang_code=target_lang_code,
        model_name=MODEL_DISPLAY_NAME,
        inference_time_ms=elapsed * 1000,
    )
