# Transformer Machine Translation Demo

A production-quality Streamlit application that demonstrates how a **Transformer-based encoder-decoder model** performs multilingual machine translation — the original task from the landmark paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

Built for AI Demo Programmer interview presentations.

---

## Project Overview

This demo lets you translate text between five languages using a **single multilingual Transformer model** (`facebook/m2m100_418M`). No separate models per language pair — one encoder-decoder architecture handles all directions.

**Supported languages:** English, French, German, Spanish, Hindi

**Flow:** Choose source language → Choose target language → Enter text → Translate

The UI displays the original text, translated output, language pair, model name, and measured inference time.

---

## Transformer Architecture

The Transformer replaces recurrent (RNN/LSTM) and convolutional layers with **attention mechanisms** alone.

```
Input Tokens
     ↓
┌─────────────┐
│   Encoder   │  ← Self-Attention: each token attends to all others
│  (6 layers) │
└──────┬──────┘
       │ encoder output (context)
       ↓
┌─────────────┐
│   Decoder   │  ← Self-Attention + Cross-Attention to encoder
│  (6 layers) │
└──────┬──────┘
       ↓
Output Tokens (translation)
```

**Key components:**

| Component | Role |
|-----------|------|
| **Multi-Head Self-Attention** | Lets every position attend to every other position — captures long-range dependencies in parallel |
| **Cross-Attention** | Decoder queries attend to encoder keys/values — connects input to output |
| **Positional Encoding** | Injects word order since attention is permutation-invariant |
| **Feed-Forward Networks** | Applied per position after attention |
| **Layer Normalization & Residuals** | Stabilize training across deep stacks |

---

## Why Machine Translation is the Original Transformer Task

Vaswani et al. (2017) introduced the Transformer specifically for **sequence-to-sequence** problems. Machine translation was their primary benchmark:

- **Input:** A sentence in language A (e.g. English)
- **Output:** The equivalent sentence in language B (e.g. French)

Before Transformers, translation systems relied on RNN encoders/decoders, which processed tokens sequentially and struggled with long sentences. The Transformer processes the entire input in parallel and uses attention to directly model relationships between any two words — regardless of distance.

The paper achieved state-of-the-art BLEU scores on WMT 2014 English→German and English→French, demonstrating that attention alone is sufficient for high-quality translation.

This demo replicates that core idea: **encoder reads the source, decoder generates the target**, connected through cross-attention.

---

## Model Used

| Property | Value |
|----------|-------|
| **Model** | [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M) |
| **Architecture** | Encoder-decoder Transformer |
| **Parameters** | ~418 million |
| **Training** | Many-to-Many multilingual translation (M2M-100 dataset, 100 languages) |
| **Tokenizer** | SentencePiece subword tokenizer |

M2M100 is a direct descendant of the original Transformer design, extended to handle 100 languages with a single model. For this demo we use five of those languages.

---

## Installation

**Prerequisites:** Python 3.11+, pip, ~2 GB free disk space (model weights)

```bash
# Navigate to the project directory
cd transformer_translation_demo

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** PyTorch installs the CPU build by default. For GPU acceleration, install the CUDA-enabled wheel from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

> **IDE setup:** If your editor shows import warnings for `torch` or `transformers`, select the project interpreter at `.venv/bin/python`.

---

## How to Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**First run:** The model (~1.6 GB) downloads from Hugging Face Hub automatically. Allow 1–2 minutes depending on your connection. Subsequent runs load from cache.

### Quick test

1. Source: **English** → Target: **French**
2. Enter: `The Transformer architecture revolutionized natural language processing.`
3. Click **Translate**

---

## Project Structure

```
transformer_translation_demo/
├── app.py              # Streamlit UI (presentation layer)
├── translator.py       # Model loading & translation logic
├── config.py           # Constants, language map, dataclasses
├── utils.py            # Logging and formatting helpers
├── requirements.txt    # Python dependencies
├── .gitignore          # Ignores venv, caches, model weights
├── README.md           # This file
├── assets/             # Static assets (logos, icons)
└── screenshots/        # App screenshots for documentation
```

**Architecture separation:**

- `app.py` — UI only; no model code
- `translator.py` — business logic; no Streamlit imports
- `config.py` / `utils.py` — shared configuration and utilities

---

## Future Improvements

- **GPU auto-detection** — Move model to CUDA/MPS when available for faster inference
- **Batch translation** — Upload a CSV and translate multiple sentences at once
- **Attention visualization** — Heatmap showing which source tokens the decoder attends to
- **BLEU score evaluation** — Compare output against reference translations on a test set
- **Model comparison** — Side-by-side with smaller/larger M2M100 variants or NLLB
- **Translation history** — Session log of past translations with export
- **Docker deployment** — Containerized app for reproducible demo environments
- **CI/CD pipeline** — Automated linting, type checking, and smoke tests

---

## References

- Vaswani, A., et al. (2017). [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). NeurIPS.
- Fan, A., et al. (2021). [*Beyond English-Centric Multilingual Machine Translation*](https://arxiv.org/abs/2004.06700). JMLR.
- [M2M100 on Hugging Face](https://huggingface.co/facebook/m2m100_418M)

---

## License

This demo project is provided for educational and interview presentation purposes.
