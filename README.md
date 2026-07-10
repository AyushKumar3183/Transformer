# Transformer Machine Translation Demo

A production-quality Streamlit application demonstrating **Transformer-based machine translation** using the paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

Built for AI Demo Programmer interview presentations.

## What's Inside

This repository contains a multilingual translation demo powered by a single Transformer model:

- **Model:** [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M)
- **Languages:** English, French, German, Spanish, Hindi
- **Stack:** Python, Streamlit, Hugging Face Transformers, PyTorch

## Quick Start

```bash
cd transformer_translation_demo

python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **First run:** The model (~1.6 GB) downloads automatically from Hugging Face. Allow a few minutes on the first translation.

## Project Structure

```
Transformer/
├── README.md                          # This file
└── transformer_translation_demo/
    ├── app.py                         # Streamlit UI
    ├── translator.py                  # Translation pipeline
    ├── config.py                      # Configuration & constants
    ├── utils.py                       # Logging & helpers
    ├── requirements.txt
    ├── .gitignore
    ├── assets/
    └── screenshots/
```

## Documentation

See the full project documentation in [`transformer_translation_demo/README.md`](transformer_translation_demo/README.md) for:

- Transformer architecture overview
- Why machine translation is the original Transformer task
- Model details
- Installation & usage
- Future improvements

## Demo Flow

```
Source Language  →  Target Language  →  Enter Text  →  Translate
```

The app displays the original text, translated output, language pair, model name, and inference time.

## References

- Vaswani, A., et al. (2017). [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
- Fan, A., et al. (2021). [*Beyond English-Centric Multilingual Machine Translation*](https://arxiv.org/abs/2004.06700)
- [M2M100 on Hugging Face](https://huggingface.co/facebook/m2m100_418M)
