"""Streamlit UI for the Transformer Machine Translation Demo."""

import streamlit as st

from config import APP_SUBTITLE, APP_TITLE, LANGUAGES, PAGE_ICON
from translator import TranslationError, is_model_loaded, load_model_and_tokenizer, translate
from utils import escape_html, format_inference_time, setup_logging

logger = setup_logging()

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def _cache_model() -> str:
    """Streamlit resource cache — ensures the model loads only once per server process."""
    load_model_and_tokenizer()
    return "loaded"


# ---------------------------------------------------------------------------
# Custom CSS — professional corporate theme (white + blue)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Base ─────────────────────────────────────────────────────────── */
    .stApp {
        background-color: #ffffff;
    }
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Typography ───────────────────────────────────────────────────── */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0d47a1;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #546e7a;
        margin-bottom: 2rem;
        line-height: 1.5;
    }

    /* ── Cards ────────────────────────────────────────────────────────── */
  .card {
        background: #ffffff;
        border: 1px solid #e3eaf2;
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(13, 71, 161, 0.06);
    }
    .card-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #1565c0;
        margin-bottom: 0.5rem;
    }
    .card-value {
        font-size: 1.15rem;
        color: #263238;
        line-height: 1.6;
        word-break: break-word;
    }
    .card-value-large {
        font-size: 1.35rem;
        font-weight: 500;
        color: #0d47a1;
        line-height: 1.6;
        word-break: break-word;
    }

    /* ── Meta badges ──────────────────────────────────────────────────── */
    .meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1rem;
    }
    .meta-badge {
        background: #e3f2fd;
        color: #1565c0;
        border-radius: 20px;
        padding: 0.35rem 0.9rem;
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* ── Pipeline diagram ───────────────────────────────────────────── */
    .pipeline-step {
        background: #f5f9ff;
        border: 1px solid #bbdefb;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        text-align: center;
        font-weight: 600;
        color: #1565c0;
        margin: 0.25rem 0;
    }
    .pipeline-arrow {
        text-align: center;
        color: #90caf9;
        font-size: 1.4rem;
        line-height: 1.2;
    }
    .pipeline-desc {
        font-size: 0.85rem;
        color: #607d8b;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* ── Buttons ──────────────────────────────────────────────────────── */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #1565c0;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 2rem;
        transition: background 0.2s;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #0d47a1;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        border: 1px solid #90caf9;
        color: #1565c0;
        border-radius: 8px;
        font-weight: 600;
    }

    /* ── Select / text area ───────────────────────────────────────────── */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextArea"] label {
        font-weight: 600 !important;
        color: #37474f !important;
    }

    /* ── Expander ─────────────────────────────────────────────────────── */
    div[data-testid="stExpander"] {
        border: 1px solid #e3eaf2;
        border-radius: 10px;
        background: #fafcff;
    }

    /* ── Divider ──────────────────────────────────────────────────────── */
    .section-divider {
        border: none;
        border-top: 2px solid #e3f2fd;
        margin: 2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _render_header() -> None:
    st.markdown(f'<p class="main-title">{APP_TITLE}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="main-subtitle">{APP_SUBTITLE}</p>', unsafe_allow_html=True)


def _render_pipeline_explainer() -> None:
    """Beginner-friendly visual explanation of the Transformer translation pipeline."""
    with st.expander("How Transformer Translation Works", expanded=False):
        st.markdown(
            "This demo uses a **single multilingual Transformer** that translates "
            "between languages without separate models per language pair. "
            "Here is what happens when you click **Translate**:"
        )

        steps = [
            ("Input Sentence", "Your text in the source language (e.g. English)"),
            ("Tokenizer", "Splits text into subword tokens the model understands"),
            ("Encoder", "Self-attention layers read the full input and build context"),
            ("Decoder", "Cross-attention generates tokens in the target language"),
            ("Generated Translation", "Token IDs are converted back to readable text"),
        ]

        for i, (title, desc) in enumerate(steps):
            st.markdown(f'<p class="pipeline-desc">{desc}</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="pipeline-step">{title}</div>', unsafe_allow_html=True
            )
            if i < len(steps) - 1:
                st.markdown('<div class="pipeline-arrow">↓</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "**Key insight from the paper:** The encoder and decoder are connected "
            "through *multi-head attention* — the decoder can \"look at\" every part "
            "of the input sentence while generating each output word. "
            "No recurrent (RNN) or convolutional layers are needed."
        )


def _render_result_card(label: str, value: str, large: bool = False) -> None:
    css_class = "card-value-large" if large else "card-value"
    safe_value = escape_html(value)
    safe_label = escape_html(label)
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{safe_label}</div>
            <div class="{css_class}">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_translation_output(result) -> None:
    """Display translation results in styled cards."""
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### Translation Result")

    col_orig, col_trans = st.columns(2)
    with col_orig:
        _render_result_card("Original Text", result.original_text)
    with col_trans:
        _render_result_card("Translated Text", result.translated_text, large=True)

    safe_source = escape_html(result.source_language)
    safe_target = escape_html(result.target_language)
    safe_model = escape_html(result.model_name)
    inference = format_inference_time(result.inference_time_ms / 1000)
    st.markdown(
        f"""
        <div class="meta-row">
            <span class="meta-badge">Source: {safe_source}</span>
            <span class="meta-badge">Target: {safe_target}</span>
            <span class="meta-badge">Model: {safe_model}</span>
            <span class="meta-badge">Inference: {inference}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _render_header()
    _render_pipeline_explainer()

    # ── Language selection ──────────────────────────────────────────────
    st.markdown("### Configure Translation")
    lang_col1, lang_arrow, lang_col2 = st.columns([5, 1, 5])

    language_names = list(LANGUAGES.keys())

    with lang_col1:
        source_language = st.selectbox(
            "Source Language",
            options=language_names,
            index=0,
            key="source_lang",
        )
    with lang_arrow:
        st.markdown(
            '<div style="text-align:center; font-size:2rem; color:#1565c0; '
            'padding-top:1.8rem;">↓</div>',
            unsafe_allow_html=True,
        )
    with lang_col2:
        # Default target to French if source is English, else English
        default_target_idx = 1 if source_language == "English" else 0
        target_language = st.selectbox(
            "Target Language",
            options=language_names,
            index=default_target_idx,
            key="target_lang",
        )

    # ── Text input ──────────────────────────────────────────────────────
    st.markdown("### Enter Text")
    input_text = st.text_area(
        "Text to translate",
        height=180,
        placeholder="Type or paste your sentence here…",
        label_visibility="collapsed",
        key="input_text",
    )

    # ── Action buttons ──────────────────────────────────────────────────
    def _clear_form() -> None:
        """Reset input and result via callback (safe for widget-backed keys)."""
        st.session_state["input_text"] = ""
        st.session_state.pop("last_result", None)

    btn_col1, btn_col2, _ = st.columns([2, 2, 6])
    with btn_col1:
        translate_clicked = st.button("Translate", type="primary", use_container_width=True)
    with btn_col2:
        st.button(
            "Clear",
            type="secondary",
            use_container_width=True,
            on_click=_clear_form,
        )

    # ── Model status indicator ──────────────────────────────────────────
    if is_model_loaded():
        st.caption("Model loaded and ready.")
    else:
        st.caption(
            "Model will be downloaded on first translation (~1.6 GB). "
            "Please allow 1–2 minutes on first run."
        )

    # ── Translation ─────────────────────────────────────────────────────
    if translate_clicked:
        if not input_text.strip():
            st.warning("Please enter text before translating.")
        else:
            source_code = LANGUAGES[source_language]
            target_code = LANGUAGES[target_language]

            with st.spinner("Translating… (encoder → decoder pipeline running)"):
                try:
                    _cache_model()
                    result = translate(
                        text=input_text,
                        source_lang_code=source_code,
                        target_lang_code=target_code,
                        source_language=source_language,
                        target_language=target_language,
                    )
                    st.session_state["last_result"] = result
                except TranslationError as exc:
                    logger.warning("Translation error: %s", exc)
                    st.error(str(exc))
                except Exception as exc:
                    logger.exception("Unexpected error during translation")
                    st.error(f"An unexpected error occurred: {exc}")

    # ── Display last result ─────────────────────────────────────────────
    if "last_result" in st.session_state:
        _render_translation_output(st.session_state["last_result"])


if __name__ == "__main__":
    main()
