"""Streamlit UI for the GPT-2 Spam/Ham Classifier.

All model, inference, and utility logic lives in the ``src`` package.
This file only handles the interactive interface.
"""

import gc
from pathlib import Path

import streamlit as st
import torch

from src.inference import classify_review, classify_batch, load_model_bundle, load_weights_only
from src.utils import get_gpt2_tokenizer

# === CONFIG ===
MODEL_PATH = Path("./models/final_best_model_state_dict.pt")
CONFIG_PATH = Path("./models/final_model_bundle.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# === MODEL LOADER ===
@st.cache_resource
def _load_cached_model():
    """Load model and tokenizer once, cached across Streamlit reruns."""
    model, config = load_model_bundle(CONFIG_PATH, device=DEVICE)
    model = load_weights_only(MODEL_PATH, model, device=DEVICE)
    tokenizer = get_gpt2_tokenizer()
    return model, config, tokenizer


# === MAIN APP ===
st.title("🛡️ GPT-2 Spam/Ham Classifier")
st.markdown("**Your fine-tuned GPT-2 model** running locally!")

# Load model ONCE
try:
    model, config, tokenizer = _load_cached_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.info(
        """
    **Quick Fix:**
    1. Ensure `models/final_best_model_state_dict.pt` exists
    2. Ensure `models/final_model_bundle.pt` exists
    3. Install dependencies: `pip install torch tiktoken streamlit`
    """
    )
    st.stop()

# === SPAM/HAM TEST EXAMPLES ===
st.sidebar.header("📱 Test Messages")
examples = {
    "💌 Spam Examples": [
        "WINNER!! You've won £1000 cash prize! Call 0906 171 2002 NOW!",
        "URGENT! Your PayPal account suspended. Verify NOW: hxxp://fake-link.com",
        "FREE iPhone 15! Limited offer - claim yours: KL892 valid 24hrs only!",
        "Hot girls waiting 4u! Text HOT to 69666. 18+ only",
        "Your mobile #xxxxxxxx is a winner! Reply WIN to 83355 for prize",
    ],
    "✅ Ham Examples": [
        "Hey, can you make it to the 3pm meeting today?",
        "Thanks for the report. I'll review it tomorrow morning.",
        "Pizza night at my place? 7pm works for everyone?",
        "Just confirming our dentist appointment is at 2:30pm Friday",
        "Team lunch at Mario's Italian this Friday? Let me know!",
    ],
}

example_type = st.sidebar.radio("Select type:", list(examples.keys()))
selected_example = st.sidebar.selectbox("Pick example:", examples[example_type])

col1, col2 = st.columns([3, 1])
with col1:
    text = st.text_area(
        "📝 Enter text to classify:",
        value=selected_example,
        height=120,
        placeholder="Paste SMS/email/SMS here...",
    )
with col2:
    confidence_threshold = st.slider("Confidence threshold", 0.5, 0.95, 0.7)

# === CLASSIFY BUTTON ===
if st.button("🔍 **Classify Text**", type="primary", use_container_width=True) and text.strip():
    with st.spinner("🤖 GPT-2 classifying..."):
        prediction, spam_prob, ham_prob = classify_review(
            text, model, device=DEVICE, max_length=config["context_length"], tokenizer=tokenizer
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", "🚨 **SPAM**" if prediction == "spam" else "✅ **HAM**")
        with col2:
            st.metric("Spam", f"{spam_prob:.1%}")
        with col3:
            st.metric("Ham", f"{ham_prob:.1%}")

        confidence = max(spam_prob, ham_prob)
        if confidence >= confidence_threshold:
            if prediction == "spam":
                st.error(f"🚨 **SPAM DETECTED** ({confidence:.1%} confidence)")
            else:
                st.success(f"✅ **HAM** ({confidence:.1%} confidence)")
        else:
            st.warning(f"⚠️ **Uncertain** ({confidence:.1%} confidence)")

        with st.expander("🔍 Debug: Raw outputs"):
            st.json(
                {
                    "prediction": prediction,
                    "spam_prob": f"{spam_prob:.4f}",
                    "ham_prob": f"{ham_prob:.4f}",
                    "confidence": f"{confidence:.4f}",
                }
            )

# === BATCH TEST ===
st.markdown("---")
with st.container():
    st.subheader("🧪 Batch Test")
    batch_text = st.text_area(
        "Test multiple texts (one per line):",
        value="WINNER!! £1000 prize!\nMeeting at 3pm?\nFree iPhone giveaway!",
        height=150,
    )

    if st.button("🚀 Test Batch", type="secondary"):
        texts = [t.strip() for t in batch_text.split("\n") if t.strip()]
        progress_bar = st.progress(0)

        def _update(current, total):
            progress_bar.progress(current / total)

        results = classify_batch(
            texts,
            model,
            device=DEVICE,
            max_length=config["context_length"],
            tokenizer=tokenizer,
            progress_callback=_update,
        )

        st.dataframe(results, use_container_width=True)

# === PERF COMPARISON ===
with st.expander("⚡ Base GPT-2 vs Fine-tuned"):
    st.info("Load base GPT-2 + your model → side-by-side spam detection comparison")

# Footer
st.markdown("---")
st.markdown(
    """
    💾 **Powered by your GPT-2 fine-tuned weights**
    🎯 Modular design: UI · Model · Inference · Utils
    """
)

# Clean VRAM
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
