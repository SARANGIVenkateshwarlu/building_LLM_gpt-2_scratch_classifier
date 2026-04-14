import streamlit as st
import torch
import torch.nn as nn
from pathlib import Path
import gc

# === CONFIG ===
MODEL_PATH = Path("./models/final_best_model_state_dict.pt")
CONFIG_PATH = Path("./models/final_model_bundle.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === GPT MODEL ARCHITECTURE (from notebook) ===
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        b, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# === MODEL LOADER ===
@st.cache_resource
def load_model():
    st.info("🔄 Loading GPT-2 Spam/Ham Classifier...")

    # Load config from bundle
    bundle = torch.load(CONFIG_PATH, map_location="cpu", weights_only=False)
    config = bundle["config"]
    
    # Create model
    model = GPTModel(config)
    
    # Load fine-tuned weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    st.success("✅ GPT-2 Spam Classifier loaded!")
    return model, config

# === CLASSIFY REVIEW FUNCTION (from notebook) ===
def classify_review(text, model, max_length=256, pad_token_id=50256):
    model.eval()
    
    # Simple tokenizer - in production use tiktoken
    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")
        input_ids = tokenizer.encode(text)
    except:
        # Fallback: use character-level encoding
        st.error("tiktoken not installed. Please install it with: pip install tiktoken")
        return "error", 0.0, 0.0
    
    supported_context_length = model.pos_emb.weight.shape[0]
    
    # Truncate sequences if they are too long
    input_ids = input_ids[:min(max_length, supported_context_length)]
    
    # Pad sequences to the max length
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)
    
    # Model inference without gradient tracking
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    
    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)
    predicted_label = torch.argmax(probs, dim=-1).item()
    spam_prob = probs[0][1].item() if probs.shape[1] > 1 else 0.0
    ham_prob = probs[0][0].item() if probs.shape[1] > 0 else 0.0
    
    # Return the classified result
    return ("spam" if predicted_label == 1 else "ham"), spam_prob, ham_prob

# === MAIN APP ===
st.title("🛡️ GPT-2 Spam/Ham Classifier")
st.markdown("**Your fine-tuned GPT-2 model** running locally!")

# Load model ONCE
try:
    model, config = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.info("""
    **Quick Fix:**
    1. Ensure `models/final_best_model_state_dict.pt` exists
    2. Ensure `models/final_model_bundle.pt` exists
    3. Install dependencies: `pip install torch tiktoken streamlit`
    """)
    st.stop()

# === SPAM/HAM TEST EXAMPLES ===
st.sidebar.header("📱 Test Messages")
examples = {
    "💌 Spam Examples": [
        "WINNER!! You've won £1000 cash prize! Call 0906 171 2002 NOW!",
        "URGENT! Your PayPal account suspended. Verify NOW: hxxp://fake-link.com",
        "FREE iPhone 15! Limited offer - claim yours: KL892 valid 24hrs only!",
        "Hot girls waiting 4u! Text HOT to 69666. 18+ only",
        "Your mobile #xxxxxxxx is a winner! Reply WIN to 83355 for prize"
    ],
    "✅ Ham Examples": [
        "Hey, can you make it to the 3pm meeting today?",
        "Thanks for the report. I'll review it tomorrow morning.",
        "Pizza night at my place? 7pm works for everyone?",
        "Just confirming our dentist appointment is at 2:30pm Friday",
        "Team lunch at Mario's Italian this Friday? Let me know!"
    ]
}

# Example selector
example_type = st.sidebar.radio("Select type:", list(examples.keys()))
selected_example = st.sidebar.selectbox("Pick example:", examples[example_type])

# Main input
col1, col2 = st.columns([3, 1])
with col1:
    text = st.text_area(
        "📝 Enter text to classify:",
        value=selected_example,
        height=120,
        placeholder="Paste SMS/email/SMS here..."
    )
with col2:
    confidence_threshold = st.slider("Confidence threshold", 0.5, 0.95, 0.7)

# === CLASSIFY BUTTON ===
if st.button("🔍 **Classify Text**", type="primary", use_container_width=True) and text.strip():
    with st.spinner("🤖 GPT-2 classifying..."):
        prediction, spam_prob, ham_prob = classify_review(text, model, max_length=config["context_length"])

        # Results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", "🚨 **SPAM**" if prediction == "spam" else "✅ **HAM**")
        with col2:
            st.metric("Spam", f"{spam_prob:.1%}")
        with col3:
            st.metric("Ham", f"{ham_prob:.1%}")

        # Color-coded verdict
        confidence = max(spam_prob, ham_prob)
        if confidence >= confidence_threshold:
            if prediction == "spam":
                st.error(f"🚨 **SPAM DETECTED** ({confidence:.1%} confidence)")
            else:
                st.success(f"✅ **HAM** ({confidence:.1%} confidence)")
        else:
            st.warning(f"⚠️ **Uncertain** ({confidence:.1%} confidence)")

        # Debug info
        with st.expander("🔍 Debug: Raw outputs"):
            st.json({
                "prediction": prediction,
                "spam_prob": f"{spam_prob:.4f}",
                "ham_prob": f"{ham_prob:.4f}",
                "confidence": f"{confidence:.4f}"
            })

# === BATCH TEST ===
st.markdown("---")
with st.container():
    st.subheader("🧪 Batch Test")
    batch_text = st.text_area(
        "Test multiple texts (one per line):",
        value="WINNER!! £1000 prize!\nMeeting at 3pm?\nFree iPhone giveaway!",
        height=150
    )

    if st.button("🚀 Test Batch", type="secondary"):
        texts = [t.strip() for t in batch_text.split("\n") if t.strip()]
        results = []

        progress = st.progress(0)
        max_length = config["context_length"]
        for i, text in enumerate(texts):
            pred, spam_prob, ham_prob = classify_review(text, model, max_length=max_length)
            results.append({
                "text": text[:50] + "..." if len(text) > 50 else text,
                "prediction": pred.upper(),
                "spam_prob": spam_prob,
                "ham_prob": ham_prob
            })
            progress.progress((i + 1) / len(texts))

        # Results table
        st.dataframe(results, use_container_width=True)

# === PERF COMPARISON ===
with st.expander("⚡ Base GPT-2 vs Fine-tuned"):
    st.info("Load base GPT-2 + your model → side-by-side spam detection comparison")

# Footer
st.markdown("---")
st.markdown(
    """
    💾 **Powered by your GPT-2 fine-tuned weights**
    🎯 Optimized for VS Code + Streamlit | GPU accelerated
    """
)

# Clean VRAM
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()