"""src package — clean re-exports for the GPT-2 classifier pipeline."""

from src.model import (
    FeedForward,
    GELU,
    GPTModel,
    LayerNorm,
    MultiHeadAttention,
    TransformerBlock,
)
from src.inference import (
    classify_batch,
    classify_review,
    load_model_bundle,
    load_weights_only,
)
from src.utils import (
    encode_text,
    get_gpt2_tokenizer,
    prepare_input,
)

__all__ = [
    # Model
    "GPTModel",
    "TransformerBlock",
    "MultiHeadAttention",
    "FeedForward",
    "GELU",
    "LayerNorm",
    # Inference
    "load_model_bundle",
    "load_weights_only",
    "classify_review",
    "classify_batch",
    # Utils
    "get_gpt2_tokenizer",
    "encode_text",
    "prepare_input",
]
