"""Tokenizer and text preprocessing utilities."""

import torch


def get_gpt2_tokenizer():
    """Return a GPT-2 tokenizer (tiktoken)."""
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def encode_text(text: str, tokenizer=None) -> list[int]:
    """Encode a string into a list of token IDs."""
    if tokenizer is None:
        tokenizer = get_gpt2_tokenizer()
    return tokenizer.encode(text)


def prepare_input(
    text: str,
    model,
    max_length: int | None = None,
    pad_token_id: int = 50256,
    device: str = "cpu",
    tokenizer=None,
) -> torch.Tensor:
    """Tokenize, truncate, pad, and return a model-ready input tensor.

    Args:
        text: Raw input string.
        model: GPTModel instance (used to read context_length).
        max_length: Maximum sequence length. Defaults to model context_length.
        pad_token_id: Token ID used for padding.
        device: Target device ("cpu" or "cuda").
        tokenizer: Optional pre-loaded tokenizer.

    Returns:
        A (1, max_length) tensor ready for model forward pass.
    """
    if tokenizer is None:
        tokenizer = get_gpt2_tokenizer()

    input_ids = encode_text(text, tokenizer)

    supported_context_length = model.pos_emb.weight.shape[0]
    effective_max = max_length if max_length is not None else supported_context_length

    # Truncate if sequence exceeds model capacity
    input_ids = input_ids[: min(effective_max, supported_context_length)]

    # Pad to the desired length
    input_ids += [pad_token_id] * (effective_max - len(input_ids))

    return torch.tensor(input_ids, device=device).unsqueeze(0)
