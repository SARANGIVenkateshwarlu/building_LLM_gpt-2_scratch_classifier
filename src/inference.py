"""Model loading and inference pipeline."""

from pathlib import Path
from typing import Any

import torch

from src.model import GPTModel
from src.utils import prepare_input


def load_model_bundle(
    config_path: str | Path,
    device: str = "cpu",
) -> tuple[GPTModel, dict]:
    """Load the full model bundle (config + weights) from disk.

    Args:
        config_path: Path to the ``final_model_bundle.pt`` file.
        device: Target device.

    Returns:
        Tuple of ``(model, config_dict)``.
    """
    bundle = torch.load(config_path, map_location=device, weights_only=False)
    config = bundle["config"]

    model = GPTModel(config)
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


def load_weights_only(
    model_path: str | Path,
    model: GPTModel,
    device: str = "cpu",
) -> GPTModel:
    """Load a plain state_dict into an existing model.

    Args:
        model_path: Path to ``final_best_model_state_dict.pt``.
        model: An already-constructed GPTModel.
        device: Target device.

    Returns:
        The model with weights loaded.
    """
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def classify_review(
    text: str,
    model: GPTModel,
    device: str = "cpu",
    max_length: int | None = None,
    pad_token_id: int = 50256,
    tokenizer=None,
) -> tuple[str, float, float]:
    """Classify a single text as spam or ham.

    Args:
        text: Raw input string.
        model: Loaded GPTModel in eval mode.
        device: Target device.
        max_length: Maximum sequence length (defaults to model context_length).
        pad_token_id: Token ID used for padding.
        tokenizer: Optional pre-loaded tokenizer.

    Returns:
        Tuple of ``(label, spam_prob, ham_prob)`` where *label* is ``"spam"``
        or ``"ham"``.
    """
    model.eval()

    input_tensor = prepare_input(
        text=text,
        model=model,
        max_length=max_length,
        pad_token_id=pad_token_id,
        device=device,
        tokenizer=tokenizer,
    )

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]

    probs = torch.softmax(logits, dim=-1)
    predicted_label = torch.argmax(probs, dim=-1).item()

    spam_prob = probs[0][1].item() if probs.shape[1] > 1 else 0.0
    ham_prob = probs[0][0].item() if probs.shape[1] > 0 else 0.0

    return ("spam" if predicted_label == 1 else "ham"), spam_prob, ham_prob


def classify_batch(
    texts: list[str],
    model: GPTModel,
    device: str = "cpu",
    max_length: int | None = None,
    pad_token_id: int = 50256,
    tokenizer=None,
    progress_callback=None,
) -> list[dict[str, Any]]:
    """Classify multiple texts and return structured results.

    Args:
        texts: List of raw input strings.
        model: Loaded GPTModel in eval mode.
        device: Target device.
        max_length: Maximum sequence length.
        pad_token_id: Token ID used for padding.
        tokenizer: Optional pre-loaded tokenizer.
        progress_callback: Optional callable receiving ``(current, total)``.

    Returns:
        List of dicts with keys ``text``, ``prediction``, ``spam_prob``,
        ``ham_prob``.
    """
    results = []
    total = len(texts)

    for i, text in enumerate(texts):
        label, spam_prob, ham_prob = classify_review(
            text, model, device, max_length, pad_token_id, tokenizer
        )
        results.append({
            "text": text[:50] + "..." if len(text) > 50 else text,
            "prediction": label.upper(),
            "spam_prob": spam_prob,
            "ham_prob": ham_prob,
        })
        if progress_callback is not None:
            progress_callback(i + 1, total)

    return results
