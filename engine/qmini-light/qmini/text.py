"""Safe text handling for Q-Mini's reserved tokenizer/control tokens."""

# Keep this list in one place so tokenizer training and all text-ingestion
# paths agree on which strings are structural tokens.
SPECIAL_TOKENS = (
    "<pad>", "<bos>", "<eos>", "<unk>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>", "<|other|>", "<|end|>",
)
LEGACY_CHAT_MARKERS = ("[SYSTEM]\n", "[USER]\n", "[ASSISTANT]\n")


def escape_special_token_literals(text: str) -> str:
    """Keep literal user/data text from becoming a structural token.

    A zero-width separator is inserted immediately after the opening '<'.
    It is visually unobtrusive but prevents the tokenizer from recognizing a
    literal token spelling as the actual reserved token ID. Call this only on
    message bodies, never on the intentional role/end markers framing a chat.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    for token in SPECIAL_TOKENS:
        escaped = token.replace("<", "<\u200b", 1)
        text = text.replace(token, escaped)
    # The legacy parser recognizes line-anchored square-bracket role markers.
    # Escape those spellings as well so fallback chat mode has the same trust
    # boundary as the dedicated-token path.
    for marker in LEGACY_CHAT_MARKERS:
        escaped = marker.replace("[", "[\u200b", 1)
        text = text.replace(marker, escaped)
    return text
