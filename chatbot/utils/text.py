"""
Text helpers for outbound messages.

The important one is split_message: WhatsApp rejects a text body over 4096
characters outright, so a long product list used to fail to send with no
fallback. We now split it, preferring paragraph then line then sentence
boundaries so a split doesn't land mid-word.
"""
import re
from typing import List

URL_PATTERN = re.compile(r"https?://\S+")

# Longest-first so we try the nicest break before falling back to a rough one.
_SEPARATORS = ("\n\n", "\n", ". ", " ")


def split_message(text: str, limit: int = 4000) -> List[str]:
    """Break `text` into chunks of at most `limit` characters.

    Splits on the most natural boundary available within the limit. A single
    unbroken run longer than the limit (a giant URL, say) is cut hard rather
    than dropped.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    remaining = text

    while len(remaining) > limit:
        window = remaining[:limit]
        cut = -1
        for separator in _SEPARATORS:
            found = window.rfind(separator)
            # Ignore breaks so early that we'd emit a tiny chunk and loop for ages.
            if found > limit * 0.4:
                cut = found + len(separator)
                break
        if cut <= 0:
            cut = limit  # nothing to break on - hard cut

        chunk = remaining[:cut].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].lstrip()

    if remaining.strip():
        chunks.append(remaining.strip())

    return chunks


def remove_urls(text: str) -> str:
    """Strip URLs, used when images are sent as attachments instead."""
    return re.sub(r"\s+", " ", URL_PATTERN.sub("", text or "")).strip()


def truncate(text: str, limit: int, suffix: str = "...") -> str:
    """Shorten for logs and DB columns without throwing on None."""
    text = "" if text is None else str(text)
    if len(text) <= limit:
        return text
    if limit <= len(suffix):
        return text[:limit]
    return text[: limit - len(suffix)] + suffix


def redact(text: str, keep: int = 4) -> str:
    """Show only the tail of a secret, for safe logging of tokens."""
    text = "" if text is None else str(text)
    if len(text) <= keep:
        return "*" * len(text)
    return "*" * (len(text) - keep) + text[-keep:]
