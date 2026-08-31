import pytest

from utils.text import redact, remove_urls, split_message, truncate


def test_short_message_is_not_split():
    assert split_message("hello", 100) == ["hello"]


def test_empty_message_returns_nothing():
    assert split_message("", 100) == []


def test_every_chunk_respects_the_limit():
    text = "\n".join(f"Line number {i} with some padding text" for i in range(200))
    chunks = split_message(text, 500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_split_preserves_all_words():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_message(text, 300)
    rejoined = " ".join(chunks).split()
    assert rejoined == text.split()


def test_prefers_paragraph_boundary():
    first = "A" * 300
    second = "B" * 300
    chunks = split_message(f"{first}\n\n{second}", 400)
    assert chunks[0] == first
    assert chunks[1] == second


def test_unbreakable_run_is_hard_cut_not_dropped():
    """A single giant token (a long URL) must still be delivered."""
    text = "x" * 250
    chunks = split_message(text, 100)
    assert "".join(chunks) == text
    assert all(len(c) <= 100 for c in chunks)


def test_whatsapp_limit_case():
    """The bug this fixes: a >4096 char reply used to be rejected outright."""
    long_catalog = "\n".join(f"{i}. Product {i} - Rs.{i * 100}" for i in range(1, 400))
    assert len(long_catalog) > 4096
    chunks = split_message(long_catalog, 4000)
    assert len(chunks) >= 2
    assert all(len(c) <= 4000 for c in chunks)


def test_invalid_limit_rejected():
    with pytest.raises(ValueError):
        split_message("abc", 0)


def test_remove_urls():
    assert remove_urls("See https://example.com/x now") == "See now"
    assert remove_urls(None) == ""


def test_truncate():
    assert truncate("abcdef", 10) == "abcdef"
    assert truncate("abcdefghij", 6) == "abc..."
    assert truncate(None, 5) == ""


def test_redact_keeps_only_the_tail():
    assert redact("supersecrettoken") == "************oken"
    assert redact("ab") == "**"
