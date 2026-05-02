"""Tests for conversation tokenization with label masking."""

import pytest
from transformers import AutoTokenizer

from bergson.config import DataConfig
from bergson.data import tokenize


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")


def _make_batch(convos):
    """Wrap a list of conversations into a batch dict."""
    return {"conversation": convos}


def test_single_turn_labels(tokenizer):
    """Assistant response in a single-turn conversation gets labels."""
    batch = _make_batch(
        [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        ]
    )
    cfg = DataConfig(conversation_column="conversation")
    result = tokenize(batch, args=cfg, tokenizer=tokenizer)
    labels = result["labels"][0]
    # Some labels should be active (not -100)
    assert any(l != -100 for l in labels)


def test_multi_turn_labels(tokenizer):
    """All assistant turns get labels in a multi-turn conversation."""
    batch = _make_batch(
        [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm great, thanks!"},
            ]
        ]
    )
    cfg = DataConfig(conversation_column="conversation")
    result = tokenize(batch, args=cfg, tokenizer=tokenizer)
    labels = result["labels"][0]
    active = [i for i, l in enumerate(labels) if l != -100]
    # Should have two non-contiguous active regions (one per assistant turn)
    assert len(active) > 2
    # Check there's a gap (labels for user content are -100)
    gaps = [active[i + 1] - active[i] for i in range(len(active) - 1)]
    assert any(g > 1 for g in gaps), "Expected gap between assistant turns"


def test_truncation_preserves_partial_span(tokenizer):
    """When truncation cuts through an assistant response, labels should be
    assigned up to the truncation boundary — not dropped entirely."""
    # Create a conversation where the assistant response is long enough to
    # extend past a short max_length
    short_answer = "Short."
    long_answer = "word " * 200  # ~200 tokens, will be truncated
    batch = _make_batch(
        [
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": short_answer},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": long_answer},
            ]
        ]
    )
    cfg = DataConfig(conversation_column="conversation", truncation=True)
    max_length = 64
    result = tokenize(batch, args=cfg, tokenizer=tokenizer, max_length=max_length)

    labels = result["labels"][0]
    tokens = result["input_ids"][0]
    assert len(labels) == len(tokens) == max_length

    # The second assistant response should have SOME active labels even though
    # it's truncated — this was the bug: they were all set to -100
    active = sum(1 for l in labels if l != -100)
    assert active > 0, (
        "Truncated assistant span should still have active labels "
        "up to the truncation boundary"
    )


def test_fully_truncated_span_skipped(tokenizer):
    """When an entire assistant span is past the truncation boundary, it should
    be skipped without error."""
    # First turn is very long, second turn is entirely truncated
    long_answer = "word " * 500
    batch = _make_batch(
        [
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": long_answer},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "This will be truncated away"},
            ]
        ]
    )
    cfg = DataConfig(conversation_column="conversation", truncation=True)
    max_length = 64
    result = tokenize(batch, args=cfg, tokenizer=tokenizer, max_length=max_length)

    labels = result["labels"][0]
    assert len(labels) == max_length
    # Should not raise, and should still have some labels from the first turn


def test_assistant_content_repeated_in_later_turn(tokenizer):
    earlier = "bin boot dev etc home lib"
    later = "~ % " + earlier
    batch = _make_batch(
        [
            [
                {"role": "user", "content": "`ls`"},
                {"role": "assistant", "content": earlier},
                {"role": "user", "content": "`mkdir x`\n`ls`"},
                {"role": "assistant", "content": later},
            ]
        ]
    )
    cfg = DataConfig(conversation_column="conversation")
    result = tokenize(batch, args=cfg, tokenizer=tokenizer)

    labels = result["labels"][0]
    tokens = result["input_ids"][0]
    rendered = tokenizer.decode(tokens)

    earlier_start = rendered.find(earlier)
    later_start = rendered.find(later)
    assert earlier_start >= 0 and later_start > earlier_start

    earlier_token = next(
        i for i, t in enumerate(tokens) if labels[i] == t and labels[i] != -100
    )
    last_active = max(i for i, l in enumerate(labels) if l != -100)
    assert last_active > earlier_token + len(tokenizer.encode(earlier))
