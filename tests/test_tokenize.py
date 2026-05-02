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


@pytest.mark.xfail(
    reason="rfind-based span lookup can't distinguish a user echo of an "
    "earlier assistant turn from the assistant turn itself when both "
    "occurrences sit before the next assistant turn."
)
def test_user_quotes_previous_assistant(tokenizer):
    ans1 = "alpha bravo charlie delta echo"
    ans2 = "Yes, confirmed."
    convo = [
        {"role": "user", "content": "Pick five NATO words."},
        {"role": "assistant", "content": ans1},
        {"role": "user", "content": f"Earlier you said '{ans1}'. Right?"},
        {"role": "assistant", "content": ans2},
    ]
    cfg = DataConfig(conversation_column="conversation")
    result = tokenize(_make_batch([convo]), args=cfg, tokenizer=tokenizer)
    labels = result["labels"][0]

    rendered = tokenizer.apply_chat_template(convo, tokenize=False)
    encodings = tokenizer(rendered, add_special_tokens=False)

    asst1_char = rendered.find(ans1)
    user_quote_char = rendered.find(ans1, asst1_char + 1)
    ans2_char = rendered.find(ans2)
    assert asst1_char >= 0 and user_quote_char > asst1_char and ans2_char > user_quote_char

    asst1_token = encodings.char_to_token(asst1_char)
    user_quote_token = encodings.char_to_token(user_quote_char)
    ans2_token = encodings.char_to_token(ans2_char)

    assert labels[asst1_token] != -100
    assert labels[user_quote_token] == -100
    assert labels[ans2_token] != -100


def test_identical_assistant_turns(tokenizer):
    repeated = "alpha bravo charlie delta echo"
    convo = [
        {"role": "user", "content": "Say it."},
        {"role": "assistant", "content": repeated},
        {"role": "user", "content": "Say it again."},
        {"role": "assistant", "content": repeated},
    ]
    cfg = DataConfig(conversation_column="conversation")
    result = tokenize(_make_batch([convo]), args=cfg, tokenizer=tokenizer)
    labels = result["labels"][0]

    rendered = tokenizer.apply_chat_template(convo, tokenize=False)
    encodings = tokenizer(rendered, add_special_tokens=False)

    first_char = rendered.find(repeated)
    second_char = rendered.find(repeated, first_char + 1)
    assert first_char >= 0 and second_char > first_char

    first_token = encodings.char_to_token(first_char)
    second_token = encodings.char_to_token(second_char)

    assert labels[first_token] != -100
    assert labels[second_token] != -100

    gap = [
        i for i in range(first_token, second_token) if labels[i] == -100
    ]
    assert gap, "Expected -100 region between the two identical assistant turns"
