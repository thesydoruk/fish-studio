"""Tests for the reference-conditioned training dataset."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fish_speech")

from fish_studio.training.reference_dataset import (  # noqa: E402
    SYSTEM_PREFIX,
    ReferenceConditionedIterableDataset,
)

SEMANTIC_BEGIN_ID = 151678


class _StubTokenizer:
    """Encodes text as one token per character, which keeps offsets checkable."""

    semantic_begin_id = SEMANTIC_BEGIN_ID

    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encoded_texts.append(text)
        return [ord(char) % 1000 for char in text]


class _Semantic:
    def __init__(self, values: list[int]) -> None:
        self.values = values


class _Sentence:
    def __init__(self, text: str, codes: list[list[int]]) -> None:
        self.texts = [text]
        self.semantics = [_Semantic(values) for values in codes]


def _sentence(text: str, frames: int, offset: int = 0, num_codebooks: int = 4) -> _Sentence:
    codes = [[offset + book + frame for frame in range(frames)] for book in range(num_codebooks)]
    return _Sentence(text, codes)


def _dataset(**kwargs) -> ReferenceConditionedIterableDataset:
    return ReferenceConditionedIterableDataset(
        proto_files=[], tokenizer=_StubTokenizer(), **kwargs
    )


def test_prompt_uses_the_inference_template() -> None:
    dataset = _dataset()
    reference = _sentence("референс", frames=5, offset=100)
    target = _sentence("ціль", frames=7, offset=500)

    dataset.build_sample(reference, target)

    joined = "".join(dataset.tokenizer.encoded_texts)
    assert SYSTEM_PREFIX in joined
    assert "<|im_start|>system\n" in joined
    assert "<|speaker:0|>референс" in joined
    assert "\n\nSpeech:\n" in joined
    assert "<|im_start|>user\nціль" in joined
    assert "<|im_start|>assistant\n<|voice|>" in joined
    # The legacy fish-speech 1.5 framing must be gone.
    assert "Speak out the provided text." not in joined
    assert "<|speaker:user|>" not in joined
    assert "<|speaker:assistant|>" not in joined


def test_only_the_target_codes_are_supervised() -> None:
    dataset = _dataset()
    reference = _sentence("референс", frames=5, offset=100)
    target = _sentence("ціль", frames=7, offset=500)

    sample = dataset.build_sample(reference, target)

    codebook_labels = sample["labels"][1:]
    supervised = codebook_labels[:, (codebook_labels != -100).any(dim=0)]
    # Seven target frames are supervised; the last column is the pad the model
    # emits after the final code, so the reference's five frames stay masked.
    assert supervised.size(1) == len(target.semantics[0].values) + 1
    assert (supervised[:, :-1] >= 500).all()
    assert (supervised[:, -1:] == 0).all()


def test_reference_codes_are_present_as_context() -> None:
    dataset = _dataset()
    reference = _sentence("референс", frames=5, offset=100)
    target = _sentence("ціль", frames=7, offset=500)

    sample = dataset.build_sample(reference, target)

    codebook_inputs = sample["tokens"][1:]
    filled = codebook_inputs[:, (codebook_inputs != 0).any(dim=0)]
    # Both blocks feed the model as input even though only one is supervised.
    assert filled.size(1) == 5 + 7


def test_oversized_sample_is_rejected() -> None:
    dataset = _dataset(max_length=32)

    sample = dataset.build_sample(
        _sentence("референс", frames=5, offset=100),
        _sentence("ціль", frames=7, offset=500),
    )

    assert sample is None


def test_base_labels_ignore_the_prompt() -> None:
    dataset = _dataset()

    sample = dataset.build_sample(
        _sentence("референс", frames=5, offset=100),
        _sentence("ціль", frames=7, offset=500),
    )

    base_labels = sample["labels"][0]
    supervised = base_labels != -100
    # Supervision starts only inside the assistant turn, so the first token of
    # the system prompt is never a target.
    assert not supervised[0]
    assert supervised.sum() > 0
