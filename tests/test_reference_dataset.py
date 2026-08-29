"""Tests for the reference-conditioned training dataset."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fish_speech")

from fish_studio.training import reference_dataset as reference_dataset_mod  # noqa: E402
from fish_studio.training.reference_dataset import (  # noqa: E402
    SYSTEM_PREFIX,
    ReferenceConditionedIterableDataset,
    capped_group_weight,
    clip_language,
    partition_sentences,
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


def test_clip_language_splits_scripts() -> None:
    assert clip_language("Hello there") == "en"
    assert clip_language("Привіт друже") == "uk"
    assert clip_language("Hello привіт") == "mixed"
    assert clip_language("...") == "other"


def test_capped_group_weight_limits_giants() -> None:
    assert capped_group_weight(44050, 10000) == 10000
    assert capped_group_weight(230, 10000) == 230
    assert capped_group_weight(44050, 0) == 44050


class _Group:
    def __init__(self, sentences: list[_Sentence]) -> None:
        self.sentences = sentences


def test_partition_sentences_indexes_once() -> None:
    sentences = [
        _sentence("Hello there", frames=2, offset=1),
        _sentence("Привіт", frames=2, offset=10),
        _sentence("...", frames=2, offset=20),
    ]
    english, ukrainian = partition_sentences(sentences, lambda item: item.texts[0])
    assert [item.texts[0] for item in english] == ["Hello there"]
    assert [item.texts[0] for item in ukrainian] == ["Привіт"]


def test_pick_pair_prefers_en_reference_to_uk_target() -> None:
    dataset = _dataset(cross_lingual_prob=1.0)
    english = [_sentence("Hello there", frames=4, offset=10 + idx) for idx in range(4)]
    ukrainian = [_sentence("Привіт друже", frames=4, offset=80 + idx) for idx in range(4)]
    dataset.groups = [_Group([*english, *ukrainian])]
    dataset.group_weights = [8]
    dataset._build_language_buckets()

    pairs = [dataset._pick_pair() for _ in range(20)]
    assert all(pair is not None for pair in pairs)
    assert all(clip_language(dataset._text(ref)) == "en" for ref, _ in pairs)
    assert all(clip_language(dataset._text(tgt)) == "uk" for _, tgt in pairs)


def test_pick_pair_does_not_rescan_language_during_sampling(monkeypatch) -> None:
    dataset = _dataset(cross_lingual_prob=1.0)
    english = [_sentence("Hello there", frames=4, offset=10 + idx) for idx in range(4)]
    ukrainian = [_sentence("Привіт друже", frames=4, offset=80 + idx) for idx in range(4)]
    dataset.groups = [_Group([*english, *ukrainian])]
    dataset.group_weights = [8]
    dataset._build_language_buckets()

    def fail(_text: str) -> str:
        raise AssertionError("language split must be cached before sampling")

    monkeypatch.setattr(reference_dataset_mod, "clip_language", fail)
    assert dataset._pick_pair() is not None


def test_pick_pair_stays_random_when_speaker_is_monolingual() -> None:
    dataset = _dataset(cross_lingual_prob=1.0)
    sentences = [_sentence(f"Привіт {idx}", frames=4, offset=20 + idx) for idx in range(6)]
    dataset.groups = [_Group(sentences)]
    dataset.group_weights = [6]
    dataset._build_language_buckets()

    pair = dataset._pick_pair()
    assert pair is not None
    reference, target = pair
    assert reference is not target
    assert clip_language(dataset._text(reference)) == "uk"
    assert clip_language(dataset._text(target)) == "uk"


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
