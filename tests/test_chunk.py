"""Tests for long-form TTS text chunking."""

from fish_studio.textnorm.chunk import split_synthesis_chunks


def test_short_text_stays_one_chunk() -> None:
    text = "Коротке речення."
    assert split_synthesis_chunks(text, 200) == [text]


def test_zero_max_chars_disables_splitting() -> None:
    text = "Перше. Друге. Третє."
    assert split_synthesis_chunks(text, 0) == [text]


def test_sentences_are_packed_until_the_limit() -> None:
    chunks = split_synthesis_chunks("Перше речення. Друге речення. Третє.", 32)
    assert chunks == ["Перше речення. Друге речення.", "Третє."]


def test_long_sentence_splits_on_commas() -> None:
    text = (
        "Довге речення з комами, ще один фрагмент після коми, "
        "і хвіст який теж треба відрізати окремо."
    )
    chunks = split_synthesis_chunks(text, 40)
    assert all(len(chunk) <= 40 for chunk in chunks)
    assert "".join(chunk.replace(" ", "") for chunk in chunks) == text.replace(" ", "")
    assert len(chunks) >= 3
