import numpy as np
import pytest

from fish_studio.server.voiceprint import cosine_similarity, l2_normalize, to_mono_16k


def test_cosine_identical_is_one() -> None:
    vector = l2_normalize([1.0, 2.0, 3.0])
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_to_mono_16k_passthrough() -> None:
    audio = np.ones(800, dtype=np.float32)
    out = to_mono_16k(audio, 16_000)
    assert out.shape == (800,)
    assert float(out[0]) == 1.0
