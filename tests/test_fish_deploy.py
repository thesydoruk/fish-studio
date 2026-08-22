"""Tests for vLLM deploy concurrency patching."""

from fish_studio.runtime.fish_deploy import codec_max_num_seqs, patch_deploy_concurrency


def test_codec_max_num_seqs_caps_at_six() -> None:
    assert codec_max_num_seqs(1) == 1
    assert codec_max_num_seqs(6) == 6
    assert codec_max_num_seqs(12) == 6
    assert codec_max_num_seqs(0) == 1


def test_patch_deploy_concurrency_updates_both_stages() -> None:
    deploy = {
        "stages": [
            {"stage_id": 0, "max_num_seqs": 1},
            {"stage_id": 1, "max_num_seqs": 1},
        ]
    }
    patch_deploy_concurrency(deploy, max_concurrent_requests=12)
    assert deploy["stages"][0]["max_num_seqs"] == 12
    assert deploy["stages"][1]["max_num_seqs"] == 6
