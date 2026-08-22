from pathlib import Path

from fish_studio.runtime.vllm_health import (
    check_http,
    check_stages,
    models_ready,
    running_stage_ids,
    stage_ids_from_text,
)


def test_models_ready_requires_an_id() -> None:
    assert models_ready({"data": [{"id": "/data/training/fish/vllm"}]})
    assert not models_ready({"data": []})
    assert not models_ready({"data": [{}]})
    assert not models_ready([])


def test_stage_ids_from_process_name() -> None:
    assert stage_ids_from_text("VLLM::StageEngineCoreProc_stage0_replica0_DP0") == {0}
    assert stage_ids_from_text("VLLM::StageEngineCoreProc_stage1_replica0_DP0") == {1}
    assert stage_ids_from_text("vllm serve --omni") == set()


def test_running_stage_ids_reads_proc(tmp_path: Path) -> None:
    first = tmp_path / "378"
    first.mkdir()
    (first / "comm").write_bytes(b"VLLM::StageEngineCoreProc_stage0_replica0_DP0\n")
    (first / "cmdline").write_bytes(b"VLLM::StageEngineCoreProc_stage0_replica0_DP0\x00")
    second = tmp_path / "776"
    second.mkdir()
    (second / "comm").write_bytes(b"VLLM::StageEngineCoreProc_stage1_replica0_DP0\n")
    (second / "cmdline").write_bytes(b"VLLM::StageEngineCoreProc_stage1_replica0_DP0\x00")
    assert running_stage_ids(str(tmp_path)) == {0, 1}


def test_check_stages_reports_missing(monkeypatch) -> None:
    monkeypatch.setattr("fish_studio.runtime.vllm_health.running_stage_ids", lambda: {0})
    result = check_stages()
    assert result["ok"] is False
    assert result["reason"] == "missing stage processes: [1]"


def test_check_http_reports_unreachable(monkeypatch) -> None:
    def boom(_url: str, timeout: float = 5.0) -> tuple[int, bytes]:
        del timeout
        raise ConnectionError("Connection refused")

    monkeypatch.setattr("fish_studio.runtime.vllm_health.fetch", boom)
    result = check_http("http://127.0.0.1:8091")
    assert result["ok"] is False
    assert "unreachable" in result["reason"]
