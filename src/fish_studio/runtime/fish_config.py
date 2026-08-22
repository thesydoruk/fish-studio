"""Fish Speech checkpoint path resolution for training and CLI infer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fish_studio.config import ProjectConfig, StressConfig
from fish_studio.paths import resolve_under_data_root
from fish_studio.training.layout import resolve_base_checkpoint, resolve_merged_checkpoint


@dataclass(frozen=True)
class FishSpeechRuntimeSettings:
    """Runtime Fish Speech settings for in-process CLI inference."""

    llama_checkpoint: Path
    decoder_checkpoint: Path
    decoder_config_name: str
    device: str
    default_language: str
    half: bool
    compile: bool
    chunk_length: int
    max_new_tokens: int
    default_reference_text: str
    use_finetuned: bool = False
    stress: StressConfig = field(default_factory=StressConfig)

    @classmethod
    def from_project(
        cls,
        project: ProjectConfig,
        *,
        prefer_finetuned: bool | None = None,
    ) -> FishSpeechRuntimeSettings:
        """Resolve LLAMA + codec paths. ``prefer_finetuned`` overrides ``use_finetuned``."""
        inf = project.inference
        ws = project.workspace()
        fish = project.fish_speech

        use_finetuned = fish.use_finetuned
        if prefer_finetuned is True:
            use_finetuned = True
        elif prefer_finetuned is False:
            use_finetuned = False

        if use_finetuned:
            llama = resolve_merged_checkpoint(project, ws)
            if not _has_llama_weights(llama):
                raise FileNotFoundError(
                    f"Fine-tuned Fish Speech checkpoint not found under {llama}. "
                    "Run ./run.sh train merge first or set fish_speech.use_finetuned: false."
                )
        else:
            llama = resolve_under_data_root(ws.data_root, fish.llama_checkpoint)

        if fish.decoder_checkpoint:
            decoder = resolve_under_data_root(ws.data_root, fish.decoder_checkpoint)
        else:
            # Stock s2-pro ships codec.pth next to the LLAMA weights.
            decoder = resolve_base_checkpoint(project, ws) / "codec.pth"

        return cls(
            llama_checkpoint=llama,
            decoder_checkpoint=decoder,
            decoder_config_name=fish.decoder_config_name,
            device=inf.device,
            default_language=inf.language,
            half=fish.half,
            compile=fish.compile,
            chunk_length=fish.chunk_length,
            max_new_tokens=fish.max_new_tokens if fish.max_new_tokens > 0 else 1024,
            default_reference_text=fish.default_reference_text,
            use_finetuned=use_finetuned,
            stress=project.stress,
        )


def resolve_speaker_reference(project: ProjectConfig) -> Path:
    """Default reference clip from the configured training dataset."""
    ft = project.training
    return project.workspace().dataset_dir(ft.dataset_id) / "reference.wav"


def _has_llama_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "model.pth").is_file():
        return True
    if (path / "model.safetensors").is_file():
        return True
    return (path / "model.safetensors.index.json").is_file()
