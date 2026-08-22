"""Training dataset that matches the s2-pro inference prompt template.

The upstream ``AutoTextSemanticInstructionIterableDataset`` is a leftover from
fish-speech 1.5: it packs samples as ``Speak out the provided text.`` plus
``<|speaker:user|>``/``<|speaker:assistant|>`` role tokens and never includes a
reference-audio block. s2-pro generates from a completely different prompt --
``convert the provided text to speech reference to the following:`` followed by
the reference transcript and its VQ codes (see
``fish_speech.models.text2semantic.inference``). Fine-tuning on the old template
teaches the slow layers a prompt shape that inference never produces, which
collapses generation to silence.

This dataset rebuilds each sample in the inference template, conditioning on
another clip from the same speaker so voice cloning stays intact.
"""

from __future__ import annotations

from pathlib import Path
from random import Random

import torch
from fish_speech.content_sequence import ContentSequence, TextPart, VQPart
from fish_speech.datasets.protos.text_data_stream import read_pb_stream
from fish_speech.datasets.semantic import CODEBOOK_PAD_TOKEN_ID, split_by_rank_worker
from fish_speech.tokenizer import IM_END_TOKEN, IM_START_TOKEN, MODALITY_TOKENS
from fish_speech.utils import RankedLogger
from fish_speech.utils.braceexpand import braceexpand
from torch.utils.data import IterableDataset, get_worker_info

log = RankedLogger(__name__, rank_zero_only=True)

# Verbatim from fish_speech.models.text2semantic.inference so the fine-tuned
# model sees byte-identical framing at train and generation time.
SYSTEM_PREFIX = "convert the provided text to speech reference to the following:\n\nText:\n"
SYSTEM_SPEECH_SEPARATOR = "\n\nSpeech:\n"


class ReferenceConditionedIterableDataset(IterableDataset):
    """Yield ``{"tokens", "labels"}`` samples in the s2-pro inference template.

    The signature mirrors ``AutoTextSemanticInstructionIterableDataset`` so this
    class is a drop-in ``_target_`` replacement in ``text2semantic_finetune.yaml``.
    ``causal``, ``use_speaker``, ``interactive_prob`` and ``skip_text_prob`` only
    exist for that compatibility: the inference template has a fixed shape, so
    there is no interactive mode or speaker-tag toggle to honour.
    """

    def __init__(
        self,
        proto_files: list[str],
        tokenizer=None,
        max_length: int = 4096,
        seed: int = 42,
        num_codebooks: int | None = None,
        max_reference_frames: int = 250,
        causal: bool = True,
        use_speaker: bool | float = False,
        interactive_prob: float = 0.0,
        skip_text_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.proto_files = proto_files
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.num_codebooks = num_codebooks
        self.max_reference_frames = max_reference_frames

        self.groups: list | None = None
        self.group_weights: list[int] | None = None
        self.rng = Random(seed)

    # ------------------------------------------------------------------ loading
    def _expand_proto_files(self) -> list[Path]:
        expanded: list[Path] = []
        for filename in self.proto_files:
            for entry in braceexpand(filename):
                path = Path(entry)
                if path.is_file():
                    expanded.append(path)
                elif path.is_dir():
                    expanded.extend(path.rglob("*.proto"))
                    expanded.extend(path.rglob("*.protos"))
                else:
                    raise ValueError(f"{path} is not a file or directory")
        return sorted(expanded)

    def _init_groups(self) -> None:
        if self.groups is not None:
            return

        expanded = self._expand_proto_files()
        Random(self.seed).shuffle(expanded)
        shard_files = split_by_rank_worker(expanded)
        log.info(f"Reading {len(shard_files)} / {len(expanded)} proto files")

        groups = []
        for filename in shard_files:
            with open(filename, "rb") as handle:
                for text_data in read_pb_stream(handle):
                    # A reference has to come from a different clip by the same
                    # speaker, so single-sentence groups are unusable.
                    if len(text_data.sentences) >= 2:
                        groups.append(text_data)

        if not groups:
            raise ValueError(
                "no proto group has 2+ sentences; reference conditioning needs "
                "at least two clips per speaker"
            )

        self.groups = groups
        self.group_weights = [len(group.sentences) for group in groups]

        worker = get_worker_info()
        self.rng = Random(self.seed + (worker.id if worker else 0))
        log.info(f"Read {len(groups)} speaker groups, {sum(self.group_weights)} clips")

    # ------------------------------------------------------------------ sampling
    @staticmethod
    def _codes(sentence) -> torch.Tensor:
        return torch.tensor([book.values for book in sentence.semantics], dtype=torch.int32)

    @staticmethod
    def _text(sentence) -> str:
        # Deliberately unnormalised: inference feeds raw user text, and any
        # cleaning applied only during training would reintroduce a mismatch.
        return sentence.texts[0].strip()

    def _pick_pair(self):
        """Pick a (reference, target) clip pair from one speaker."""
        group = self.rng.choices(self.groups, weights=self.group_weights, k=1)[0]
        sentences = group.sentences

        target = self.rng.choice(sentences)
        for _ in range(8):
            reference = self.rng.choice(sentences)
            if reference is target:
                continue
            if len(reference.semantics[0].values) > self.max_reference_frames:
                continue
            if not self._text(reference) or not self._text(target):
                continue
            return reference, target
        return None

    def build_sample(self, reference, target) -> dict[str, torch.Tensor] | None:
        reference_codes = self._codes(reference)
        target_codes = self._codes(target)

        sequence = ContentSequence()

        # System turn: the reference transcript and its audio.
        sequence.append(TextPart(text=f"{IM_START_TOKEN}system\n"))
        sequence.append(TextPart(text=SYSTEM_PREFIX))
        sequence.append(TextPart(text=f"<|speaker:0|>{self._text(reference)}"))
        sequence.append(TextPart(text=SYSTEM_SPEECH_SEPARATOR))
        sequence.append(VQPart(codes=reference_codes, cal_loss=False))
        sequence.append(TextPart(text=f"{IM_END_TOKEN}\n"))

        # User turn: the text to speak.
        sequence.append(TextPart(text=f"{IM_START_TOKEN}user\n"))
        sequence.append(TextPart(text=self._text(target)))
        sequence.append(TextPart(text=f"{IM_END_TOKEN}\n"))

        # Assistant turn: the audio we train on.
        sequence.append(
            TextPart(text=f"{IM_START_TOKEN}assistant\n{MODALITY_TOKENS['voice']}")
        )
        sequence.append(VQPart(codes=target_codes, cal_loss=True))
        sequence.append(TextPart(text=f"{IM_END_TOKEN}\n", cal_loss=True))

        encoded = sequence.encode(tokenizer=self.tokenizer)
        if len(encoded.tokens) > self.max_length:
            # The collator would truncate the assistant turn away.
            return None

        num_codebooks = self.num_codebooks or target_codes.size(0)
        vq_parts = list(encoded.vq_parts)
        all_codes = torch.cat(vq_parts, dim=1)

        tokens = torch.zeros((num_codebooks + 1, len(encoded.tokens)), dtype=torch.int)
        tokens[0] = encoded.tokens
        tokens[1:, encoded.vq_mask_tokens] = all_codes

        labels = torch.full((num_codebooks + 1, len(encoded.labels)), -100, dtype=torch.int)
        labels[0, :] = encoded.labels
        # Only the assistant's codes are a prediction target; the reference block
        # is context, exactly as at inference.
        supervised = torch.cat(
            [
                torch.full((part.size(1),), bool(flag), dtype=torch.bool)
                for part, flag in zip(vq_parts, encoded.vq_require_losses)
            ]
        )
        codebook_labels = all_codes.clone().to(torch.int)
        codebook_labels[:, ~supervised] = -100
        labels[1:, encoded.vq_mask_labels] = codebook_labels
        labels[1:, -1:] = CODEBOOK_PAD_TOKEN_ID

        return {"tokens": tokens.long(), "labels": labels.long()}

    def __iter__(self):
        self._init_groups()
        while True:
            pair = self._pick_pair()
            if pair is None:
                continue
            sample = self.build_sample(*pair)
            if sample is not None:
                yield sample
