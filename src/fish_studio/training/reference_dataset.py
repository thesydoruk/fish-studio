"""Training dataset that matches the s2-pro inference prompt template.

Each sample is ``convert the provided text to speech reference to the following:``
plus a same-speaker reference transcript and its VQ codes — the same shape
``fish_speech.models.text2semantic.inference`` uses at generate time.
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


def clip_language(text: str) -> str:
    """Coarse script tag used to pair an EN reference with a UK target."""
    cyrillic = sum(1 for char in text if "\u0400" <= char <= "\u04FF")
    latin = sum(1 for char in text if "A" <= char <= "Z" or "a" <= char <= "z")
    if cyrillic and not latin:
        return "uk"
    if latin and not cyrillic:
        return "en"
    if cyrillic and latin:
        return "mixed"
    return "other"


def capped_group_weight(clip_count: int, max_group_weight: int) -> int:
    """Down-weight huge speaker folders without dropping their clips."""
    if max_group_weight <= 0:
        return clip_count
    return min(clip_count, max_group_weight)


def partition_sentences(sentences, text_of) -> tuple[list, list]:
    """Split a speaker group into EN / UK buckets once, at dataset load."""
    english: list = []
    ukrainian: list = []
    for item in sentences:
        language = clip_language(text_of(item))
        if language == "en":
            english.append(item)
        elif language == "uk":
            ukrainian.append(item)
    return english, ukrainian


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
        max_group_weight: int = 10000,
        cross_lingual_prob: float = 0.7,
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
        self.max_group_weight = max_group_weight
        self.cross_lingual_prob = cross_lingual_prob

        self.groups: list | None = None
        self.group_weights: list[int] | None = None
        self.group_en: list[list] | None = None
        self.group_uk: list[list] | None = None
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

        raw_counts = [len(group.sentences) for group in groups]
        self.groups = groups
        self.group_weights = [
            capped_group_weight(count, self.max_group_weight) for count in raw_counts
        ]
        self._build_language_buckets()

        worker = get_worker_info()
        self.rng = Random(self.seed + (worker.id if worker else 0))
        bilingual = sum(
            1 for en, uk in zip(self.group_en, self.group_uk, strict=True) if en and uk
        )
        log.info(
            f"Read {len(groups)} speaker groups, {sum(raw_counts)} clips, "
            f"capped weights {sum(self.group_weights)} "
            f"(max_group_weight={self.max_group_weight}, "
            f"bilingual_groups={bilingual})"
        )

    # ------------------------------------------------------------------ sampling
    @staticmethod
    def _codes(sentence) -> torch.Tensor:
        return torch.tensor([book.values for book in sentence.semantics], dtype=torch.int32)

    @staticmethod
    def _text(sentence) -> str:
        # Deliberately unnormalised: inference feeds raw user text, and any
        # cleaning applied only during training would reintroduce a mismatch.
        return sentence.texts[0].strip()

    def _build_language_buckets(self) -> None:
        """Index EN/UK clips per speaker once; sampling only draws from these lists."""
        english: list[list] = []
        ukrainian: list[list] = []
        for group in self.groups or []:
            en, uk = partition_sentences(group.sentences, self._text)
            english.append(en)
            ukrainian.append(uk)
        self.group_en = english
        self.group_uk = ukrainian

    def _usable_reference(self, reference) -> bool:
        return bool(self._text(reference)) and (
            len(reference.semantics[0].values) <= self.max_reference_frames
        )

    def _pick_cross_lingual_pair(self, group_index: int):
        """Prefer EN reference в†’ UK target when a speaker has both."""
        if self.group_en is None or self.group_uk is None:
            self._build_language_buckets()
        english = self.group_en[group_index]
        ukrainian = self.group_uk[group_index]
        if not english or not ukrainian:
            return None
        if self.rng.random() >= self.cross_lingual_prob:
            return None
        for _ in range(8):
            reference = self.rng.choice(english)
            target = self.rng.choice(ukrainian)
            if not self._usable_reference(reference) or not self._text(target):
                continue
            return reference, target
        return None

    def _pick_pair(self):
        """Pick a (reference, target) clip pair from one speaker."""
        group_index = self.rng.choices(
            range(len(self.groups)), weights=self.group_weights, k=1
        )[0]
        group = self.groups[group_index]

        cross = self._pick_cross_lingual_pair(group_index)
        if cross is not None:
            return cross

        sentences = group.sentences
        target = self.rng.choice(sentences)
        for _ in range(8):
            reference = self.rng.choice(sentences)
            if reference is target:
                continue
            if not self._usable_reference(reference) or not self._text(target):
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
