"""Drop speech segments that overlap any non-speech sound event.

Voice-dataset policy is fixed: if audio-intel marked any non-speech event on a
clip (music, noise, applause, …), that clip is trash. No thresholds, no label
allowlists — enable ``AUDIO_INTEL_SOUND_EVENTS`` / server PANNs and keep
``SOUND_EVENTS_EXCLUDE_SPEECH`` so the timeline is already non-speech-only.
"""

from __future__ import annotations

from fish_studio.dataset.stats import SegmentStats
from fish_studio.dataset.transcript import SoundEvent, TranscriptSegment


def _interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def overlaps_any_sound_event(segment: TranscriptSegment, events: list[SoundEvent]) -> bool:
    """True when the speech interval overlaps any sound-event interval at all."""
    for event in events:
        if _interval_overlap(segment.start, segment.end, event.start, event.end) > 0.0:
            return True
    return False


def filter_junk_segments(
    segments: list[TranscriptSegment],
    sound_events: list[SoundEvent],
    *,
    stats: SegmentStats,
) -> list[TranscriptSegment]:
    """Reject every speech segment that overlaps any sound event."""
    if not sound_events:
        return segments

    kept: list[TranscriptSegment] = []
    for segment in segments:
        if overlaps_any_sound_event(segment, sound_events):
            stats.rejected_junk += 1
            continue
        kept.append(segment)
    return kept
