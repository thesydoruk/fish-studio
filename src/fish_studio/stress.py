"""Ukrainian lexical stress marking, shared by dataset prep and synthesis.

Ukrainian stress is lexical: it cannot be derived from spelling, so a TTS model
has to memorise it per word form. s2-pro honours U+0301 (combining acute) in its
input text and ignores the spacing acute U+00B4, which makes explicit marks a
reliable way to fix stress on words the model never learned -- including
domain vocabulary that no audiobook corpus contains.

Marked text only sounds natural once the model has been fine-tuned on marked
transcripts, so training text and synthesis input must be marked identically.
Both paths go through this module for that reason.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fish_studio.config import StressConfig

COMBINING_ACUTE = "\u0301"
_LOGGER = logging.getLogger(__name__)

def _preserve_case(template: str, replacement: str) -> str:
    """Copy capitalisation from ``template`` onto ``replacement`` (keeps ')."""
    if template.isupper():
        return "".join(ch.upper() if ch.isalpha() else ch for ch in replacement)
    if template[:1].isupper():
        chars: list[str] = []
        uppercased = False
        for ch in replacement:
            if ch.isalpha() and not uppercased:
                chars.append(ch.upper())
                uppercased = True
            else:
                chars.append(ch)
        return "".join(chars)
    return replacement


# ASR / book OCR often drops the Ukrainian apostrophe before jotated vowels.
# Pattern group(1) is the head used for capitalisation; replacement is the
# corrected head (apostrophe included). Order: longer / more specific first.
_APOSTROPHE_FIXES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in (
        (r"\b(пам)(?=[яюєї])", "пам'"),
        (r"\b(зв)я(?=[зж])", "зв'я"),
        (r"\b(ім)(?=я)", "ім'"),
        (r"\b(кров)(?=ю)", "кров'"),
        (r"\b(м)яс", "м'яс"),
        (r"\b(в)я(?=[зжснлтд])", "в'я"),
        (r"\b(п)ят", "п'ят"),
        (r"\b(об)є", "об'є"),
        (r"\b(під)ї", "під'ї"),
        (r"\b(роз)ї", "роз'ї"),
        (r"\b(з)ї", "з'ї"),
        (r"\b(з)яв", "з'яв"),
        (r"\b(з)єд", "з'єд"),
        (r"\b(б)є", "б'є"),
        (r"\b(в)ю", "в'ю"),
    )
)

_APOSTROPHE_CHARS = {
    "\u2019",  # ’
    "\u2018",  # ‘
    "\u02bc",  # ʼ
    "\uff07",  # ＇
    "`",
}

_WORD_RE = re.compile(
    r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ'’ʼ\u0301-]+",
    re.UNICODE,
)

_cpu_stanza_patched = False


def has_stress_marks(text: str) -> bool:
    return COMBINING_ACUTE in text


def strip_stress_marks(text: str) -> str:
    return text.replace(COMBINING_ACUTE, "")


def normalize_uk_text(text: str) -> str:
    """Repair apostrophes and quote variants so dictionary lookup can match."""
    if not text:
        return text
    for ch in _APOSTROPHE_CHARS:
        text = text.replace(ch, "'")
    # Spacing acute after a letter is a mistaken stress mark; elsewhere, apostrophe.
    text = re.sub(r"(?<=[А-Яа-яІіЇїЄєҐґA-Za-z])\u00b4", COMBINING_ACUTE, text)
    text = text.replace("\u00b4", "'")
    for pattern, repl in _APOSTROPHE_FIXES:
        def _sub(match: re.Match[str], *, _repl: str = repl) -> str:
            return _preserve_case(match.group(1), _repl)

        text = pattern.sub(_sub, text)
    return text


def _casefold_key(word: str) -> str:
    return strip_stress_marks(word).replace("\u2019", "'").casefold()


def _match_case(template: str, stressed: str) -> str:
    """Copy capitalisation from ``template`` onto a stressed lexicon form."""
    plain_template = strip_stress_marks(template)
    plain_stressed = strip_stress_marks(stressed)
    if plain_template.isupper():
        out: list[str] = []
        for ch in stressed:
            out.append(ch.upper() if ch != COMBINING_ACUTE else ch)
        return "".join(out)
    if plain_template[:1].isupper() and plain_template[1:].islower():
        chars = list(stressed)
        for i, ch in enumerate(chars):
            if ch != COMBINING_ACUTE:
                chars[i] = ch.upper()
                break
        return "".join(chars)
    if plain_template == plain_stressed:
        return stressed
    # Fallback: keep lexicon spelling.
    return stressed


@lru_cache(maxsize=8)
def _load_lexicon(path: str) -> dict[str, str]:
    """Map casefolded unstressed form → stressed form (with combining acute)."""
    file_path = Path(path)
    if not file_path.is_file():
        return {}

    lexicon: dict[str, str] = {}
    for line_no, raw in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            source, target = line.split("\t", 1)
            source, target = source.strip(), target.strip()
        else:
            source, target = strip_stress_marks(line), line
        if not source or not target:
            _LOGGER.warning("stress lexicon %s:%d: empty entry, skipped", file_path, line_no)
            continue
        if COMBINING_ACUTE not in target:
            _LOGGER.warning(
                "stress lexicon %s:%d: target has no U+0301, skipped: %s",
                file_path,
                line_no,
                target,
            )
            continue
        lexicon[_casefold_key(source)] = target
    return lexicon


def apply_lexicon(text: str, lexicon: dict[str, str]) -> str:
    """Replace whole words using the stress lexicon (lexicon wins)."""
    if not text or not lexicon:
        return text

    def repl(match: re.Match[str]) -> str:
        word = match.group(0)
        key = _casefold_key(word)
        hit = lexicon.get(key)
        if hit is None:
            return word
        return _match_case(word, hit)

    return _WORD_RE.sub(repl, text)


def _install_cpu_stanza_pipeline() -> None:
    """Force Stanza onto CPU so stress marking does not contend for the TTS GPU."""
    global _cpu_stanza_patched
    if _cpu_stanza_patched:
        return

    import ukrainian_word_stress.stressify_ as stressify_mod

    def _create_stanza_pipeline():
        try:
            import stanza
        except ImportError as exc:  # pragma: no cover - import guard mirrored from upstream
            raise RuntimeError(
                "Stanza is required for STRESS_DISAMBIGUATION=stanza. "
                "Install it with: pip install stanza && python -c \"import stanza; stanza.download('uk')\""
            ) from exc

        try:
            return stanza.Pipeline(
                "uk",
                processors="tokenize,pos,mwt",
                download_method=stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
                logging_level=logging.getLevelName(_LOGGER.getEffectiveLevel()),
                device="cpu",
            )
        except Exception as exc:  # pragma: no cover - upstream init errors
            raise RuntimeError(
                "Failed to initialize the Stanza Ukrainian pipeline on CPU. "
                "Download models with: python -c \"import stanza; stanza.download('uk')\""
            ) from exc

    stressify_mod._create_stanza_pipeline = _create_stanza_pipeline  # type: ignore[attr-defined]
    _cpu_stanza_patched = True


@lru_cache(maxsize=8)
def _stressifier(on_ambiguity: str, disambiguation: str, prefer_cpu: bool):
    try:
        from ukrainian_word_stress import StressSymbol, Stressifier
    except ImportError as exc:  # pragma: no cover - depends on the install extras
        raise ImportError(
            "ukrainian-word-stress is required for stress marking. "
            "Install it with: pip install -e '.[dataset]'"
        ) from exc

    if disambiguation == "stanza" and prefer_cpu:
        _install_cpu_stanza_pipeline()

    return Stressifier(
        stress_symbol=StressSymbol.CombiningAcuteAccent,
        on_ambiguity=on_ambiguity,
        disambiguation=disambiguation,
    )


def apply_stress_marks(
    text: str,
    *,
    on_ambiguity: str = "skip",
    disambiguation: str = "dictionary",
    lexicon: dict[str, str] | None = None,
    prefer_cpu: bool = True,
    force: bool = False,
) -> str:
    """Return ``text`` with a combining acute after every stressed vowel.

    Text is normalised (apostrophes) first. Unless ``force`` is set, already-marked
    text is left untouched apart from lexicon overrides, so repeated dataset passes
    stay idempotent. ``disambiguation`` defaults to dictionary lookup rather than
    the library's ``auto``: ``auto`` silently switches to Stanza when that package
    happens to be installed, which would make training and synthesis mark the same
    sentence differently.
    """
    if not text.strip():
        return text

    text = normalize_uk_text(text)
    if force:
        text = strip_stress_marks(text)
    elif has_stress_marks(text):
        return apply_lexicon(text, lexicon or {})

    marked = _stressifier(on_ambiguity, disambiguation, prefer_cpu)(text)
    return apply_lexicon(marked, lexicon or {})


def stressify(
    text: str,
    config: StressConfig,
    *,
    force: bool = False,
    audio_path: Path | str | None = None,
) -> str:
    """Apply stress marks according to project config, or pass text through.

    When ``audio_path`` is set and ``config.acoustic_fallback`` is enabled, words
    still unmarked after dictionary/Stanza/lexicon are filled from WAV energy.
    Synthesis callers omit ``audio_path``.
    """
    if not config.enabled:
        return text

    lexicon: dict[str, str] = {}
    lexicon_path = (config.lexicon_path or "").strip()
    if lexicon_path:
        lexicon = _load_lexicon(str(Path(lexicon_path).resolve()))

    marked = apply_stress_marks(
        text,
        on_ambiguity=config.on_ambiguity,
        disambiguation=config.disambiguation,
        lexicon=lexicon,
        prefer_cpu=config.prefer_cpu,
        force=force,
    )
    if audio_path is not None and config.acoustic_fallback:
        from fish_studio.stress_acoustic import apply_acoustic_stress

        marked = apply_acoustic_stress(marked, audio_path)
    return marked
