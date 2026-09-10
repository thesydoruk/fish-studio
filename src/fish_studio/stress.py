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
_homonym_patch_installed = False

# When the dictionary lists two accents and skip would leave the word bare,
# pick the reading that matches Stanza features. No match → still skip, so a
# rare alternate is not forced the way a lexicon override would force it.
# Values are insert-after indexes used by ukrainian_word_stress.
#
# зв'язок (masc, connections / dating) vs зв'язка (fem, bundle): the
# dictionary puts both stem and ending stress on the same masc tags.
_ZVIAZOK_VS_ZVIAZKA: tuple[tuple[tuple[str, ...], int], ...] = (
    (("Gender=Masc",), 7),  # зв'язку́ / зв'язки́ / зв'язка́ми
    (("Gender=Fem",), 4),  # зв'я́зку / зв'я́зки / зв'я́зками
)

_HOMONYM_ACCENT_BY_FEATS: dict[str, tuple[tuple[tuple[str, ...], int], ...]] = {
    "зв'язок": ((("Gender=Masc",), 6),),  # зв'язо́к
    "зв'язку": _ZVIAZOK_VS_ZVIAZKA,
    "зв'язком": ((("Gender=Masc",), 7),),  # зв'язко́м
    "зв'язки": _ZVIAZOK_VS_ZVIAZKA,
    "зв'язків": _ZVIAZOK_VS_ZVIAZKA,
    "зв'язкам": _ZVIAZOK_VS_ZVIAZKA,
    "зв'язками": _ZVIAZOK_VS_ZVIAZKA,
    "зв'язках": _ZVIAZOK_VS_ZVIAZKA,
    # Only when the required tag actually separates the two dictionary readings.
    "бути": ((("VerbForm=Inf",), 2),),  # бу́ти, not emphatic бути́
    "піти": ((("VerbForm=Inf",), 4),),  # піти́
    "тікати": ((("VerbForm=Inf",), 4),),  # тіка́ти
    # наси́пати (pf) / насипа́ти (ipf) share the same Inf tags. Stanza also
    # tags почали насипати as Perf, so Aspect cannot separate them.
    "насипати": ((("VerbForm=Inf",), 6),),  # насипа́ти
    "поводилися": ((("Number=Plur",), 4),),  # пово́дилися (behave)
    "поводились": ((("Number=Plur",), 4),),
    # стрільби́ (gen of стрільба́) vs стрі́льби (pl. shooting drills).
    "стрільби": (
        (("Case=Gen", "Number=Sing"), 8),
        (("Number=Plur",), 4),
    ),
    "помилки": (
        (("Number=Plur",), 7),  # помилки́
        (("Case=Gen", "Number=Sing"), 4),  # поми́лки
    ),
    "вислухати": ((("VerbForm=Inf",), 2),),  # ви́слухати
    "вислухали": ((("Number=Plur",), 2),),  # ви́слухали
    "вислухав": ((("Number=Sing",), 2),),
    "вислухала": ((("Number=Sing",), 2),),
    "вислухай": ((("Mood=Imp",), 2),),
    "вислухайте": ((("Mood=Imp",), 2),),
    "послухати": ((("VerbForm=Inf",), 5),),  # послу́хати
    "послухай": ((("Mood=Imp",), 5),),  # послу́хай
    "послухайте": ((("Mood=Imp",), 5),),  # послу́хайте
    "своїм": ((("Case=Ins",), 4),),  # свої́м, not dative сво́їм
    "руці": ((("Case=Dat",), 4), (("Case=Loc",), 4)),  # руці́
    "спокої": ((("Case=Loc",), 3),),  # спо́кої
    "сорок": ((("upos=NUM",), 2),),  # со́рок
    "саме": (
        (("upos=ADV",), 2),  # са́ме «що саме»
        (("upos=PART",), 2),
    ),
    "батьків": (
        (("upos=NOUN",), 6),  # батькі́в
        (("upos=ADJ",), 2),  # ба́тьків
    ),
    "коли": (
        (("upos=ADV",), 4),  # коли́
        (("upos=CCONJ",), 4),
        (("upos=SCONJ",), 4),
    ),
    "яка": (
        (("upos=PRON",), 3),  # яка́
        (("upos=DET",), 3),
        (("upos=NOUN",), 1),  # я́ка, gen/acc of як the animal
    ),
    "яку": (
        (("upos=PRON",), 3),  # яку́
        (("upos=DET",), 3),
        (("upos=NOUN",), 1),  # я́ку, dat/loc of як
    ),
    "зірки": (
        (("Number=Plur",), 5),  # зірки́
        (("Case=Gen", "Number=Sing"), 2),  # зі́рки
    ),
    # птахи́ (pl. of птах) vs пта́хи (gen of пта́ха).
    "птахи": (
        (("Number=Plur",), 5),
        (("Case=Gen", "Number=Sing"), 3),
    ),
    "птахам": ((("Number=Plur",), 5),),  # птаха́м
    "птахами": ((("Number=Plur",), 5),),
    "птахах": ((("Number=Plur",), 5),),
}


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


def _lexicon_form(word: str, lexicon: dict[str, str]) -> str | None:
    hit = lexicon.get(_casefold_key(word))
    if hit is None:
        return None
    return _match_case(word, hit)


def apply_lexicon(text: str, lexicon: dict[str, str]) -> str:
    """Replace whole words using the stress lexicon (lexicon wins).

    Hyphenated compounds also try each stem, so ``коли́-небудь`` becomes
    ``коли́-не́будь`` from the ``не́будь`` entry without listing every form.
    """
    if not text or not lexicon:
        return text

    def repl(match: re.Match[str]) -> str:
        word = match.group(0)
        hit = _lexicon_form(word, lexicon)
        if hit is not None:
            return hit
        if "-" not in strip_stress_marks(word):
            return word
        parts = word.split("-")
        rewritten = [_lexicon_form(part, lexicon) or part for part in parts]
        if rewritten == parts:
            return word
        return "-".join(rewritten)

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
                "Install it with: ./run.sh install server"
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


def _preferred_homonym_accent(parse: dict) -> int | None:
    """Accent index when Stanza tags pick one reading, or None to keep skip."""
    word = _casefold_key(str(parse.get("text") or ""))
    rules = _HOMONYM_ACCENT_BY_FEATS.get(word)
    if not rules:
        return None
    feats = str(parse.get("feats") or "")
    feat_list = [part for part in feats.split("|") if part]
    feat_list.append(f"upos={parse.get('upos') or ''}")
    for required, accent in rules:
        if all(tag in feat_list for tag in required):
            return accent
    return None


def _install_homonym_disambiguation() -> None:
    """Prefer the Stanza-tagged reading even if skip or a unique dict match
    already picked the other accent (e.g. genderless past-plural rows)."""
    global _homonym_patch_installed
    if _homonym_patch_installed:
        return

    import ukrainian_word_stress.stressify_ as stressify_mod

    original = stressify_mod._accent_positions_from_values
    skip = getattr(stressify_mod.OnAmbiguity, "Skip", "skip")
    mark_all = getattr(stressify_mod.OnAmbiguity, "All", "all")

    def _accent_positions_from_values(values, parse, on_ambiguity=skip):
        preferred = _preferred_homonym_accent(parse)
        if preferred is not None:
            all_accents = original(values, parse, mark_all)
            if preferred in all_accents:
                return [preferred]
        return original(values, parse, on_ambiguity)

    stressify_mod._accent_positions_from_values = _accent_positions_from_values
    _homonym_patch_installed = True


@lru_cache(maxsize=8)
def _stressifier(on_ambiguity: str, disambiguation: str, prefer_cpu: bool):
    try:
        from ukrainian_word_stress import StressSymbol, Stressifier
    except ImportError as exc:  # pragma: no cover - depends on the install extras
        raise ImportError(
            "ukrainian-word-stress is required for stress marking. "
            "Install it with: ./run.sh install server"
        ) from exc

    if disambiguation == "stanza" and prefer_cpu:
        _install_cpu_stanza_pipeline()
    _install_homonym_disambiguation()

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
    stay idempotent. ``lexicon`` is the unambiguous override file
    (``configs/stress_lexicon.txt``). ``disambiguation`` defaults to dictionary
    lookup rather than the library's ``auto``: ``auto`` silently switches to
    Stanza when that package happens to be installed, which would make training
    and synthesis mark the same sentence differently.
    """
    if not text.strip():
        return text

    text = normalize_uk_text(text)
    lexicon = lexicon or {}
    if force:
        text = strip_stress_marks(text)
    elif has_stress_marks(text):
        return apply_lexicon(text, lexicon)

    marked = _stressifier(on_ambiguity, disambiguation, prefer_cpu)(text)
    return apply_lexicon(marked, lexicon)


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
