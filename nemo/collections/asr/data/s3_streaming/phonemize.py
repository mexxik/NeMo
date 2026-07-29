"""Grapheme-to-phoneme helpers for multilingual LID targets.

The default backend is eSpeak NG through ``phonemizer``.  It is deterministic
and fast enough to precompute a manifest cache. Galician uses ``pycotovia``;
Mandarin uses ``pypinyin`` plus ``pinyin-to-ipa``; and Japanese uses OpenJTalk.

The public representation is ``List[str]``: one NFC-normalized IPA string per
spoken word.  Phone boundaries inside a word are intentionally removed because
the phoneme tokenizer is a fixed character model; the trailing language tag in
the dataset remains the authoritative word boundary.
"""

from functools import lru_cache
import unicodedata
from typing import Dict, List, Sequence, Tuple


SUPPORTED_LANGUAGES: Tuple[str, ...] = (
    "ar",
    "bg",
    "ca",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "ga",
    "gl",
    "hi",
    "hr",
    "hu",
    "it",
    "ja",
    "ko",
    "lb",
    "lt",
    "lv",
    "mt",
    "nb",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "tr",
    "uk",
    "zh",
)

# Prefer locale-specific voices when available, but accept the generic alias
# shipped by older eSpeak NG releases.  The first installed candidate wins.
_ESPEAK_VOICE_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "ar": ("ar",),
    "as": ("as",),
    "bg": ("bg",),
    "bn": ("bn",),
    # eSpeak has no dedicated Bodo or Dogri voices. Both corpora use
    # Devanagari, so the Hindi rules are the deterministic fallback for the
    # catch-all class.
    "brx": ("hi",),
    "ca": ("ca",),
    "cs": ("cs",),
    "da": ("da",),
    "de": ("de",),
    "doi": ("hi",),
    "el": ("el",),
    "en": ("en-us", "en"),
    "es": ("es", "es-es"),
    "et": ("et",),
    "fi": ("fi",),
    "fr": ("fr-fr", "fr"),
    "ga": ("ga",),
    "gu": ("gu",),
    "he": ("he",),
    "hi": ("hi",),
    "hr": ("hr",),
    "hu": ("hu",),
    "it": ("it",),
    "ja": ("ja",),
    "kn": ("kn",),
    "ko": ("ko",),
    "lb": ("lb",),
    "lt": ("lt",),
    "lv": ("lv",),
    "ml": ("ml",),
    "mt": ("mt",),
    "mr": ("mr",),
    "nb": ("nb", "no"),
    "ne": ("ne",),
    "nl": ("nl",),
    "pl": ("pl",),
    "pt": ("pt-pt", "pt"),
    "ro": ("ro",),
    "ru": ("ru",),
    "sd": ("sd",),
    "sk": ("sk",),
    "sl": ("sl",),
    "sv": ("sv",),
    "ta": ("ta",),
    "te": ("te",),
    "tr": ("tr",),
    "uk": ("uk",),
    "vi": ("vi",),
    "zh": ("cmn", "zh"),
}

G2P_LANGUAGES: Tuple[str, ...] = tuple(
    sorted(set(_ESPEAK_VOICE_CANDIDATES) | {"gl", "ja", "zh"})
)

# Private-use separators cannot collide with IPA emitted by eSpeak.
_PHONE_SEPARATOR = "\ue000"
_WORD_SEPARATOR = "\ue001"
_ESPEAK_IPA_SUBSTITUTES = {
    "^": "ʲ",  # eSpeak's residual palatalization mnemonic (notably Estonian)
}

# eSpeak's Vietnamese voice exposes its internal tone mnemonics in otherwise
# IPA output. Convert them to IPA tone contours before tokenization.
_VI_ESPEAK_TONES = {
    "1": "˧",     # ngang (closed syllable variant)
    "2": "˨˩",    # huyền
    "4": "˧˩˧",   # hỏi
    "5": "˧ˀ˥",   # ngã
    "6": "˨˩ˀ",   # nặng
    "7": "˧",     # ngang
    "ɜ": "˧˥",    # sắc
}


def _normalize_words(words: Sequence[str]) -> List[str]:
    from ipatok.ipa import replace_substitutes

    return [
        unicodedata.normalize("NFC", replace_substitutes(word.strip()))
        for word in words
        if word and word.strip()
    ]


@lru_cache(maxsize=1)
def _espeak_languages() -> frozenset:
    from phonemizer.backend import EspeakBackend

    return frozenset(EspeakBackend.supported_languages())


def resolve_backend(lang: str) -> str:
    """Return the concrete G2P backend identifier for an ISO-639-1 language."""
    lang = (lang or "").strip().lower()
    if lang == "gl":
        return "pycotovia:gl"
    if lang == "ja":
        return "pyopenjtalk:ja"
    if lang == "zh":
        return "pypinyin:cmn"
    candidates = _ESPEAK_VOICE_CANDIDATES.get(lang)
    if candidates is None:
        raise ValueError(
            f"Unsupported phoneme language {lang!r}; expected one of "
            f"{', '.join(G2P_LANGUAGES)}"
        )
    installed = _espeak_languages()
    for voice in candidates:
        if voice in installed:
            return f"espeak:{voice}"
    raise RuntimeError(
        f"No eSpeak NG voice for {lang!r}; tried {candidates}. "
        "Rebuild the training image with espeak-ng installed."
    )


@lru_cache(maxsize=None)
def _espeak_backend(voice: str):
    from phonemizer.backend import EspeakBackend

    return EspeakBackend(
        language=voice,
        preserve_punctuation=False,
        with_stress=False,
        tie=False,
        language_switch="remove-flags",
        words_mismatch="ignore",
    )


@lru_cache(maxsize=1)
def _separator():
    from phonemizer.separator import Separator

    return Separator(phone=_PHONE_SEPARATOR, word=_WORD_SEPARATOR)


def _parse_espeak_line(line: str, lang: str) -> List[str]:
    words = []
    for raw_word in (line or "").split(_WORD_SEPARATOR):
        # eSpeak's phone separator is useful while debugging but the model's
        # fixed char vocabulary represents every IPA atom directly.
        word = raw_word.replace(_PHONE_SEPARATOR, "")
        for mnemonic, ipa_symbol in _ESPEAK_IPA_SUBSTITUTES.items():
            word = word.replace(mnemonic, ipa_symbol)
        if lang == "vi":
            for mnemonic, ipa_tone in _VI_ESPEAK_TONES.items():
                word = word.replace(mnemonic, ipa_tone)
        if word:
            words.append(word)
    return _normalize_words(words)


def _phonemize_galician(texts: Sequence[str]) -> List[List[str]]:
    from pycotovia import cotovia_to_ipa, phonemize

    output: List[List[str]] = []
    for text in texts:
        cotovia = phonemize(text or "", lang="gl")
        ipa = cotovia_to_ipa(cotovia)
        output.append(_normalize_words(ipa.split()))
    return output


def _phonemize_chinese(texts: Sequence[str]) -> List[List[str]]:
    from pinyin_to_ipa import pinyin_to_ipa
    from pypinyin import Style, lazy_pinyin

    output: List[List[str]] = []
    for text in texts:
        syllables = lazy_pinyin(
            text or "",
            style=Style.TONE3,
            neutral_tone_with_five=True,
            errors="ignore",
        )
        ipa_syllables = []
        for syllable in syllables:
            # pinyin-to-ipa exposes pronunciation alternatives as an OrderedSet
            # of phone tuples. pypinyin has already resolved the polyphone from
            # phrase context, so use the first deterministic realization.
            alternatives = pinyin_to_ipa(syllable)
            phones = next(iter(alternatives), ())
            if phones:
                ipa_syllables.append("".join(phones))
        output.append(_normalize_words(ipa_syllables))
    return output


_OPENJTALK_TO_IPA = {
    "a": "a",
    "i": "i",
    "u": "ɯ",
    "e": "e",
    "o": "o",
    "k": "k",
    "ky": "kʲ",
    "kw": "kʷ",
    "g": "ɡ",
    "gy": "ɡʲ",
    "gw": "ɡʷ",
    "s": "s",
    "sh": "ɕ",
    "z": "z",
    "j": "d͡ʑ",
    "t": "t",
    "ty": "tʲ",
    "ch": "t͡ɕ",
    "ts": "t͡s",
    "d": "d",
    "dy": "dʲ",
    "n": "n",
    "ny": "ɲ",
    "h": "h",
    "hy": "ç",
    "f": "ɸ",
    "b": "b",
    "by": "bʲ",
    "p": "p",
    "py": "pʲ",
    "m": "m",
    "my": "mʲ",
    "y": "j",
    "r": "ɾ",
    "ry": "ɾʲ",
    "w": "w",
    "v": "v",
    "vy": "vʲ",
    "N": "ɴ",
    "cl": "ʔ",
}
_OPENJTALK_MORA_ENDS = {"a", "i", "u", "e", "o", "N", "cl"}
_OPENJTALK_SKIP = {"pau", "sil", "sp"}


def _phonemize_japanese(texts: Sequence[str]) -> List[List[str]]:
    import pyopenjtalk

    output: List[List[str]] = []
    for text in texts:
        raw_phones = pyopenjtalk.g2p(text or "", kana=False).split()
        morae: List[str] = []
        current: List[str] = []
        for raw_phone in raw_phones:
            if raw_phone in _OPENJTALK_SKIP:
                if current:
                    morae.append("".join(current))
                    current = []
                continue
            # Uppercase vowels are OpenJTalk's devoiced variants.
            phone = raw_phone.lower() if raw_phone in {"A", "I", "U", "E", "O"} else raw_phone
            mapped = _OPENJTALK_TO_IPA.get(phone)
            if mapped is None:
                raise RuntimeError(f"Unknown OpenJTalk phone {raw_phone!r} in text {text!r}")
            current.append(mapped)
            if phone in _OPENJTALK_MORA_ENDS:
                morae.append("".join(current))
                current = []
        if current:
            morae.append("".join(current))
        output.append(_normalize_words(morae))
    return output


def phonemize_texts(texts: Sequence[str], lang: str) -> List[List[str]]:
    """Phonemize a batch, preserving one output item per input utterance."""
    normalized_lang = (lang or "").strip().lower()
    backend_id = resolve_backend(normalized_lang)
    batch = [text or "" for text in texts]
    if backend_id == "pycotovia:gl":
        return _phonemize_galician(batch)
    if backend_id == "pyopenjtalk:ja":
        return _phonemize_japanese(batch)
    if backend_id == "pypinyin:cmn":
        return _phonemize_chinese(batch)

    voice = backend_id.split(":", 1)[1]
    lines = _espeak_backend(voice).phonemize(
        batch,
        separator=_separator(),
        strip=True,
        njobs=1,
    )
    if len(lines) != len(batch):
        raise RuntimeError(
            f"phonemizer returned {len(lines)} rows for {len(batch)} inputs "
            f"(lang={normalized_lang}, voice={voice})"
        )
    return [_parse_espeak_line(line, normalized_lang) for line in lines]


def phonemize_words(text: str, lang: str) -> List[str]:
    """Phonemize one utterance into one IPA string per spoken word."""
    return phonemize_texts([text or ""], lang)[0]


def validate_languages(langs: Sequence[str]) -> Dict[str, str]:
    """Resolve all requested languages and return ``lang -> backend``."""
    backends = {lang: resolve_backend(lang) for lang in langs}
    # Import specialist dependencies eagerly so a cache run fails before it
    # starts writing millions of rows if the image is incomplete.
    if "gl" in backends:
        import pycotovia  # noqa: F401
    if "ja" in backends:
        import pyopenjtalk  # noqa: F401
    if "zh" in backends:
        import pinyin_to_ipa  # noqa: F401
        import pypinyin  # noqa: F401
    return backends
