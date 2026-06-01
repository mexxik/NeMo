"""
Deterministic romanization for the LID `romanize` label scheme.

Both languages are mapped to a shared lowercase-Latin space-separated form,
so the model can't cheat off the script (Cyrillic ⇒ uk) and must commit to a
language from acoustics. The romanized text is the dense content target that
sits between `<lang_S>` / `<lang_E>` boundary tokens.

Add new languages by adding a transliteration map below and registering it in
ROMANIZERS. Languages already in Latin script (en, de, es, fr, it, pt, ...)
use the identity romanizer (lowercase, fold diacritics to base ASCII, strip to
[a-z ]).
"""

import re
import unicodedata

# German/ligature special-cases that NFKD won't fold to a base [a-z] letter.
# Applied before NFKD diacritic-stripping in romanize_identity.
_LATIN_SPECIAL = str.maketrans({
    "ß": "ss", "ä": "ae", "ö": "oe", "ü": "ue",
    "æ": "ae", "œ": "oe", "ø": "o", "ð": "d", "þ": "th",
})

# Ukrainian National romanization (2010), lowercased. Multi-char outputs are
# fine — they're just Latin letters the shared tokenizer will sub-word.
_UK_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "е": "e",
    "є": "ie", "ж": "zh", "з": "z", "и": "y", "і": "i", "ї": "i", "й": "i",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch",
    "ш": "sh", "щ": "shch", "ь": "", "ю": "iu", "я": "ia",
    "'": "", "’": "", "ʼ": "",
}

_KEEP_RE = re.compile(r"[^a-z ]+")
_WS_RE = re.compile(r"\s+")


def _finalize(s: str) -> str:
    """Lowercase Latin, drop everything except [a-z] and space, collapse ws."""
    s = _KEEP_RE.sub(" ", s.lower())
    return _WS_RE.sub(" ", s).strip()


def romanize_uk(text: str) -> str:
    out = []
    for ch in (text or "").lower():
        out.append(_UK_MAP.get(ch, ch))
    return _finalize("".join(out))


def romanize_identity(text: str) -> str:
    """For languages already written in Latin script. Fold diacritics to base
    ASCII (é→e, ç→c) and special letters (ß→ss, ä→ae, ü→ue) BEFORE _finalize,
    so accented characters aren't deleted — which would split words and, under
    the romanize_word scheme, emit a stray extra `<lang>` tag per split."""
    s = (text or "").lower().translate(_LATIN_SPECIAL)
    s = unicodedata.normalize("NFKD", s)                       # é → e + combining ´
    s = "".join(c for c in s if not unicodedata.combining(c))  # drop combining marks
    return _finalize(s)


ROMANIZERS = {
    "uk": romanize_uk,
    # Latin-script languages — extend as you add languages.
    "en": romanize_identity,
    "de": romanize_identity,
    "es": romanize_identity,
    "fr": romanize_identity,
    "it": romanize_identity,
    "pt": romanize_identity,
}


def romanize(text: str, lang: str) -> str:
    """Romanize `text` for `lang` to shared lowercase-Latin space-separated form.
    Unknown languages fall back to the identity romanizer."""
    fn = ROMANIZERS.get((lang or "").lower(), romanize_identity)
    return fn(text)
