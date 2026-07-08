"""
Deterministic romanization for the LID `romanize` label scheme.

Every language is mapped to a shared lowercase-Latin space-separated form, so
the model can't cheat off the script (Cyrillic ⇒ uk) and must commit to a
language from acoustics. The romanized text is the dense content target that
sits between `<lang_S>` / `<lang_E>` boundary tokens (or trails each word under
`romanize_word`).

Romanizer selection (see `romanize()`):
  - `uk` uses a hand-tuned National-2010 transliteration map (kept for
    consistency with the existing en+uk model).
  - `en, de, es, fr, it, pt` use the identity romanizer (Latin script; just
    lowercase + fold diacritics to base ASCII). This keeps e.g. fr "ça" -> "ca"
    rather than uroman's phonetic "sa".
  - EVERY other language falls back to `romanize_universal`, backed by uroman
    (the ISI universal romanizer). This covers all non-Latin scripts (Cyrillic,
    Arabic, CJK, Devanagari, Greek, Thai, Hangul, ...) and any Latin-script
    language not listed above. Without it, non-Latin text is stripped to the
    empty string by `_finalize` — which silently voids the training target.

uroman is an OPTIONAL dependency. If it is not importable, the universal path
degrades to `romanize_identity` (i.e. non-Latin text -> "") and warns once, so
this module still imports in environments that only need en/uk (the app, host
tooling). Install it in the training image: `pip install uroman`.
"""

import functools
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

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


def _fold_latin(text: str) -> str:
    """Lowercase, map ß/ä/ü/… special-cases, then NFKD-strip combining marks so
    accented Latin folds to base ASCII (é→e, ç→c). Returns the folded string
    (NOT yet run through `_finalize`)."""
    s = (text or "").lower().translate(_LATIN_SPECIAL)
    s = unicodedata.normalize("NFKD", s)                       # é → e + combining ´
    return "".join(c for c in s if not unicodedata.combining(c))  # drop combining marks


def romanize_identity(text: str) -> str:
    """For languages already written in Latin script. Fold diacritics to base
    ASCII (é→e, ç→c) and special letters (ß→ss, ä→ae, ü→ue) BEFORE _finalize,
    so accented characters aren't deleted — which would split words and, under
    the romanize_word scheme, emit a stray extra `<lang>` tag per split."""
    return _finalize(_fold_latin(text))


# --- Universal romanizer (uroman) ---------------------------------------------
# Lazily constructed singleton: uroman loads sizeable rule tables at init, so we
# build it once per process on first use. `False` means "tried and failed" —
# don't retry (and don't spam the log) every call.
_UROMAN = None


def _get_uroman():
    global _UROMAN
    if _UROMAN is None:
        try:
            import uroman as _ur
            _UROMAN = _ur.Uroman()
        except Exception as e:  # not installed, or init failed
            logger.warning(
                "uroman unavailable (%s); non-Latin languages will romanize to "
                "empty strings. Install it in the training image: pip install uroman",
                e,
            )
            _UROMAN = False
    return _UROMAN


@functools.lru_cache(maxsize=200_000)
def romanize_universal(text: str) -> str:
    """Romanize arbitrary-script `text` to shared lowercase-Latin via uroman,
    then fold + `_finalize` so the output matches the other romanizers' charset
    ([a-z ]). Falls back to `romanize_identity` if uroman is unavailable.

    Cached: the training set is a bounded corpus, so after the first epoch every
    call is a dict hit (uroman itself is ~0.25 ms/call uncached)."""
    if not (text or "").strip():
        return ""
    u = _get_uroman()
    if not u:
        return romanize_identity(text)
    return _finalize(_fold_latin(u.romanize_string(text)))


# Explicit per-language romanizers. Anything NOT listed here uses
# `romanize_universal` (uroman) — see `romanize()`.
ROMANIZERS = {
    "uk": romanize_uk,
    # Latin-script languages kept on the identity romanizer (no uroman phonetic
    # rewrites, e.g. fr "ça" -> "ca" not "sa").
    "en": romanize_identity,
    "de": romanize_identity,
    "es": romanize_identity,
    "fr": romanize_identity,
    "it": romanize_identity,
    "pt": romanize_identity,
    # Scandinavian (Latin script; _LATIN_SPECIAL folds ø→o, æ→ae, å/ä/ö).
    "da": romanize_identity,
    "sv": romanize_identity,
}


def romanize(text: str, lang: str) -> str:
    """Romanize `text` for `lang` to shared lowercase-Latin space-separated form.
    Languages without an explicit romanizer use the universal (uroman) path,
    which covers every script."""
    fn = ROMANIZERS.get((lang or "").lower(), romanize_universal)
    return fn(text)
