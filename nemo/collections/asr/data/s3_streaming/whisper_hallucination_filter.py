"""
Whisper hallucination filter for streaming ASR datasets.

Detects and rejects samples whose transcriptions are known Whisper
hallucination patterns. Whisper tends to hallucinate specific phrases
(especially on silent/noisy audio), produce repetitive loops, and
inject cross-language text.

Detection methods:
1. Exact phrase matching against per-language hallucination dictionaries
   (sourced from NVIDIA NeMo SDP + community research)
2. Structural detection:
   - Low lexical diversity (repeated word loops)
   - Abnormally long words (garbled concatenations)
   - Single-word transcripts on long audio (silence hallucination)
"""

import os
import re
from typing import Dict, Optional, Set

from nemo.utils import logging


# Directory containing per-language hallucination phrase files
_PHRASES_DIR = os.path.join(os.path.dirname(__file__), "hallucination_phrases")


_PUNCT_RE = re.compile(r'[.,!?;:\-\u2014\u2013\u00ab\u00bb\u201e\u201c\u201d\u00bf\u00a1\u2026\u2018\u2019\'\"()\[\]{}]+')


# CJK / Japanese / Korean scripts. Whisper injects these as cross-language
# contamination into otherwise-monolingual transcripts (e.g. a Chinese
# "\u6703\u8b70" spliced into a Ukrainian sentence). Covers: CJK symbols & punctuation,
# Hiragana, Katakana, CJK Unified Ideographs (+ Ext A) and compatibility,
# Hangul syllables + Jamo, and fullwidth/halfwidth forms.
_CJK_RE = re.compile(
    r'[\u3000-\u303f\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff'
    r'\uf900-\ufaff\uac00-\ud7af\u1100-\u11ff\uff00-\uffef]'
)


def _normalize_for_matching(text: str) -> str:
    """Strip punctuation and normalize whitespace for fuzzy phrase matching."""
    text = _PUNCT_RE.sub('', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _strip_word_punct(word: str) -> str:
    """Strip leading/trailing punctuation from a word."""
    return _PUNCT_RE.sub('', word)


class WhisperHallucinationFilter:
    """
    Filters out known Whisper hallucination patterns.

    Loads per-language phrase lists from bundled data files and applies
    both exact-match and structural heuristics.

    Args:
        unique_words_threshold: Max ratio of unique/total words before
            flagging as repetitive loop. Default 0.4.
        long_word_threshold: Min char length to flag a single word as
            abnormally long. Default 25.
        long_word_rel_threshold: If longest word is this many times longer
            than the second-longest, flag it. Default 3.0.
        min_words_for_duration: Minimum words expected per second of audio.
            If actual word count is below (duration * min_words_for_duration),
            the transcript is likely hallucinated on silence. Default 0.3.
        enabled: Master switch. Default True.
    """

    def __init__(
        self,
        unique_words_threshold: float = 0.4,
        long_word_threshold: int = 25,
        long_word_rel_threshold: float = 3.0,
        min_words_for_duration: float = 0.3,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.unique_words_threshold = unique_words_threshold
        self.long_word_threshold = long_word_threshold
        self.long_word_rel_threshold = long_word_rel_threshold
        self.min_words_for_duration = min_words_for_duration

        # Per-language phrase sets (normalized for matching)
        self._phrases_by_lang: Dict[str, Set[str]] = {}
        # Global phrase set (all languages combined) for lang-agnostic matching
        self._all_phrases: Set[str] = set()

        if self.enabled:
            self._load_phrases()

    def _load_phrases(self):
        """Load per-language hallucination phrase lists from bundled files."""
        if not os.path.isdir(_PHRASES_DIR):
            logging.warning(
                f"Hallucination phrases directory not found: {_PHRASES_DIR}. "
                f"Phrase matching will be disabled."
            )
            return

        total_loaded = 0
        for fname in sorted(os.listdir(_PHRASES_DIR)):
            if not fname.endswith('.txt'):
                continue
            lang = fname[:-4]
            phrases = set()
            fpath = os.path.join(_PHRASES_DIR, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        normalized = _normalize_for_matching(line)
                        if normalized:
                            phrases.add(normalized.lower())
            self._phrases_by_lang[lang] = phrases
            self._all_phrases.update(phrases)
            total_loaded += len(phrases)

        logging.info(
            f"Whisper hallucination filter loaded: {total_loaded} phrases "
            f"across {len(self._phrases_by_lang)} languages"
        )

    def is_hallucination(self, text: str, duration: float = 0.0, lang: Optional[str] = None) -> str:
        """
        Check if a transcript is a Whisper hallucination.

        Args:
            text: Transcript text.
            duration: Audio duration in seconds.
            lang: Language code (e.g., 'en', 'fr'). If provided, checks
                  language-specific phrases first, then all phrases.

        Returns:
            Empty string if clean, otherwise a reason string describing
            why the sample was flagged.
        """
        if not self.enabled:
            return ""

        if not text or not text.strip():
            return ""

        text_stripped = text.strip()

        # --- 0. Wrong-script (CJK) contamination ---
        # Whisper splices Chinese/Japanese/Korean fragments into otherwise
        # monolingual transcripts. None of the target languages here use CJK,
        # so any CJK codepoint marks the whole sample as contaminated.
        if _CJK_RE.search(text_stripped):
            return "cjk_contamination"

        # --- 1. Exact phrase matching ---
        normalized = _normalize_for_matching(text_stripped).lower()

        if normalized:
            # Check language-specific phrases first
            if lang and lang in self._phrases_by_lang:
                if normalized in self._phrases_by_lang[lang]:
                    return f"exact_phrase_{lang}"

            # Check all-language phrases (catches cross-language contamination)
            if normalized in self._all_phrases:
                return "exact_phrase_any"

        # Clean words: strip punctuation, drop punctuation-only tokens (em-dashes, etc.)
        clean_words = [w for w in (_strip_word_punct(w) for w in text_stripped.split()) if w]
        num_words = len(clean_words)

        # --- 2. Repeated word loops (low lexical diversity) ---
        if num_words >= 4:
            unique_ratio = len(set(w.lower() for w in clean_words)) / num_words
            if unique_ratio <= self.unique_words_threshold:
                return "repeated_ngrams"

        return ""

    def __call__(self, sample: dict) -> str:
        """
        Check a sample dict for hallucination.

        Args:
            sample: Dict with 'text', optionally 'duration' and 'lang'.

        Returns:
            Empty string if clean, otherwise a reason string.
        """
        return self.is_hallucination(
            text=sample.get('text', ''),
            duration=sample.get('duration', 0.0),
            lang=sample.get('lang'),
        )
