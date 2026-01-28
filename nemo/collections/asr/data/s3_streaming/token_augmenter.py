"""
Token augmentation for S3 streaming dataset.

Adds special tokens like <eou> (end of utterance) based on punctuation rules.
"""

from typing import Set


# Sentence-ending punctuation by script type
SENTENCE_ENDINGS: Set[str] = {
    # Latin/Cyrillic/Hebrew (standard Western punctuation)
    '.', '!', '?',
    # Chinese/Japanese full-width punctuation
    '。',  # Chinese period
    '！',  # Chinese exclamation
    '？',  # Chinese question mark
    # Chinese ellipsis (sometimes used as ending)
    '…',
    # Arabic punctuation
    '؟',  # Arabic question mark
    '۔',  # Urdu full stop
}


class TokenAugmenter:
    """
    Augments text with special tokens.

    Rules for <eou> (end of utterance):
    - Add <eou> ONLY if text ends with sentence-ending punctuation
    - Do NOT add if text ends mid-sentence (no punctuation, comma, etc.)

    This helps the model learn when an utterance is complete vs. continuing.
    """

    def __init__(
        self,
        eou_token: str = "<eou>",
        add_eou: bool = True,
        sentence_endings: Set[str] = None,
    ):
        """
        Initialize token augmenter.

        Args:
            eou_token: Token to add at end of complete utterances
            add_eou: Whether to add EOU tokens
            sentence_endings: Custom set of sentence-ending punctuation
        """
        self.eou_token = eou_token
        self.add_eou = add_eou
        self.sentence_endings = sentence_endings or SENTENCE_ENDINGS

        self._stats = {
            'total': 0,
            'eou_added': 0,
            'eou_skipped': 0,
        }

    def __call__(self, sample: dict) -> dict:
        """
        Augment sample text with special tokens.

        Args:
            sample: Dict with 'text' key

        Returns:
            Sample with augmented text
        """
        self._stats['total'] += 1

        if not self.add_eou:
            return sample

        text = sample.get('text', '').rstrip()

        if text and text[-1] in self.sentence_endings:
            sample['text'] = text + ' ' + self.eou_token
            self._stats['eou_added'] += 1
        else:
            self._stats['eou_skipped'] += 1

        return sample

    def should_add_eou(self, text: str) -> bool:
        """
        Check if text ends with sentence-ending punctuation.

        Args:
            text: Text to check

        Returns:
            True if text ends with sentence-ending punctuation
        """
        text = text.rstrip()
        if not text:
            return False
        return text[-1] in self.sentence_endings

    def get_stats(self) -> dict:
        """Return augmentation statistics."""
        return self._stats.copy()
