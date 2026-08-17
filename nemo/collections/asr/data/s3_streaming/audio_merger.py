"""
Audio merger for on-the-fly utterance concatenation with silence gaps.

This module enables training ASR models to recognize end-of-utterance boundaries
by creating multi-utterance samples with silence between them.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from nemo.utils import logging

from .token_augmenter import SENTENCE_ENDINGS


@dataclass
class MergeConfig:
    """Configuration for audio merging."""
    enabled: bool = True
    merge_probability: float = 0.3  # 30% of samples will be merged
    min_utterances: int = 2
    max_utterances: int = 3
    silence_min_sec: float = 0.5
    silence_max_sec: float = 2.0
    max_merged_duration: float = 30.0  # Max total duration after merge
    add_trailing_silence: bool = True  # Add silence after last utterance
    trailing_silence_min_sec: float = 0.3
    trailing_silence_max_sec: float = 1.0
    # LID/code-switch mode: skip the "first sample must end with punctuation"
    # gate. EOU isn't being learned here, so punctuation is irrelevant; we want
    # to merge regardless so cross-language concatenations get produced.
    require_punctuation: bool = True
    # Require every merged segment to START with an uppercase letter so we
    # don't glue together mid-sentence fragments (e.g. "e depois..." +
    # "imitando..."). Together with require_punctuation this guarantees each
    # segment is a complete sentence in soft/cased training.
    require_starts_capital: bool = True


class AudioMerger:
    """
    Merges multiple audio samples with silence gaps for EOU training.

    This helps the model learn:
    1. Sentence boundaries marked by silence + punctuation = <eou>
    2. Continue transcribing after pauses (not emit blank forever)
    3. Distinguish between hesitation pauses vs turn-ending pauses

    Example:
        Input samples:
            - {"audio": [audio1], "text": "Hello world.", "duration": 2.0}
            - {"audio": [audio2], "text": "How are you?", "duration": 1.5}

        Output (merged):
            - {"audio": [audio1 + silence + audio2 + trailing_silence],
               "text": "Hello world.|How are you?",  # | marks EOU positions
               "duration": 5.0,
               "eou_positions": [0, 1]}  # Both texts end with punctuation
    """

    def __init__(self, config: MergeConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        self._stats = {
            'total_processed': 0,
            'merged': 0,
            'skipped_duration': 0,
            'skipped_no_punctuation': 0,
            'skipped_no_capital': 0,
            'outputs': 0,
            'merged_outputs': 0,
            'single_outputs': 0,
            'merged_segments': 0,
        }

    def should_merge(self) -> bool:
        """Decide whether to merge based on probability."""
        return self.config.enabled and random.random() < self.config.merge_probability

    def _generate_silence(self, duration_sec: float) -> np.ndarray:
        """
        Generate silence (or very low-level noise for realism).

        Args:
            duration_sec: Duration in seconds

        Returns:
            Numpy array of silence samples
        """
        num_samples = int(duration_sec * self.sample_rate)
        # Add very small noise to avoid perfect silence (more realistic)
        # Amplitude ~0.0001 is essentially inaudible
        noise = np.random.randn(num_samples).astype(np.float32) * 0.0001
        return noise

    def _get_random_silence_duration(self) -> float:
        """Get random silence duration between min and max."""
        return random.uniform(
            self.config.silence_min_sec,
            self.config.silence_max_sec
        )

    def _get_trailing_silence_duration(self) -> float:
        """Get random trailing silence duration."""
        return random.uniform(
            self.config.trailing_silence_min_sec,
            self.config.trailing_silence_max_sec
        )

    def _ends_with_punctuation(self, text: str) -> bool:
        """Check if text ends with sentence-ending punctuation."""
        text = text.rstrip()
        if not text:
            return False
        return text[-1] in SENTENCE_ENDINGS

    @staticmethod
    def _starts_with_capital(text: str) -> bool:
        """Whether the first alphabetical character is uppercase.
        Skips leading non-letter characters (quotes, digits, etc.)."""
        for ch in text or '':
            if ch.isalpha():
                return ch.isupper()
        return False

    def can_merge(self, samples: List[dict]) -> Tuple[bool, str]:
        """
        Check if samples can be merged.

        Args:
            samples: List of sample dicts with 'audio', 'text', 'duration'

        Returns:
            Tuple of (can_merge, reason_if_not)
        """
        if len(samples) < self.config.min_utterances:
            return False, "not_enough_samples"

        # Selective backprop: never merge clean with noisy. A merged sample
        # carries a single cleanliness flag, so its segments must agree —
        # otherwise we'd either pollute the decoder/joint with noisy text
        # (if tagged clean) or waste clean text (if tagged noisy). The default
        # 'clean'=True means samples without the tag (feature off) merge freely.
        first_clean = samples[0].get('clean', True)
        if any(s.get('clean', True) != first_clean for s in samples[1:]):
            return False, "mixed_cleanliness"

        # Calculate total duration with minimum silence
        total_duration = sum(s.get('duration', 0) for s in samples)
        min_silence = self.config.silence_min_sec * (len(samples) - 1)
        if self.config.add_trailing_silence:
            min_silence += self.config.trailing_silence_min_sec

        if total_duration + min_silence > self.config.max_merged_duration:
            return False, "exceeds_max_duration"

        # Every merged sample should end with terminal punctuation so that the
        # downstream tokenizer can emit <eou> after each segment. Without this,
        # the lang tag floats without an <eou> in front of it and the model
        # learns an inconsistent end-of-utterance pattern.
        # In LID/code-switch mode this gate is disabled (require_punctuation=False).
        if self.config.require_punctuation:
            for idx, s in enumerate(samples):
                if not self._ends_with_punctuation(s.get('text', '')):
                    return False, f"sample_{idx}_no_punctuation"

        # Every merged segment should also START with a capital letter — these
        # are full sentences, not mid-sentence fragments. Pairs with the
        # punctuation gate above.
        if self.config.require_starts_capital:
            for idx, s in enumerate(samples):
                if not self._starts_with_capital(s.get('text', '')):
                    return False, f"sample_{idx}_no_capital"

        return True, "ok"

    def merge(self, samples: List[dict]) -> Optional[dict]:
        """
        Merge multiple samples into one with silence gaps.

        Args:
            samples: List of sample dicts with 'audio', 'text', 'duration'

        Returns:
            Merged sample dict or None if merge failed
        """
        self._stats['total_processed'] += 1

        can_merge, reason = self.can_merge(samples)
        if not can_merge:
            if reason == "exceeds_max_duration":
                self._stats['skipped_duration'] += 1
            elif reason.endswith("_no_punctuation"):
                self._stats['skipped_no_punctuation'] += 1
            elif reason.endswith("_no_capital"):
                self._stats['skipped_no_capital'] += 1
            return None

        # Build merged audio and text
        merged_audio_parts = []
        merged_texts = []
        merged_phonemes = []
        eou_positions = []  # Track which segments should have EOU
        total_duration = 0.0

        for i, sample in enumerate(samples):
            audio = sample.get('audio')
            text = sample.get('text', '').strip()
            duration = sample.get('duration', 0)

            if audio is None:
                return None

            # Convert to numpy if needed
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)

            # Add audio
            merged_audio_parts.append(audio)
            merged_texts.append(text)
            merged_phonemes.append(sample.get('phonemes'))
            total_duration += duration

            # Track if this segment ends with punctuation (should have EOU)
            if self._ends_with_punctuation(text):
                eou_positions.append(i)

            # Add silence gap between samples (not after last one yet)
            if i < len(samples) - 1:
                silence_duration = self._get_random_silence_duration()
                # Check if adding this silence exceeds max duration
                remaining_audio = sum(s.get('duration', 0) for s in samples[i+1:])
                if total_duration + silence_duration + remaining_audio > self.config.max_merged_duration:
                    # Use minimum silence instead
                    silence_duration = self.config.silence_min_sec

                silence = self._generate_silence(silence_duration)
                merged_audio_parts.append(silence)
                total_duration += silence_duration

        # Add trailing silence if configured
        if self.config.add_trailing_silence:
            trailing_duration = self._get_trailing_silence_duration()
            if total_duration + trailing_duration <= self.config.max_merged_duration:
                trailing_silence = self._generate_silence(trailing_duration)
                merged_audio_parts.append(trailing_silence)
                total_duration += trailing_duration

        # Concatenate all audio parts
        merged_audio = np.concatenate(merged_audio_parts)

        self._stats['merged'] += 1

        return {
            'audio': merged_audio,
            'text': '|'.join(merged_texts),  # Use | as segment separator
            'duration': total_duration,
            'merged_count': len(samples),
            'eou_positions': eou_positions,
            'original_texts': merged_texts,  # Keep original texts for tokenization
            # Precomputed IPA JSON strings, parallel to original_texts.
            'original_phonemes': merged_phonemes,
            # First-sample lang preserved for backward compat (legacy add_lang_token path).
            'lang': samples[0].get('lang') if samples else None,
            # Per-segment langs parallel to original_texts. Used by lid_mode to
            # emit `[<lang_i>] * num_words_i` per segment for code-switch training.
            'langs': [s.get('lang') for s in samples],
            # Cleanliness for selective backprop. can_merge guarantees all
            # segments agree, so the first sample's flag is the merged flag.
            'clean': samples[0].get('clean', True),
        }

    def get_stats(self) -> dict:
        """Return merger statistics."""
        return self._stats.copy()

    def record_output(self, segment_count: int) -> None:
        """Record one emitted output independently of merge attempts."""
        segment_count = max(1, int(segment_count))
        self._stats['outputs'] += 1
        if segment_count > 1:
            self._stats['merged_outputs'] += 1
            self._stats['merged_segments'] += segment_count
        else:
            self._stats['single_outputs'] += 1

    def log_stats(self):
        """Log merger statistics."""
        attempts = self._stats['total_processed']
        outputs = self._stats['outputs']
        if attempts == 0 and outputs == 0:
            logging.info("AudioMerger stats: No samples processed yet")
            return

        merged = self._stats['merged']
        attempt_rate = 100.0 * merged / attempts if attempts > 0 else 0
        merged_outputs = self._stats['merged_outputs']
        output_rate = 100.0 * merged_outputs / outputs if outputs > 0 else 0
        avg_segments = (
            self._stats['merged_segments'] / merged_outputs
            if merged_outputs > 0 else 0.0
        )

        logging.info(
            f"AudioMerger stats: outputs={outputs} merged={merged_outputs} "
            f"single={self._stats['single_outputs']} "
            f"merged_output_rate={output_rate:.1f}% avg_segments={avg_segments:.2f}"
        )
        logging.info(
            f"  Merge attempts: {merged}/{attempts} succeeded ({attempt_rate:.1f}%)"
        )
        logging.info(f"  Skipped (duration too long): {self._stats['skipped_duration']}")
        logging.info(f"  Skipped (segment no terminal punct): {self._stats['skipped_no_punctuation']}")
        logging.info(f"  Skipped (segment no leading capital): {self._stats['skipped_no_capital']}")
        logging.info(f"  Config: enabled={self.config.enabled}, probability={self.config.merge_probability}")


class MergeBuffer:
    """
    Buffer for collecting samples to merge.

    Collects samples and yields either merged samples or single samples
    based on merge probability and sample compatibility.
    """

    def __init__(self, merger: AudioMerger, config: MergeConfig):
        self.merger = merger
        self.config = config
        self.buffer: List[dict] = []
        self._target_count: Optional[int] = None

    def add(self, sample: dict) -> Optional[dict]:
        """
        Add a sample to the buffer.

        Returns:
            A sample (merged or single) if ready to yield, None otherwise
        """
        if not self.config.enabled:
            return sample

        self.buffer.append(sample)

        # Decide once per OUTPUT. Previously a Bernoulli miss merely delayed
        # the decision and max_utterances forced a merge, so p=0.5 yielded
        # almost no singles in short-duration windows.
        if self._target_count is None:
            if self.merger.should_merge():
                self._target_count = random.randint(
                    self.config.min_utterances,
                    self.config.max_utterances,
                )
            else:
                self._target_count = 1

        if self._target_count == 1:
            result = self.buffer.pop(0)
            self._target_count = None
            self.merger.record_output(1)
            return result

        if len(self.buffer) < self._target_count:
            return None

        num_to_merge = min(
            self._target_count,
            len(self.buffer),
            self.config.max_utterances,
        )
        if num_to_merge >= self.config.min_utterances:
            samples_to_merge = self.buffer[:num_to_merge]
            merged = self.merger.merge(samples_to_merge)
            if merged is not None:
                self.buffer = self.buffer[num_to_merge:]
                self._target_count = None
                self.merger.record_output(num_to_merge)
                return merged

        # Incompatible merge (normally duration): emit one single and make a
        # fresh output decision next time. The remaining samples stay ordered.
        result = self.buffer.pop(0)
        self._target_count = None
        self.merger.record_output(1)
        return result

    def drain(self) -> List[dict]:
        """Drain all remaining samples from buffer."""
        remaining = self.buffer
        self.buffer = []
        self._target_count = None
        for _ in remaining:
            self.merger.record_output(1)
        return remaining
