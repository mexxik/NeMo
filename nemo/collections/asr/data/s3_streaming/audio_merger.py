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
    silence_max_sec: float = 1.5
    max_merged_duration: float = 30.0  # Max total duration after merge
    add_trailing_silence: bool = True  # Add silence after last utterance
    trailing_silence_min_sec: float = 0.3
    trailing_silence_max_sec: float = 1.0


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

        # Calculate total duration with minimum silence
        total_duration = sum(s.get('duration', 0) for s in samples)
        min_silence = self.config.silence_min_sec * (len(samples) - 1)
        if self.config.add_trailing_silence:
            min_silence += self.config.trailing_silence_min_sec

        if total_duration + min_silence > self.config.max_merged_duration:
            return False, "exceeds_max_duration"

        # At least one sample should end with punctuation for meaningful EOU
        has_punctuation = any(
            self._ends_with_punctuation(s.get('text', ''))
            for s in samples[:-1]  # Check all but last (last always gets EOU if punctuated)
        )
        # Actually, we want ALL samples to ideally end with punctuation
        # But we can be flexible - at least require the first one
        if not self._ends_with_punctuation(samples[0].get('text', '')):
            return False, "first_sample_no_punctuation"

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
            elif reason == "first_sample_no_punctuation":
                self._stats['skipped_no_punctuation'] += 1
            return None

        # Build merged audio and text
        merged_audio_parts = []
        merged_texts = []
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

        # Log progress periodically
        if self._stats['merged'] % 1000 == 0:
            logging.info(
                f"AudioMerger: {self._stats['merged']} samples merged "
                f"({len(samples)} utterances, {total_duration:.1f}s)"
            )

        return {
            'audio': merged_audio,
            'text': '|'.join(merged_texts),  # Use | as segment separator
            'duration': total_duration,
            'merged_count': len(samples),
            'eou_positions': eou_positions,
            'original_texts': merged_texts,  # Keep original texts for tokenization
        }

    def get_stats(self) -> dict:
        """Return merger statistics."""
        return self._stats.copy()

    def log_stats(self):
        """Log merger statistics."""
        total = self._stats['total_processed']
        if total == 0:
            logging.info("AudioMerger stats: No samples processed yet")
            return

        merged = self._stats['merged']
        merge_rate = 100.0 * merged / total if total > 0 else 0

        logging.info(f"AudioMerger stats: {merged}/{total} merged ({merge_rate:.1f}%)")
        logging.info(f"  Skipped (duration too long): {self._stats['skipped_duration']}")
        logging.info(f"  Skipped (first sample no punctuation): {self._stats['skipped_no_punctuation']}")
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

    def add(self, sample: dict) -> Optional[dict]:
        """
        Add a sample to the buffer.

        Returns:
            A sample (merged or single) if ready to yield, None otherwise
        """
        if not self.config.enabled:
            return sample

        self.buffer.append(sample)

        # Check if we should try to merge
        if len(self.buffer) >= self.config.max_utterances:
            # Buffer full, must yield something
            return self._try_merge_and_yield(force_decision=True)

        # With some probability, try to merge even with fewer samples
        if len(self.buffer) >= self.config.min_utterances and self.merger.should_merge():
            return self._try_merge_and_yield(force_decision=False)

        return None

    def _try_merge_and_yield(self, force_decision: bool = False) -> dict:
        """
        Try to merge buffered samples, yield result.

        Args:
            force_decision: If True, always attempt merge (buffer full).
                           If False, merge decision already made by should_merge().
        """
        should_try_merge = force_decision or len(self.buffer) >= self.config.min_utterances

        if should_try_merge:
            # Try to merge
            num_to_merge = random.randint(
                self.config.min_utterances,
                min(len(self.buffer), self.config.max_utterances)
            )
            samples_to_merge = self.buffer[:num_to_merge]
            merged = self.merger.merge(samples_to_merge)

            if merged is not None:
                # Successfully merged
                self.buffer = self.buffer[num_to_merge:]
                return merged

        # Merge failed or not merging, yield first sample
        return self.buffer.pop(0)

    def drain(self) -> List[dict]:
        """Drain all remaining samples from buffer."""
        remaining = self.buffer
        self.buffer = []
        return remaining
