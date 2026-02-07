"""
Fast on-the-fly audio augmentation for ASR training.

Designed for minimal latency - only includes augmentations that can be
applied quickly without significantly impacting training throughput.

Supported augmentations:
- Speed perturbation: Change playback speed (affects duration)
- Gain: Adjust volume level
- Time masking: Zero out random segments (SpecAugment-style on waveform)
"""

import random
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch

try:
    import torchaudio
    import torchaudio.functional as F
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

from nemo.utils import logging


@dataclass
class AugmentationConfig:
    """Configuration for audio augmentation."""

    # Whether augmentation is enabled
    enabled: bool = True

    # Base probability of augmentation for non-repeated data (cycle 0)
    base_prob: float = 0.1

    # Speed perturbation
    speed_enabled: bool = True
    speed_prob: float = 0.5
    speed_range: Tuple[float, float] = (0.9, 1.1)

    # Gain adjustment
    gain_enabled: bool = True
    gain_prob: float = 0.5
    gain_range_db: Tuple[float, float] = (-6.0, 6.0)

    # Time masking (zero out random segments)
    time_mask_enabled: bool = True
    time_mask_prob: float = 0.3
    time_mask_max_ratio: float = 0.1  # Max 10% of audio masked


class AudioAugmentor:
    """
    Fast on-the-fly audio augmentor.

    Applies stacked augmentations with configurable probabilities.
    Designed for minimal latency to avoid becoming a training bottleneck.

    Usage:
        augmentor = AudioAugmentor(AugmentationConfig())

        # Apply augmentation (audio is numpy array or torch tensor)
        augmented_audio = augmentor(audio, sample_rate, force=True)

        # Or check if should augment based on source cycle
        if augmentor.should_augment(source_cycle=2, base_prob=0.1):
            audio = augmentor(audio, sample_rate, force=True)
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize augmentor.

        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()

        if not TORCHAUDIO_AVAILABLE and self.config.speed_enabled:
            logging.warning(
                "torchaudio not available - speed perturbation disabled. "
                "Install with: pip install torchaudio"
            )
            self.config.speed_enabled = False

        # Stats tracking
        self._total_calls = 0
        self._augmented_calls = 0
        self._speed_applied = 0
        self._gain_applied = 0
        self._time_mask_applied = 0

    def should_augment(self, source_cycle: int = 0) -> bool:
        """
        Determine if sample should be augmented based on source cycle.

        Args:
            source_cycle: How many times the source has cycled through its data.
                         0 = first pass (original data)
                         1+ = repeated data (always augment)

        Returns:
            True if sample should be augmented
        """
        if not self.config.enabled:
            return False

        if source_cycle > 0:
            # Repeated data - always augment
            return True

        # First pass - augment with base probability
        return random.random() < self.config.base_prob

    def __call__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        force: bool = False,
    ) -> np.ndarray:
        """
        Apply augmentation to audio.

        Args:
            audio: Audio samples as numpy array (1D)
            sample_rate: Audio sample rate
            force: If True, skip probability check and always apply augmentations

        Returns:
            Augmented audio as numpy array
        """
        self._total_calls += 1

        if not self.config.enabled:
            return audio

        # Check if we should augment at all (unless forced)
        if not force and random.random() >= self.config.base_prob:
            return audio

        self._augmented_calls += 1

        # Apply augmentations in numpy (no torch conversion for gain/time_mask)

        # 1. Speed perturbation (requires torch/torchaudio)
        if self.config.speed_enabled and random.random() < self.config.speed_prob:
            audio_tensor = torch.from_numpy(audio).float()
            audio_tensor = self._speed_perturb(audio_tensor, sample_rate)
            audio = audio_tensor.numpy()
            self._speed_applied += 1

        # 2. Gain adjustment (pure numpy)
        if self.config.gain_enabled and random.random() < self.config.gain_prob:
            audio = self._gain_np(audio)
            self._gain_applied += 1

        # 3. Time masking (pure numpy)
        if self.config.time_mask_enabled and random.random() < self.config.time_mask_prob:
            audio = self._time_mask_np(audio)
            self._time_mask_applied += 1

        return audio

    def _speed_perturb(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Apply speed perturbation via resampling.

        Changes both speed and pitch (like playing a record at different speed).
        This is the most common augmentation for ASR.
        """
        speed = random.uniform(*self.config.speed_range)

        # Compute new sample rate for speed change
        # speed > 1.0 = faster = higher apparent sample rate
        new_sr = int(sample_rate * speed)

        # Resample to new rate (changes duration)
        audio = F.resample(audio.unsqueeze(0), sample_rate, new_sr).squeeze(0)

        # Resample back to original rate
        audio = F.resample(audio.unsqueeze(0), new_sr, sample_rate).squeeze(0)

        return audio

    def _gain(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply gain adjustment in dB (torch version).
        """
        gain_db = random.uniform(*self.config.gain_range_db)
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear

    def _gain_np(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply gain adjustment in dB (pure numpy).
        """
        gain_db = random.uniform(*self.config.gain_range_db)
        gain_linear = 10 ** (gain_db / 20)
        return (audio * gain_linear).astype(audio.dtype)

    def _time_mask(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply time masking by zeroing out a random segment (torch version).
        """
        length = audio.shape[-1]
        max_mask_len = int(length * self.config.time_mask_max_ratio)

        if max_mask_len < 1:
            return audio

        mask_len = random.randint(1, max_mask_len)
        start = random.randint(0, length - mask_len)

        audio = audio.clone()
        audio[start:start + mask_len] = 0

        return audio

    def _time_mask_np(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply time masking by zeroing out a random segment (pure numpy).
        """
        length = audio.shape[-1]
        max_mask_len = int(length * self.config.time_mask_max_ratio)

        if max_mask_len < 1:
            return audio

        mask_len = random.randint(1, max_mask_len)
        start = random.randint(0, length - mask_len)

        audio = audio.copy()
        audio[start:start + mask_len] = 0

        return audio

    def get_stats(self) -> dict:
        """Get augmentation statistics."""
        return {
            'total_calls': self._total_calls,
            'augmented_calls': self._augmented_calls,
            'augmentation_rate': self._augmented_calls / max(1, self._total_calls),
            'speed_applied': self._speed_applied,
            'gain_applied': self._gain_applied,
            'time_mask_applied': self._time_mask_applied,
        }

    def log_stats(self):
        """Log augmentation statistics."""
        stats = self.get_stats()
        logging.info(
            f"Augmentation stats: {stats['augmented_calls']}/{stats['total_calls']} "
            f"({stats['augmentation_rate']:.1%}) - "
            f"speed={stats['speed_applied']}, gain={stats['gain_applied']}, "
            f"time_mask={stats['time_mask_applied']}"
        )

    def reset_stats(self):
        """Reset statistics counters."""
        self._total_calls = 0
        self._augmented_calls = 0
        self._speed_applied = 0
        self._gain_applied = 0
        self._time_mask_applied = 0
