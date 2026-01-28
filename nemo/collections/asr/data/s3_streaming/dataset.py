"""
S3 Multi-Language Streaming Dataset for ASR training.

Main dataset class that integrates all S3 streaming components.
"""

import os
import random
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from nemo.collections.asr.data.audio_to_text import _speech_collate_fn
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging

from .filters import FilterConfig, SampleFilter
from .lang_source_manager import LanguageSourceManager
from .round_robin import RoundRobinInterleaver
from .token_augmenter import TokenAugmenter

try:
    import boto3
    from botocore.config import Config as BotoConfig
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class ShuffleBuffer:
    """
    A shuffle buffer that accumulates samples and yields them in random order.

    This enables local randomization while maintaining streaming efficiency.
    """

    def __init__(self, buffer_size: int = 2000):
        self.buffer_size = buffer_size
        self.buffer: List = []

    def add(self, item):
        """Add an item to the buffer."""
        self.buffer.append(item)

    def is_ready(self) -> bool:
        """Check if buffer is full enough to start yielding."""
        return len(self.buffer) >= self.buffer_size

    def get_random(self):
        """Get and remove a random item from the buffer."""
        if not self.buffer:
            return None
        idx = random.randint(0, len(self.buffer) - 1)
        # Swap with last and pop for O(1) removal
        self.buffer[idx], self.buffer[-1] = self.buffer[-1], self.buffer[idx]
        return self.buffer.pop()

    def drain(self) -> Iterator:
        """Drain all remaining items in random order."""
        while self.buffer:
            yield self.get_random()

    def __len__(self):
        return len(self.buffer)


class S3MultiLangStreamingDataset(IterableDataset):
    """
    Streaming dataset that loads audio from S3 TAR files across multiple languages.

    Features:
    - Direct S3 streaming (no pre-download)
    - Round-robin language sampling
    - On-the-fly filtering (duration, char rate)
    - Token augmentation (<eou>)
    - Shuffle buffer for local randomization
    - Distributed training support

    Config example:
        model:
          train_ds:
            dataset_type: "s3_multilang_streaming"
            s3_bucket: "my-asr-bucket"
            s3_prefix: "asr-data/"
            language_sources:
              en: [common_en_train, yodas_en_en000]
              uk: [common_uk_train]
              zh: [common_zh_train]
            min_duration: 0.5
            max_duration: 15.0
            add_eou_token: true
            eou_token: "<eou>"
    """

    def __init__(
        self,
        # S3 config
        s3_bucket: str,
        s3_prefix: str = "",
        s3_endpoint_url: str = None,  # For R2, MinIO, etc.
        aws_region: str = "us-east-1",

        # Language config
        language_sources: Dict[str, List[str]] = None,

        # Tokenizer
        tokenizer=None,
        sample_rate: int = 16000,

        # Filtering
        min_duration: float = 0.5,
        max_duration: float = 15.0,
        max_chars_per_sec: float = 25.0,

        # Token augmentation
        add_eou_token: bool = True,
        eou_token: str = "<eou>",

        # Shuffle buffer
        shuffle_buffer_size: int = 2000,

        # Distributed training
        global_rank: int = 0,
        world_size: int = 1,

        # BPE options
        use_start_end_token: bool = True,

        # Other
        return_sample_id: bool = False,
    ):
        """
        Initialize S3 multi-language streaming dataset.

        Args:
            s3_bucket: S3 bucket name
            s3_prefix: Prefix path in bucket (e.g., "asr-data/")
            s3_endpoint_url: Custom S3 endpoint (for R2, MinIO, etc.)
            aws_region: AWS region
            language_sources: Dict mapping language code to list of source names
            tokenizer: NeMo tokenizer for encoding text
            sample_rate: Audio sample rate (default 16000)
            min_duration: Minimum sample duration in seconds
            max_duration: Maximum sample duration in seconds
            max_chars_per_sec: Maximum characters per second (filter bad transcripts)
            add_eou_token: Whether to add <eou> token after sentence-ending punctuation
            eou_token: The EOU token string
            shuffle_buffer_size: Size of shuffle buffer for randomization
            global_rank: Rank for distributed training
            world_size: World size for distributed training
            use_start_end_token: Whether to add BOS/EOS tokens
            return_sample_id: Whether to return sample IDs
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3 streaming. Install with: pip install boto3")

        if language_sources is None or len(language_sources) == 0:
            raise ValueError("language_sources must be provided")

        if tokenizer is None:
            raise ValueError("tokenizer must be provided")

        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_endpoint_url = s3_endpoint_url
        self.aws_region = aws_region
        self.language_sources = language_sources
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.shuffle_buffer_size = shuffle_buffer_size
        self.global_rank = global_rank
        self.world_size = world_size
        self.use_start_end_token = use_start_end_token
        self.return_sample_id = return_sample_id

        # Create S3 client (supports R2, MinIO, and other S3-compatible storage)
        boto_config = BotoConfig(
            region_name=aws_region,
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            signature_version='s3v4',
        )
        client_kwargs = {'config': boto_config}
        if s3_endpoint_url:
            client_kwargs['endpoint_url'] = s3_endpoint_url
        self.s3_client = boto3.client('s3', **client_kwargs)

        # Create filter
        filter_config = FilterConfig(
            min_duration=min_duration,
            max_duration=max_duration,
            max_chars_per_sec=max_chars_per_sec,
        )
        self.sample_filter = SampleFilter(filter_config)

        # Create token augmenter - but we'll use it only for checking if EOU should be added,
        # NOT for modifying the text (since SentencePiece doesn't tokenize <eou> correctly)
        self.token_augmenter = TokenAugmenter(
            eou_token=eou_token,
            add_eou=False,  # Don't modify text, we'll append ID directly
        )
        self.add_eou_token = add_eou_token
        self.eou_token = eou_token

        # Get EOU token ID from vocabulary (for direct ID appending)
        # SentencePiece doesn't recognize USER_DEFINED tokens during encoding,
        # so we must append the token ID directly after tokenization
        self.eou_token_id = None
        if add_eou_token and eou_token:
            # Use piece_to_id for accurate lookup
            if hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'piece_to_id'):
                sp = tokenizer.tokenizer
                eou_id = sp.piece_to_id(eou_token)
                if eou_id != sp.unk_id():
                    self.eou_token_id = eou_id
                    logging.info(f"EOU token '{eou_token}' found at ID {self.eou_token_id} (via piece_to_id)")
                else:
                    logging.warning(f"EOU token '{eou_token}' not found in vocabulary (piece_to_id returned unk)")
            else:
                # Fallback: search through vocabulary
                for i in range(tokenizer.vocab_size):
                    try:
                        token = tokenizer.ids_to_tokens([i])
                        if token and token[0] == eou_token:
                            self.eou_token_id = i
                            logging.info(f"EOU token '{eou_token}' found at ID {self.eou_token_id} (via search)")
                            break
                    except Exception:
                        continue

                if self.eou_token_id is None:
                    logging.warning(f"EOU token '{eou_token}' not found in vocabulary")

        # BOS/EOS tokens
        # NeMo uses -1 to indicate "no BOS/EOS", so we need to check for >= 0
        if hasattr(tokenizer, 'bos_id') and tokenizer.bos_id is not None and tokenizer.bos_id >= 0:
            self.bos_id = tokenizer.bos_id if use_start_end_token else None
        else:
            self.bos_id = None
        if hasattr(tokenizer, 'eos_id') and tokenizer.eos_id is not None and tokenizer.eos_id >= 0:
            self.eos_id = tokenizer.eos_id if use_start_end_token else None
        else:
            self.eos_id = None

        logging.info(f"BOS/EOS tokens: bos_id={self.bos_id}, eos_id={self.eos_id}")

        # Language managers will be created lazily per worker
        self._interleaver: Optional[RoundRobinInterleaver] = None

        endpoint_info = s3_endpoint_url or "AWS S3"
        logging.info(
            f"S3MultiLangStreamingDataset: bucket={s3_bucket}, "
            f"endpoint={endpoint_info}, "
            f"languages={list(language_sources.keys())}, "
            f"add_eou={add_eou_token}"
        )

    def _create_interleaver(self) -> RoundRobinInterleaver:
        """Create language managers and interleaver (called once per worker)."""
        language_managers = {}

        for lang, sources in self.language_sources.items():
            manager = LanguageSourceManager(
                lang=lang,
                s3_bucket=self.s3_bucket,
                s3_prefix=self.s3_prefix,
                sources=sources,
                s3_client=self.s3_client,
                sample_filter=self.sample_filter,
                token_augmenter=self.token_augmenter,
            )
            language_managers[lang] = manager

        return RoundRobinInterleaver(
            language_managers=language_managers,
            languages_order=sorted(self.language_sources.keys()),
        )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        """
        Iterate over training samples.

        Yields:
            Tuple of (audio, audio_len, tokens, tokens_len) or
            (audio, audio_len, tokens, tokens_len, sample_id) if return_sample_id=True
        """
        # Get worker info for multi-worker data loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # In multi-worker mode, we could shard languages across workers
            # For now, each worker gets all languages but different random seeds
            random.seed(worker_info.id + self.global_rank * 1000)

        # Create interleaver for this worker
        interleaver = self._create_interleaver()

        # Shuffle buffer
        shuffle_buffer = ShuffleBuffer(buffer_size=self.shuffle_buffer_size)

        sample_idx = 0

        # Stream samples through interleaver and shuffle buffer
        for sample in interleaver:
            shuffle_buffer.add(sample)

            # Yield when buffer is full
            while shuffle_buffer.is_ready():
                buffered_sample = shuffle_buffer.get_random()
                if buffered_sample is not None:
                    result = self._process_sample(buffered_sample, sample_idx)
                    if result is not None:
                        yield result
                        sample_idx += 1

        # Drain remaining samples from buffer
        for buffered_sample in shuffle_buffer.drain():
            result = self._process_sample(buffered_sample, sample_idx)
            if result is not None:
                yield result
                sample_idx += 1

    def _process_sample(self, sample: dict, sample_idx: int) -> Optional[Tuple]:
        """
        Process a sample into training tensors.

        Args:
            sample: Dict with 'audio' (np.array) and 'text' (str)
            sample_idx: Sample index

        Returns:
            Tuple of tensors or None if processing failed
        """
        audio = sample.get('audio')
        text = sample.get('text', '')

        if audio is None:
            return None

        # Audio tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()

        audio_len = torch.tensor(audio_tensor.shape[0]).long()

        # Check if we should add EOU token (based on sentence-ending punctuation)
        should_add_eou = (
            self.add_eou_token
            and self.eou_token_id is not None
            and self.token_augmenter.should_add_eou(text)
        )

        # Tokenize text (without <eou> - we'll append the ID directly)
        tokens = self.tokenizer.text_to_ids(text)

        # Add BOS token if configured
        if self.bos_id is not None:
            tokens = [self.bos_id] + tokens

        # Add EOU token ID directly (before EOS, after text)
        if should_add_eou:
            tokens = tokens + [self.eou_token_id]

        # Add EOS token if configured
        if self.eos_id is not None:
            tokens = tokens + [self.eos_id]

        tokens_tensor = torch.tensor(tokens).long()
        tokens_len = torch.tensor(len(tokens)).long()

        if self.return_sample_id:
            return audio_tensor, audio_len, tokens_tensor, tokens_len, sample_idx
        else:
            return audio_tensor, audio_len, tokens_tensor, tokens_len

    def collate_fn(self, batch):
        """Collate function for DataLoader."""
        return _speech_collate_fn(batch, pad_id=0)

    def __len__(self):
        """
        Approximate dataset length.

        Since we're streaming, this is an estimate based on typical shard sizes.
        """
        # Rough estimate: assume 1000 samples per source
        total_sources = sum(len(sources) for sources in self.language_sources.values())
        return total_sources * 1000

    @property
    def output_types(self):
        """Define output types for NeMo."""
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
        }
