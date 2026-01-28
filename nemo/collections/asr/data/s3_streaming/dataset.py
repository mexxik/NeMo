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

from .audio_merger import AudioMerger, MergeBuffer, MergeConfig
from .filters import FilterConfig, SampleFilter
from .lang_source_manager import LanguageSourceManager
from .round_robin import RoundRobinInterleaver
from .s3_tar_stream import S3ManifestLoader
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

        # Audio merging (for multi-utterance training with silence gaps)
        merge_utterances: bool = False,
        merge_probability: float = 0.3,
        merge_min_utterances: int = 2,
        merge_max_utterances: int = 3,
        merge_silence_min: float = 0.5,
        merge_silence_max: float = 1.5,
        merge_max_duration: float = 30.0,

        # Other
        return_sample_id: bool = False,
        samples_per_epoch: int = None,  # Override automatic length calculation
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

        # Create filter with all the same filters as 01_gather.py
        filter_config = FilterConfig(
            min_duration=min_duration,
            max_duration=max_duration,
            min_chars_per_sec=3.0,  # Only applied for duration > 2s
            max_chars_per_sec=max_chars_per_sec,
            min_chars=1,
            max_chars=500,
            validate_text=True,  # Garbage/corrupted text detection
            validate_chars=True,  # Per-language character validation
            languages=list(language_sources.keys()),  # Languages for char validation
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

        # Audio merger for multi-utterance training
        self.merge_config = MergeConfig(
            enabled=merge_utterances,
            merge_probability=merge_probability,
            min_utterances=merge_min_utterances,
            max_utterances=merge_max_utterances,
            silence_min_sec=merge_silence_min,
            silence_max_sec=merge_silence_max,
            max_merged_duration=merge_max_duration,
        )
        self.audio_merger = AudioMerger(self.merge_config, sample_rate=sample_rate)

        # Store samples_per_epoch for __len__
        # If not provided, auto-calculate from manifest counts
        if samples_per_epoch is not None:
            self.samples_per_epoch = samples_per_epoch
            logging.info(f"Using provided samples_per_epoch: {samples_per_epoch}")
        else:
            self.samples_per_epoch = self._calculate_total_samples()
            if self.samples_per_epoch > 0:
                logging.info(f"Auto-calculated samples_per_epoch from manifests: {self.samples_per_epoch}")
            else:
                # Fallback to default estimate if manifest loading failed
                total_sources = sum(len(sources) for sources in language_sources.values())
                self.samples_per_epoch = total_sources * 100000
                logging.warning(
                    f"Could not load manifests for counting, using estimate: {self.samples_per_epoch}"
                )

        endpoint_info = s3_endpoint_url or "AWS S3"
        logging.info(
            f"S3MultiLangStreamingDataset: bucket={s3_bucket}, "
            f"endpoint={endpoint_info}, "
            f"languages={list(language_sources.keys())}, "
            f"add_eou={add_eou_token}, "
            f"merge_utterances={merge_utterances}"
        )

    def _calculate_total_samples(self) -> int:
        """
        Calculate total samples by counting manifest entries.

        This loads each manifest once to get the accurate count.
        The manifests will be loaded again by workers during iteration,
        but this gives us the correct epoch length upfront.

        Returns:
            Total number of samples across all language sources
        """
        manifest_loader = S3ManifestLoader(s3_client=self.s3_client)
        total = 0

        for lang, sources in self.language_sources.items():
            for source in sources:
                source_prefix = f"{self.s3_prefix.rstrip('/') + '/' if self.s3_prefix else ''}{source}/"
                manifest_key = f"{source_prefix}tarred_audio_manifest.json"

                try:
                    count = manifest_loader.count_manifest_entries(
                        self.s3_bucket, manifest_key
                    )
                    total += count
                    logging.info(f"[{lang}] {source}: {count} manifest entries")
                except Exception as e:
                    logging.warning(f"[{lang}] Failed to count manifest for {source}: {e}")

        return total

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

        # Merge buffer for multi-utterance merging
        merge_buffer = MergeBuffer(self.audio_merger, self.merge_config)

        sample_idx = 0

        def process_and_yield(sample):
            """Process sample through merge buffer and yield results."""
            nonlocal sample_idx

            # Try to add to merge buffer - may return merged sample, single sample, or None
            result_sample = merge_buffer.add(sample)
            if result_sample is not None:
                result = self._process_sample(result_sample, sample_idx)
                if result is not None:
                    sample_idx += 1

                    # Log merge stats periodically
                    if sample_idx % 10000 == 0 and self.merge_config.enabled:
                        self.audio_merger.log_stats()

                    return result
            return None

        # Stream samples through interleaver and shuffle buffer
        for sample in interleaver:
            shuffle_buffer.add(sample)

            # Yield when buffer is full
            while shuffle_buffer.is_ready():
                buffered_sample = shuffle_buffer.get_random()
                if buffered_sample is not None:
                    result = process_and_yield(buffered_sample)
                    if result is not None:
                        yield result

        # Drain remaining samples from shuffle buffer
        for buffered_sample in shuffle_buffer.drain():
            result = process_and_yield(buffered_sample)
            if result is not None:
                yield result

        # Drain remaining samples from merge buffer
        for remaining_sample in merge_buffer.drain():
            result = self._process_sample(remaining_sample, sample_idx)
            if result is not None:
                yield result
                sample_idx += 1

    def _process_sample(self, sample: dict, sample_idx: int) -> Optional[Tuple]:
        """
        Process a sample into training tensors.

        Handles both regular samples and merged multi-utterance samples.

        Args:
            sample: Dict with 'audio' (np.array) and 'text' (str)
                   For merged samples, also has 'original_texts' and 'eou_positions'
            sample_idx: Sample index

        Returns:
            Tuple of tensors or None if processing failed
        """
        audio = sample.get('audio')

        if audio is None:
            return None

        # Audio tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()

        audio_len = torch.tensor(audio_tensor.shape[0]).long()

        # Check if this is a merged sample
        original_texts = sample.get('original_texts')
        eou_positions = sample.get('eou_positions', [])

        if original_texts is not None:
            # Merged sample: tokenize each segment and add EOU where appropriate
            tokens = self._tokenize_merged(original_texts, eou_positions)
        else:
            # Regular sample
            text = sample.get('text', '')

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

    def _tokenize_merged(self, texts: List[str], eou_positions: List[int]) -> List[int]:
        """
        Tokenize merged multi-utterance sample.

        Args:
            texts: List of text segments
            eou_positions: Indices of segments that should have EOU appended

        Returns:
            Combined token list with EOU tokens inserted appropriately
        """
        tokens = []

        # Add BOS token if configured (only at the very beginning)
        if self.bos_id is not None:
            tokens.append(self.bos_id)

        for i, text in enumerate(texts):
            # Tokenize this segment
            segment_tokens = self.tokenizer.text_to_ids(text)
            tokens.extend(segment_tokens)

            # Add EOU after this segment if it's in eou_positions
            if self.add_eou_token and self.eou_token_id is not None and i in eou_positions:
                tokens.append(self.eou_token_id)

        # Add EOS token if configured (only at the very end)
        if self.eos_id is not None:
            tokens.append(self.eos_id)

        return tokens

    def collate_fn(self, batch):
        """Collate function for DataLoader."""
        return _speech_collate_fn(batch, pad_id=0)

    def __len__(self):
        """
        Dataset length for epoch calculation.

        Returns samples_per_epoch, which is either:
        - Explicitly provided by user
        - Auto-calculated from manifest entry counts
        - Fallback estimate if manifest loading failed
        """
        return self.samples_per_epoch

    @property
    def output_types(self):
        """Define output types for NeMo."""
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
        }
