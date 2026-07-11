"""
Multi-Language Streaming Dataset for ASR training.

Main dataset class that integrates all streaming components.
Supports both S3 and local disk storage.
"""

import multiprocessing as mp
import os
import random
import re
import unicodedata
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from nemo.collections.asr.data.audio_to_text import _speech_collate_fn
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging

from .audio_merger import AudioMerger, MergeBuffer, MergeConfig
from .augmentation import AudioAugmentor, AugmentationConfig
from .disk_tar_stream import DiskManifestLoader
from .filters import FilterConfig, SampleFilter
from .lang_source_manager import LanguageSourceManager, SingleSourceManager
from .round_robin import RoundRobinInterleaver, SourceRoundRobinInterleaver
from .s3_tar_stream import S3ManifestLoader
from .sqlite_manifest import SQLiteManifestCache, get_default_cache_dir
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
        if self.buffer_size == 0:
            # Pass-through mode: ready if we have any items (no buffering)
            return len(self.buffer) > 0
        return len(self.buffer) >= self.buffer_size

    def get_random(self):
        """Get and remove a random item from the buffer."""
        if not self.buffer:
            return None
        if self.buffer_size == 0:
            # Pass-through mode: just pop (no randomness needed)
            return self.buffer.pop()
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
    Streaming dataset that loads audio from TAR files across multiple languages.

    Supports both S3 and local disk storage:
    - S3: Direct streaming from S3-compatible storage (AWS, R2, MinIO)
    - Disk: Streaming from local disk storage

    Features:
    - Direct streaming (no pre-download)
    - Round-robin language sampling
    - On-the-fly filtering (duration, char rate)
    - Token augmentation (<eou>)
    - Shuffle buffer for local randomization
    - Distributed training support

    Config example (S3):
        model:
          train_ds:
            dataset_type: "s3_multilang_streaming"
            s3_bucket: "my-asr-bucket"
            s3_prefix: "asr-data/"
            language_sources:
              en: [common_en_train, yodas_en_en000]
              uk: [common_uk_train]

    Config example (Disk):
        model:
          train_ds:
            dataset_type: "s3_multilang_streaming"
            data_root: "/media/storage/asr-data"
            language_sources:
              en: [common_en_train, yodas_en_en000]
              uk: [common_uk_train]
    """

    def __init__(
        self,
        # Storage config - use EITHER S3 OR disk
        # S3 config
        s3_bucket: str = None,
        s3_prefix: str = "",
        s3_endpoint_url: str = None,  # For R2, MinIO, etc.
        aws_region: str = "us-east-1",
        # Disk config
        data_root: str = None,  # Local path like "/media/storage/asr-data"
        # Layout: if True, sources resolve to <data_root>/<lang>/<source>
        # instead of the flat <data_root>/<source>. Applies to both disk
        # paths and (for s3) the <s3_prefix><lang>/<source>/ key prefix.
        lang_subdir: bool = False,

        # Language config
        language_sources: Dict[str, List[str]] = None,

        # Tokenizer
        tokenizer=None,
        sample_rate: int = 16000,

        # Filtering
        min_duration: float = 0.5,
        max_duration: float = 15.0,
        max_chars: int = 50000,
        filter_whisper_hallucinations: bool = True,

        # Token augmentation
        add_eou_token: bool = True,
        eou_token: str = "<eou>",
        add_eot_token: bool = False,
        eot_token: str = "<eot>",
        # Per-language tag at the END of an utterance (e.g. "...text <uk>").
        # Pieces are looked up as f"<{lang}>" in the SentencePiece vocab.
        # When add_lang_token=True, lang_token_prob controls how often the tag
        # is inserted (1.0 = always, lets you read the lang back from decoded
        # output reliably; <1.0 = sometimes, model handles tagged + untagged).
        add_lang_token: bool = False,
        lang_token_prob: float = 1.0,
        # Additionally inject the lang tag mid-text, immediately after the N-th
        # word (0 = disabled; only emit at end of utterance).  Trains the model
        # to commit to a language as early as ~2 spoken words instead of waiting
        # for <eou>. Has no effect unless add_lang_token is also True, and only
        # fires for samples/segments whose word count exceeds N.
        early_lang_token_after_word: int = 0,

        # Text normalization: "none", "full" (lowercase + remove all punctuation), "soft" (keep casing + keep only . , ! ?)
        normalize_text = "none",

        # When True, drop any utterance that isn't fully punctuated AND
        # capitalized (ends with . ! ? and every sentence starts uppercase).
        # Enforced at ingestion (in SampleFilter) so both single and merged
        # samples stay clean. Pairs well with normalize_text="soft".
        require_strict_pnc: bool = False,

        # Shuffle buffer (0 = no buffering, stream directly)
        shuffle_buffer_size: int = 0,

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

        # Selective backprop: names of sources whose transcripts are "clean"
        # (properly punctuated/capitalized). Samples from clean sources are
        # tagged sample['clean']=True; everything else is tagged False. The
        # training loop uses this flag to update the full network on clean
        # batches but only the encoder on noisy batches (see grad-gating in
        # speech_to_text_s3_finetune.py). None or empty => feature off: every
        # sample is treated as clean (standard full-network training).
        clean_sources: Optional[List[str]] = None,

        # Selective backprop, PnC-gated variant. When True, the clean/noisy flag
        # is computed PER SAMPLE from is_strict_pnc(text) instead of per-source
        # membership: a fully punctuated + capitalized utterance is tagged
        # clean=True (updates the full network), anything else clean=False
        # (encoder-only update). Unlike `require_strict_pnc` (a hard filter that
        # DROPS non-PnC samples), this keeps every sample — non-PnC audio still
        # trains the encoder. Mutually exclusive with `require_strict_pnc`
        # (filter vs gate) and with `clean_sources` (two clean criteria).
        clean_pnc_gate: bool = False,

        # Other
        return_sample_id: bool = False,
        samples_per_epoch: int = None,  # Override automatic length calculation

        # Memory optimization
        use_sqlite_cache: bool = True,  # Use SQLite for manifest caching (saves RAM with multiple workers)
        sqlite_cache_dir: str = None,  # Custom cache directory (default: ~/.cache/nemo_manifest_cache)
        sqlite_cache_path: str = None,  # Explicit cache file path (overrides auto-generated name)

        # Rotation mode
        rotation_level: str = "source",  # "source" or "language"
        samples_per_source: int = 50,  # Samples to get from each source before rotating (reduces parallel TAR streams)

        # Prefetch (background sample loading)
        prefetch_buffer_size: int = 0,  # Number of samples to prefetch per source (0 to disable)

        # Source balancing mode: "none", "max", "min", "distributed", "cap"
        balance_mode: str = "max",
        # Per-lang hours cap for balance_mode="cap". Each lang is sampled to
        # exactly this many hours per epoch: oversampled if natural < cap,
        # undersampled if natural > cap. Required when balance_mode="cap".
        balance_target_hours: Optional[float] = None,
        augment_prob: float = 0.1,  # Base probability of augmentation for non-repeated samples
        augment_speed: bool = False,  # Enable speed perturbation (0.9x-1.1x)
        augment_gain: bool = False,  # Enable gain adjustment (+/-6dB)
        augment_time_mask: bool = True,  # Enable time masking (zero out random segments)
        augment_bandwidth: bool = False,  # Enable bandwidth limitation (8kHz telephone)
        augment_noise: bool = False,  # Enable additive white noise

        # Curriculum bucketing: process filterN_ sources from shortest to longest
        curriculum_buckets: bool = False,

        # Language-ID training mode. When True, the label sequence for each
        # utterance is built from per-word lang tokens. EOU/EOT/BOS/EOS/
        # text-normalization are skipped — they're irrelevant for LID. Combined
        # with merge_utterances + cross-language sample rotation, this produces
        # code-switch training samples.
        lid_mode: bool = False,
        # Label scheme for lid_mode:
        #   "word"     -> `<lang>` × num_words   (token-per-word, current default)
        #   "syllable" -> `<lang_S> <syl>×N <lang_E>` per word, N = syllable count
        #                 from a vowel-group heuristic. Gives word boundaries +
        #                 syllable-rate anchoring (useful for VAD + alignment).
        #   "romanize" -> `<lang_S>` + BPE(romanized text) + `<lang_E>` per segment.
        #                 Dense content target (real transliterated ASR); the
        #                 boundaries carry the LID label. Needs a tokenizer built
        #                 with --label-scheme romanize (real BPE over romanized text).
        #   "romanize_word" -> per word: BPE(romanized word) + `<lang>`. The tag
        #                 trails each word so LID is committed after the word is
        #                 heard (authoritative per word, no provisional opening
        #                 tag). Uses the single `<lang>` tokens like "word"; needs
        #                 a tokenizer built with --label-scheme romanize_word.
        lid_label_scheme: str = "word",
    ):
        """
        Initialize multi-language streaming dataset.

        Args:
            s3_bucket: S3 bucket name (for S3 storage)
            s3_prefix: Prefix path in bucket (e.g., "asr-data/")
            s3_endpoint_url: Custom S3 endpoint (for R2, MinIO, etc.)
            aws_region: AWS region
            data_root: Local path for disk storage (e.g., "/media/storage/asr-data")
            language_sources: Dict mapping language code to list of source names
            tokenizer: NeMo tokenizer for encoding text
            sample_rate: Audio sample rate (default 16000)
            min_duration: Minimum sample duration in seconds
            max_duration: Maximum sample duration in seconds
            add_eou_token: Whether to add <eou> token after sentence-ending punctuation
            eou_token: The EOU token string
            shuffle_buffer_size: Size of shuffle buffer for randomization
            global_rank: Rank for distributed training
            world_size: World size for distributed training
            use_start_end_token: Whether to add BOS/EOS tokens
            return_sample_id: Whether to return sample IDs
            samples_per_epoch: Override automatic length calculation
            rotation_level: How to rotate through data:
                - "source": Round-robin across individual datasets/sources (default)
                - "language": Round-robin across languages
        """
        if language_sources is None or len(language_sources) == 0:
            raise ValueError("language_sources must be provided")

        if tokenizer is None:
            raise ValueError("tokenizer must be provided")

        if rotation_level not in ("language", "source"):
            raise ValueError(f"rotation_level must be 'language' or 'source', got: {rotation_level}")
        self.rotation_level = rotation_level
        self.samples_per_source = samples_per_source
        self.prefetch_buffer_size = prefetch_buffer_size

        # Balance mode: backward compat for old balance_sources bool
        if balance_mode is True:
            balance_mode = "max"
        elif balance_mode is False:
            balance_mode = "none"
        # "sync": budget/oversample exactly like "max", but end the epoch the
        # instant any language drains (language interleaver only). Normalize to
        # "max" here so all balance/budget logic below is unchanged; the drain
        # behavior is carried by self._stop_on_lang_exhaustion.
        self._stop_on_lang_exhaustion = (balance_mode == "sync")
        if balance_mode == "sync":
            if rotation_level != "language":
                logging.warning(
                    "balance_mode='sync' only changes behavior with "
                    "rotation_level='language'; with 'source' it behaves as 'max'."
                )
            balance_mode = "max"
        if balance_mode not in ("none", "max", "min", "distributed", "cap"):
            raise ValueError(
                f"balance_mode must be 'none', 'max', 'min', 'distributed', 'cap', or 'sync', "
                f"got: {balance_mode}"
            )
        if balance_mode == "cap" and (balance_target_hours is None or balance_target_hours <= 0):
            raise ValueError(
                "balance_mode='cap' requires balance_target_hours > 0 (hours per language per epoch)"
            )
        if balance_target_hours is not None and balance_target_hours > 0 \
                and balance_mode not in ("max", "cap"):
            logging.warning(
                f"balance_target_hours={balance_target_hours} is ignored for "
                f"balance_mode='{balance_mode}'. Only 'max' and 'cap' modes use it."
            )
        self.balance_mode = balance_mode
        self.balance_target_hours = balance_target_hours

        # Audio augmentation
        augment_any = augment_speed or augment_gain or augment_time_mask or augment_bandwidth or augment_noise
        augment_config = AugmentationConfig(
            enabled=augment_any,
            base_prob=augment_prob,
            speed_enabled=augment_speed,
            gain_enabled=augment_gain,
            time_mask_enabled=augment_time_mask,
            bandwidth_enabled=augment_bandwidth,
            noise_enabled=augment_noise,
        )
        self.augmentor = AudioAugmentor(augment_config)
        if augment_any:
            logging.info(
                f"Augmentation enabled: prob={augment_prob}, "
                f"speed={augment_speed}, gain={augment_gain}, time_mask={augment_time_mask}, "
                f"bandwidth={augment_bandwidth}, noise={augment_noise}"
            )
        self.curriculum_buckets = curriculum_buckets
        if curriculum_buckets:
            logging.info(f"Curriculum bucketing: ENABLED (will process filterN_ sources short→long)")
        logging.info(f"Balance mode: {balance_mode}")

        # Determine storage type
        if data_root is not None:
            self.storage_type = "disk"
            self.data_root = data_root
            self.s3_bucket = None
            self.s3_prefix = None
            self.s3_endpoint_url = None
            self.s3_client = None
            logging.info(f"[DATASET_INIT] Using DISK storage: {data_root}")
            if not os.path.isdir(data_root):
                raise ValueError(f"data_root directory not found: {data_root}")
        elif s3_bucket is not None:
            if not BOTO3_AVAILABLE:
                raise ImportError("boto3 is required for S3 streaming. Install with: pip install boto3")
            self.storage_type = "s3"
            self.data_root = None
            self.s3_bucket = s3_bucket
            self.s3_prefix = s3_prefix
            self.s3_endpoint_url = s3_endpoint_url
            # Create S3 client
            boto_config = BotoConfig(
                region_name=aws_region,
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                signature_version='s3v4',
            )
            client_kwargs = {'config': boto_config}
            if s3_endpoint_url:
                client_kwargs['endpoint_url'] = s3_endpoint_url
            self.s3_client = boto3.client('s3', **client_kwargs)
        else:
            raise ValueError("Either s3_bucket or data_root must be provided")

        self.aws_region = aws_region
        self.lang_subdir = bool(lang_subdir)
        if self.lang_subdir:
            logging.info("[DATASET_INIT] lang_subdir=True (sources resolve to <root>/<lang>/<source>)")
        self.language_sources = language_sources
        # Selective-backprop clean-source set. Empty/None => feature off (all
        # samples treated as clean). Stored as a set for O(1) membership checks
        # in the per-source managers.
        self.clean_sources = set(clean_sources) if clean_sources else None
        if self.clean_sources:
            logging.info(
                f"[DATASET_INIT] Selective backprop enabled — clean sources: "
                f"{sorted(self.clean_sources)} (all other sources tagged noisy)"
            )
        # PnC-gated selective backprop: clean flag derived per-sample from
        # is_strict_pnc(text). See the constructor arg docstring above.
        self.clean_pnc_gate = bool(clean_pnc_gate)
        if self.clean_pnc_gate:
            if self.clean_sources:
                raise ValueError(
                    "clean_pnc_gate and clean_sources are both set — pick one "
                    "clean/noisy criterion (per-sample PnC OR per-source list)."
                )
            if require_strict_pnc:
                raise ValueError(
                    "clean_pnc_gate and require_strict_pnc are both set — "
                    "require_strict_pnc DROPS non-PnC samples, leaving nothing "
                    "to gate. Use one or the other."
                )
            logging.info(
                "[DATASET_INIT] Selective backprop enabled — PnC-gated: "
                "per-sample clean flag = is_strict_pnc(text); non-PnC samples "
                "kept but update the encoder only."
            )
        # Per-yield cleanliness side-channel, mirrors `_cur_yield_weight`. Set by
        # `_process_sample` for the sample about to be yielded; read by the
        # dynamic batcher right after each pull to group homogeneous batches.
        self._cur_yield_clean = True
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.shuffle_buffer_size = shuffle_buffer_size
        self.global_rank = global_rank
        self.world_size = world_size
        self.use_start_end_token = use_start_end_token
        self.return_sample_id = return_sample_id

        # Create filter (duration, text length, and whisper hallucination detection)
        filter_config = FilterConfig(
            min_duration=min_duration,
            max_duration=max_duration,
            min_chars=1,
            max_chars=max_chars,
            filter_whisper_hallucinations=filter_whisper_hallucinations,
            require_strict_pnc=require_strict_pnc,
        )
        self.sample_filter = SampleFilter(filter_config)
        if require_strict_pnc:
            logging.info("Strict PnC filter ENABLED: dropping utterances that "
                         "aren't fully punctuated and capitalized")

        # Create token augmenter - but we'll use it only for checking if EOU should be added,
        # NOT for modifying the text (since SentencePiece doesn't tokenize <eou> correctly)
        self.token_augmenter = TokenAugmenter(
            eou_token=eou_token,
            add_eou=False,  # Don't modify text, we'll append ID directly
        )
        self.add_eou_token = add_eou_token
        self.eou_token = eou_token

        # Text normalization mode: "none", "full", "soft"
        # Backward compat: bool -> str
        if normalize_text is True:
            normalize_text = "full"
        elif normalize_text is False:
            normalize_text = "none"
        if normalize_text not in ("none", "full", "soft"):
            raise ValueError(f"normalize_text must be 'none', 'full', or 'soft', got: {normalize_text}")
        self.normalize_text = normalize_text
        if normalize_text != "none":
            logging.info(f"Text normalization: {normalize_text}")

        # Resolve the tokenizer's <unk> id. With byte_fallback enabled the
        # encoder *should* fall back to UTF-8 bytes for unseen chars, so an
        # <unk> in the encoded sequence means some character even byte fallback
        # couldn't handle. Training on UNKs gives the model no signal (the
        # acoustic frame has no consistent referent), so we drop such samples
        # in _process_sample below. None means we can't detect → no filter.
        self._unk_id = None
        if hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'unk_id'):
            try:
                _uid = int(tokenizer.tokenizer.unk_id())
                if _uid >= 0:
                    self._unk_id = _uid
                    logging.info(
                        f"Tokenizer <unk> id = {self._unk_id} "
                        f"(samples that tokenize to any UNK will be dropped)"
                    )
            except Exception as e:
                logging.warning(f"Could not resolve tokenizer unk_id: {e}")

        # Per-character UNK cache for _clean_unk_chars. Maps a character to
        # whether the tokenizer encodes it to <unk> (no token, no byte_fallback).
        # Populated lazily; bounded by the alphabet size of the corpus.
        self._unk_char_cache = {}

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

        # Get EOT token ID from vocabulary (same approach as EOU)
        self.add_eot_token = add_eot_token
        self.eot_token = eot_token
        self.eot_token_id = None
        if add_eot_token and eot_token:
            if hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'piece_to_id'):
                sp = tokenizer.tokenizer
                eot_id = sp.piece_to_id(eot_token)
                if eot_id != sp.unk_id():
                    self.eot_token_id = eot_id
                    logging.info(f"EOT token '{eot_token}' found at ID {self.eot_token_id} (via piece_to_id)")
                else:
                    logging.warning(f"EOT token '{eot_token}' not found in vocabulary (piece_to_id returned unk)")
            else:
                for i in range(tokenizer.vocab_size):
                    try:
                        token = tokenizer.ids_to_tokens([i])
                        if token and token[0] == eot_token:
                            self.eot_token_id = i
                            logging.info(f"EOT token '{eot_token}' found at ID {self.eot_token_id} (via search)")
                            break
                    except Exception:
                        continue

                if self.eot_token_id is None:
                    logging.warning(f"EOT token '{eot_token}' not found in vocabulary")

        # LID mode: force lang-token vocab lookup, disable text/normalization paths.
        self.lid_mode = bool(lid_mode)
        self.lid_label_scheme = lid_label_scheme if lid_label_scheme in ("word", "syllable", "romanize", "romanize_word") else "word"
        if self.lid_mode:
            add_lang_token = True
            add_eou_token = False
            add_eot_token = False
            use_start_end_token = False
            normalize_text = "none"
            logging.info(
                "[DATASET_INIT] LID mode ENABLED: label_scheme=%s, EOU/EOT/BOS/EOS/text-norm disabled",
                self.lid_label_scheme,
            )

        # Per-language tag tokens. Looked up once from the SentencePiece vocab
        # so we can append the corresponding ID directly during tokenization
        # (SentencePiece doesn't tokenize USER_DEFINED tokens during encoding).
        self.add_lang_token = add_lang_token
        self.lang_token_prob = float(lang_token_prob)
        self.early_lang_token_after_word = max(0, int(early_lang_token_after_word))
        self.lang_token_ids: Dict[str, int] = {}
        if add_lang_token:
            sp = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else None
            for lang in language_sources.keys():
                token_str = f"<{lang}>"
                tid = None
                if sp is not None and hasattr(sp, 'piece_to_id'):
                    candidate = sp.piece_to_id(token_str)
                    if candidate != sp.unk_id():
                        tid = candidate
                if tid is None:
                    # Fallback: linear scan
                    for i in range(tokenizer.vocab_size):
                        try:
                            piece = tokenizer.ids_to_tokens([i])
                            if piece and piece[0] == token_str:
                                tid = i
                                break
                        except Exception:
                            continue
                if tid is not None:
                    self.lang_token_ids[lang] = tid
                    logging.info(f"Lang token '{token_str}' found at ID {tid}")
                else:
                    logging.warning(
                        f"Lang token '{token_str}' not found in vocabulary — "
                        f"samples for lang='{lang}' will not be tagged"
                    )

        # Boundary-scheme token IDs: per-language `<lang_S>`, `<lang_E>`.
        # The syllable scheme additionally needs a generic `<syl>` marker; the
        # romanize scheme puts BPE'd romanized text between the boundaries.
        self.lang_start_ids: Dict[str, int] = {}
        self.lang_end_ids: Dict[str, int] = {}
        self.syllable_token_id: Optional[int] = None
        if self.lid_mode and self.lid_label_scheme in ("syllable", "romanize"):
            sp = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else None

            def _lookup(piece_str):
                if sp is not None and hasattr(sp, 'piece_to_id'):
                    cand = sp.piece_to_id(piece_str)
                    if cand != sp.unk_id():
                        return cand
                return None

            if self.lid_label_scheme == "syllable":
                self.syllable_token_id = _lookup("<syl>")
                if self.syllable_token_id is None:
                    raise ValueError(
                        "lid_label_scheme='syllable' requires the tokenizer to contain `<syl>`. "
                        "Rebuild the tokenizer with --label-scheme syllable."
                    )
                logging.info(f"Syllable token '<syl>' found at ID {self.syllable_token_id}")

            for lang in language_sources.keys():
                s_id = _lookup(f"<{lang}_S>")
                e_id = _lookup(f"<{lang}_E>")
                if s_id is None or e_id is None:
                    raise ValueError(
                        f"lid_label_scheme='{self.lid_label_scheme}' requires `<{lang}_S>` and "
                        f"`<{lang}_E>` in the tokenizer (got s_id={s_id}, e_id={e_id}). "
                        f"Rebuild the tokenizer with --label-scheme {self.lid_label_scheme}."
                    )
                self.lang_start_ids[lang] = s_id
                self.lang_end_ids[lang] = e_id
                logging.info(f"Lang boundary tokens: <{lang}_S>={s_id}, <{lang}_E>={e_id}")

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

        # SQLite manifest cache for memory-efficient multi-worker loading
        self.use_sqlite_cache = use_sqlite_cache
        self._sqlite_cache_path: Optional[str] = None
        self._explicit_cache_path = False  # True if user provided cache path (trust it, don't rebuild)
        if use_sqlite_cache:
            if sqlite_cache_path:
                # Use explicit cache path (allows train/validation to share same cache)
                self._sqlite_cache_path = sqlite_cache_path
                self._explicit_cache_path = True  # User provided - trust it completely
                cache_dir = os.path.dirname(sqlite_cache_path)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                logging.info(f"Using explicit SQLite cache path: {sqlite_cache_path}")
            else:
                # Auto-generate cache name from source names
                cache_dir = sqlite_cache_dir or get_default_cache_dir()
                # Create cache name from sorted source names (path-independent)
                all_sources = sorted(
                    source for sources in language_sources.values() for source in sources
                )
                cache_name = "_".join(all_sources)
                # Truncate if too long, but keep it readable
                if len(cache_name) > 100:
                    import hashlib
                    cache_name = cache_name[:60] + "_" + hashlib.md5(cache_name.encode()).hexdigest()[:8]
                self._sqlite_cache_path = os.path.join(cache_dir, f"manifest_{cache_name}.db")
                os.makedirs(cache_dir, exist_ok=True)

        # Audio merger for multi-utterance training.
        # In LID mode the punctuation gate is irrelevant — disable it so cross-
        # language merges go through regardless of where utterances end.
        self.merge_config = MergeConfig(
            enabled=merge_utterances,
            merge_probability=merge_probability,
            min_utterances=merge_min_utterances,
            max_utterances=merge_max_utterances,
            silence_min_sec=merge_silence_min,
            silence_max_sec=merge_silence_max,
            max_merged_duration=merge_max_duration,
            require_punctuation=not self.lid_mode,
            require_starts_capital=not self.lid_mode,
        )
        self.audio_merger = AudioMerger(self.merge_config, sample_rate=sample_rate)

        # Shared counter of raw inputs pulled from the interleaver this epoch.
        # Workers (DataLoader fork) atomically bump it; the main process resets
        # it per epoch and reads it from SampleProgressBar. With merging on,
        # multiple raw inputs collapse into one yielded sample, so counting
        # yielded samples undercounts progress; this counter matches
        # samples_per_epoch exactly when the interleaver is drained.
        self._raw_consumed = mp.Value('q', 0)
        # Bar counter: incremented by the dynamic batcher when it EMITS a batch,
        # weighted by the raw pulls each emitted sample represents. Unlike
        # _raw_consumed (pull-time, used for the rank-budget gate), this tracks
        # consumer-side progress, so the progress bar reaches 100% at the actual
        # epoch end rather than a full sort-buffer + merge-buffer ahead of it.
        self._raw_emitted = mp.Value('q', 0)
        # Per-yield weight side-channel. The dynamic batcher drives this
        # generator single-threaded within each worker, so it can read this
        # right after each pull. Value = raw pulls since the previous yield
        # (captures merges and dropped/filtered samples).
        self._cur_yield_weight = 0
        # Epoch/pass generation, bumped by the trainer callback at the start of
        # each train epoch and validation pass. The first worker to enter
        # __iter__ for a new generation zeroes the shared gate/bar counters,
        # under the counter lock (race-free). This is immune to async prefetch /
        # persistent_workers residue that a main-process reset cannot reliably
        # synchronize against — the root cause of validation under-covering.
        self._epoch_gen = mp.Value('q', 0)
        self._counter_gen = mp.Value('q', -1)

        # Track whether samples_per_epoch was explicitly provided by user
        self._explicit_samples_per_epoch = samples_per_epoch is not None

        # Store samples_per_epoch for __len__
        # If not provided, try to get from existing cache, else calculate
        if samples_per_epoch is not None:
            self.samples_per_epoch = samples_per_epoch
            logging.info(f"Using provided samples_per_epoch: {samples_per_epoch}")
        elif self._sqlite_cache_path and os.path.exists(self._sqlite_cache_path):
            # Cache exists - just read count from it, no validation, no downloading
            try:
                from .sqlite_manifest import SQLiteManifestCache
                cache = SQLiteManifestCache(self._sqlite_cache_path, read_only=True)

                # Count only entries for configured sources, not all cache entries
                all_sources = []
                for lang_sources in language_sources.values():
                    all_sources.extend(lang_sources)

                total_count = 0
                for source in all_sources:
                    total_count += cache.count_entries(source=source)

                self.samples_per_epoch = total_count
                cache.close()
                logging.info(f"Using existing SQLite cache: {self._sqlite_cache_path}")
                logging.info(f"Samples per epoch from cache: {self.samples_per_epoch}")
                logging.info("SKIPPING cache validation - trusting existing cache")
            except Exception as e:
                logging.warning(f"Failed to read cache, will calculate: {e}")
                self.samples_per_epoch = self._calculate_total_samples()
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

        # Count total sources for logging
        total_sources = sum(len(sources) for sources in language_sources.values())

        if self.storage_type == "s3":
            endpoint_info = s3_endpoint_url or "AWS S3"
            logging.info(
                f"MultiLangStreamingDataset (S3): bucket={s3_bucket}, "
                f"endpoint={endpoint_info}, "
                f"languages={list(language_sources.keys())}, "
                f"sources={total_sources}, "
                f"rotation_level={rotation_level}, "
                f"add_eou={add_eou_token}, "
                f"add_eot={add_eot_token}, "
                f"normalize_text={normalize_text}, "
                f"merge_utterances={merge_utterances}"
            )
        else:
            logging.info(
                f"MultiLangStreamingDataset (Disk): data_root={data_root}, "
                f"languages={list(language_sources.keys())}, "
                f"sources={total_sources}, "
                f"rotation_level={rotation_level}, "
                f"add_eou={add_eou_token}, "
                f"add_eot={add_eot_token}, "
                f"normalize_text={normalize_text}, "
                f"merge_utterances={merge_utterances}"
            )

        # Compute per-language and per-source sample counts (needed for min/distributed modes)
        self._lang_sample_counts: Optional[Dict[str, int]] = None
        self._source_sample_counts: Optional[Dict[str, int]] = None
        self._lang_hours: Optional[Dict[str, float]] = None
        if self._sqlite_cache_path and os.path.exists(self._sqlite_cache_path):
            self._lang_sample_counts, self._source_sample_counts = self._get_sample_counts()
            self._lang_hours = self._get_lang_hours()

        # Log balancing strategy if enabled
        if self.balance_mode != "none" and self._sqlite_cache_path and os.path.exists(self._sqlite_cache_path):
            self._log_balancing_strategy()

    def _get_lang_hours(self) -> Optional[Dict[str, float]]:
        """Get per-language total duration in hours, restricted to the sources
        listed in language_sources (so cache entries from other configs don't
        inflate the numbers)."""
        if not self._sqlite_cache_path or not os.path.exists(self._sqlite_cache_path):
            return None
        try:
            from .sqlite_manifest import SQLiteManifestCache
            cache = SQLiteManifestCache(self._sqlite_cache_path, read_only=True)
            conn = cache._get_connection()
            out: Dict[str, float] = {}
            for lang, sources in self.language_sources.items():
                total_seconds = 0.0
                for source in sources:
                    row = conn.execute(
                        "SELECT COALESCE(SUM(duration), 0) AS s FROM manifest_entries WHERE source = ?",
                        (source,),
                    ).fetchone()
                    total_seconds += float(row["s"] if row else 0.0)
                out[lang] = total_seconds / 3600.0
            cache.close()
            return out
        except Exception as e:
            logging.warning(f"Could not get per-lang hours from cache: {e}")
            return None

    def _get_sample_counts(self):
        """Get per-language and per-source sample counts from SQLite cache."""
        if not self._sqlite_cache_path or not os.path.exists(self._sqlite_cache_path):
            return None, None
        try:
            from .sqlite_manifest import SQLiteManifestCache
            cache = SQLiteManifestCache(self._sqlite_cache_path, read_only=True)
            conn = cache._get_connection()

            lang_counts: Dict[str, int] = {}
            source_counts: Dict[str, int] = {}  # "lang:source" -> count

            for lang, sources in self.language_sources.items():
                lang_total = 0
                for source in sources:
                    row = conn.execute(
                        "SELECT COUNT(*) as cnt FROM manifest_entries WHERE source = ?",
                        (source,)
                    ).fetchone()
                    count = row['cnt'] if row else 0
                    lang_total += count
                    source_counts[f"{lang}:{source}"] = count
                lang_counts[lang] = lang_total

            cache.close()
            return lang_counts, source_counts
        except Exception as e:
            logging.warning(f"Could not get sample counts: {e}")
            return None, None

    def _log_balancing_strategy(self):
        """Log the source balancing strategy based on duration statistics."""
        try:
            from .sqlite_manifest import SQLiteManifestCache
            cache = SQLiteManifestCache(self._sqlite_cache_path, read_only=True)
            conn = cache._get_connection()

            # Get hours per language
            lang_rows = conn.execute("""
                SELECT lang, SUM(duration) / 3600.0 as hours, COUNT(*) as samples
                FROM manifest_entries
                GROUP BY lang
                ORDER BY hours DESC
            """).fetchall()
            cache.close()

            if not lang_rows:
                return

            lang_hours = {row['lang']: row['hours'] for row in lang_rows}
            lang_samples = {row['lang']: row['samples'] for row in lang_rows}
            total_hours = sum(lang_hours.values())
            total_samples = sum(lang_samples.values())
            num_languages = len(lang_hours)
            max_lang_hours = max(lang_hours.values())
            min_lang_samples = min(lang_samples.values()) if lang_samples else 0

            # If balance_target_hours is set, use it as the per-lang target
            # instead of max_lang_hours (works with "max" and "cap" modes).
            effective_target_hours = (
                self.balance_target_hours
                if self.balance_target_hours and self.balance_mode in ("max", "cap")
                else max_lang_hours
            )

            # Calculate expected epoch samples based on balance mode
            if self.balance_mode == "max":
                expected_epoch_samples = 0
                for lang, hours in lang_hours.items():
                    ratio = effective_target_hours / hours if hours > 0 else 0
                    expected_epoch_samples += lang_samples[lang] * ratio
            elif self.balance_mode == "min":
                expected_epoch_samples = min_lang_samples * num_languages
            elif self.balance_mode == "distributed":
                expected_epoch_samples = total_samples
            elif self.balance_mode == "cap":
                expected_epoch_samples = 0
                for lang, hours in lang_hours.items():
                    ratio = effective_target_hours / hours if hours > 0 else 0
                    expected_epoch_samples += lang_samples[lang] * ratio
            else:
                expected_epoch_samples = total_samples

            logging.info("=" * 70)
            logging.info(f"LANGUAGE BALANCING STRATEGY (mode: {self.balance_mode})")
            logging.info("=" * 70)
            logging.info(f"Total data: {total_hours:.1f}h across {num_languages} languages ({total_samples:,} samples)")

            if self.balance_mode == "max":
                if self.balance_target_hours:
                    logging.info(
                        f"Target per language: {self.balance_target_hours:.1f}h "
                        f"(custom cap; largest natural is {max_lang_hours:.1f}h)"
                    )
                else:
                    logging.info(f"Target per language: {max_lang_hours:.1f}h (largest language)")
            elif self.balance_mode == "min":
                logging.info(f"Target per language: {min_lang_samples:,} samples (smallest language)")
            elif self.balance_mode == "distributed":
                logging.info(f"Proportional distribution: each language uses its own data, spread evenly")
            elif self.balance_mode == "cap":
                logging.info(f"Target per language: {self.balance_target_hours:.1f}h (cap)")

            logging.info("")
            logging.info(f"Expected epoch length: ~{int(expected_epoch_samples):,} samples")
            logging.info(f"  Unique samples: {total_samples:,}")

            # Auto-adjust samples_per_epoch if not explicitly set by user
            if expected_epoch_samples != total_samples and not self._explicit_samples_per_epoch:
                self.samples_per_epoch = int(expected_epoch_samples)
                logging.info(f"  Auto-adjusted samples_per_epoch: {self.samples_per_epoch:,}")
            elif self._explicit_samples_per_epoch:
                logging.info(f"  User-set samples_per_epoch: {self.samples_per_epoch:,}")
            logging.info("")

            # Per-language summary
            logging.info("Per-language distribution:")
            for row in lang_rows:
                lang = row['lang'] or 'unknown'
                hours = row['hours']
                samples = row['samples']
                pct = (hours / total_hours) * 100 if total_hours > 0 else 0

                if self.balance_mode == "max":
                    ratio = effective_target_hours / hours if hours > 0 else 0
                    epoch_samples = int(samples * ratio)
                    if ratio >= 1.05:
                        tag = f"~{ratio:.1f}x repeat"
                    elif ratio <= 0.95:
                        tag = f"~{ratio*100:.1f}% subset"
                    else:
                        tag = "~1.0x"
                    logging.info(
                        f"  {lang:6s}: {hours:7.1f}h ({pct:5.1f}%) - {samples:>10,} samples - "
                        f"{tag} (~{epoch_samples:,} per epoch)"
                    )
                elif self.balance_mode == "min":
                    epoch_samples = min_lang_samples
                    logging.info(
                        f"  {lang:6s}: {hours:7.1f}h ({pct:5.1f}%) - {samples:>10,} samples - "
                        f"capped at {epoch_samples:,}"
                    )
                elif self.balance_mode == "distributed":
                    logging.info(
                        f"  {lang:6s}: {hours:7.1f}h ({pct:5.1f}%) - {samples:>10,} samples - "
                        f"proportional"
                    )
                elif self.balance_mode == "cap":
                    ratio = effective_target_hours / hours if hours > 0 else 0
                    epoch_samples = int(samples * ratio)
                    if ratio >= 1.05:
                        tag = f"~{ratio:.1f}x repeat"
                    elif ratio <= 0.95:
                        tag = f"~{ratio*100:.1f}% subset"
                    else:
                        tag = "~1.0x"
                    logging.info(
                        f"  {lang:6s}: {hours:7.1f}h ({pct:5.1f}%) - {samples:>10,} samples - "
                        f"{tag} (~{epoch_samples:,} per epoch)"
                    )

            logging.info("=" * 70)

        except Exception as e:
            logging.warning(f"Could not log balancing strategy: {e}")

    def _log_epoch_plan(self):
        """Log a summary of the upcoming epoch: language stats, balancing, and expected steps."""
        if not self._sqlite_cache_path or not os.path.exists(self._sqlite_cache_path):
            return
        try:
            from .sqlite_manifest import SQLiteManifestCache
            cache = SQLiteManifestCache(self._sqlite_cache_path, read_only=True)
            conn = cache._get_connection()

            lang_rows = conn.execute("""
                SELECT lang, SUM(duration) / 3600.0 as hours, COUNT(*) as samples
                FROM manifest_entries
                GROUP BY lang
                ORDER BY hours DESC
            """).fetchall()
            cache.close()

            if not lang_rows:
                return

            lang_hours = {row['lang']: row['hours'] for row in lang_rows}
            lang_samples = {row['lang']: row['samples'] for row in lang_rows}
            max_lang = max(lang_hours, key=lang_hours.get)
            max_lang_hours = lang_hours[max_lang]
            num_languages = len(lang_hours)
            total_hours = sum(lang_hours.values())

            logging.info("")
            logging.info("=" * 60)
            logging.info("EPOCH PLAN")
            logging.info("=" * 60)
            logging.info(f"Languages: {num_languages} | Total unique data: {total_hours:.1f}h")
            logging.info(f"Balance mode: {self.balance_mode}")
            if self.balance_mode == "max":
                if self.balance_target_hours:
                    cap = self.balance_target_hours
                    logging.info(
                        f"Per-language cap: {cap:.1f}h "
                        f"(custom; largest natural is {max_lang} at {max_lang_hours:.1f}h)"
                    )
                    logging.info(
                        f"Expected epoch: ~{cap * num_languages:.1f}h "
                        f"({num_languages} x {cap:.1f}h)"
                    )
                else:
                    logging.info(f"Largest language: {max_lang} ({max_lang_hours:.1f}h)")
                    logging.info(f"Expected epoch: ~{max_lang_hours * num_languages:.1f}h "
                                 f"({num_languages} x {max_lang_hours:.1f}h)")
            elif self.balance_mode == "cap" and self.balance_target_hours:
                cap = self.balance_target_hours
                logging.info(f"Per-language cap: {cap:.1f}h")
                logging.info(f"Expected epoch: ~{cap * num_languages:.1f}h "
                             f"({num_languages} x {cap:.1f}h)")
            logging.info("-" * 60)

            for lang in sorted(lang_hours.keys()):
                hours = lang_hours[lang]
                samples = lang_samples[lang]
                if self.balance_mode == "max" and max_lang_hours > 0:
                    target = self.balance_target_hours or max_lang_hours
                    ratio = target / hours if hours > 0 else 0
                    epoch_samples = int(samples * ratio)
                    if ratio > 1.05:
                        tag = f"{ratio:.1f}x repeat (oversample)"
                    elif ratio < 0.95:
                        tag = f"{ratio*100:.1f}% subset (undersample)"
                    else:
                        tag = "1.0x (at cap)" if self.balance_target_hours else "1.0x (largest)"
                    logging.info(
                        f"  {lang}: {hours:.1f}h ({samples:,} samples) "
                        f"-> {tag} -> ~{epoch_samples:,} samples/epoch"
                    )
                elif self.balance_mode == "cap" and self.balance_target_hours and hours > 0:
                    ratio = self.balance_target_hours / hours
                    epoch_samples = int(samples * ratio)
                    if ratio > 1.05:
                        tag = f"{ratio:.1f}x repeat (oversample)"
                    elif ratio < 0.95:
                        tag = f"{ratio*100:.1f}% subset (undersample)"
                    else:
                        tag = "1.0x (at cap)"
                    logging.info(
                        f"  {lang}: {hours:.1f}h ({samples:,} samples) -> {tag} "
                        f"-> ~{epoch_samples:,} samples/epoch"
                    )
                else:
                    logging.info(f"  {lang}: {hours:.1f}h ({samples:,} samples)")

            logging.info("-" * 60)
            logging.info(f"samples_per_epoch: {self.samples_per_epoch:,}")
            logging.info("=" * 60)
            logging.info("")
        except Exception as e:
            logging.debug(f"Could not log epoch plan: {e}")

    def _calculate_total_samples(self) -> int:
        """
        Calculate total samples by counting manifest entries.

        This loads each manifest once to get the accurate count.
        If use_sqlite_cache is enabled, also builds the SQLite cache
        so workers don't need to reload manifests into memory.

        Returns:
            Total number of samples across all language sources
        """
        total = 0

        # Initialize SQLite cache if enabled
        sqlite_cache = None
        if self.use_sqlite_cache and self._sqlite_cache_path:
            # Check if cache already exists
            if os.path.exists(self._sqlite_cache_path):
                try:
                    sqlite_cache = SQLiteManifestCache(self._sqlite_cache_path, read_only=True)

                    # If explicit cache path provided by user, trust it completely (don't validate/rebuild)
                    if self._explicit_cache_path:
                        total = sqlite_cache.count_entries()
                        logging.info(f"Using pre-built SQLite cache (explicit path): {self._sqlite_cache_path}")
                        logging.info(f"Total manifest entries from cache: {total}")
                        sqlite_cache.close()
                        return total

                    # Auto-generated cache: validate it matches expected sources
                    cached_sources = set(sqlite_cache.get_sources())
                    expected_sources = set()
                    for sources in self.language_sources.values():
                        expected_sources.update(sources)

                    # Also check TAR file lists are cached
                    tar_cache_valid = all(
                        sqlite_cache.has_tar_files(source)
                        for source in expected_sources
                    )

                    if cached_sources == expected_sources and tar_cache_valid:
                        # Cache is valid, just count from it
                        total = sqlite_cache.count_entries()
                        logging.info(f"Using existing SQLite manifest cache: {self._sqlite_cache_path}")
                        logging.info(f"Total manifest entries from cache: {total}")
                        sqlite_cache.close()
                        return total
                    else:
                        # Cache is stale, rebuild it
                        reason = "TAR file lists missing" if not tar_cache_valid else "sources mismatch"
                        logging.info(f"SQLite cache invalid ({reason}), rebuilding...")
                        sqlite_cache.close()
                        os.remove(self._sqlite_cache_path)
                except Exception as e:
                    logging.warning(f"Failed to read existing cache: {e}")
                    if sqlite_cache:
                        sqlite_cache.close()
                    # Only remove if not explicit (user-provided cache should not be deleted)
                    if not self._explicit_cache_path and os.path.exists(self._sqlite_cache_path):
                        os.remove(self._sqlite_cache_path)
                    elif self._explicit_cache_path:
                        logging.error(f"Pre-built cache is corrupted or invalid: {self._sqlite_cache_path}")
                        raise

            elif self._explicit_cache_path:
                # User provided explicit cache path but file doesn't exist
                raise FileNotFoundError(
                    f"Explicit SQLite cache path provided but file not found: {self._sqlite_cache_path}\n"
                    f"Build the cache first with: python build_cache.py --datasets-config <config> --output {self._sqlite_cache_path}"
                )

            # Create new cache (only for auto-generated paths)
            sqlite_cache = SQLiteManifestCache(self._sqlite_cache_path)
            logging.info(f"Building SQLite manifest cache: {self._sqlite_cache_path}")

        # Count total sources for progress indicator
        total_sources = sum(len(sources) for sources in self.language_sources.values())

        # Build flat list of (lang, source) for progress bar
        all_sources = [
            (lang, source)
            for lang, sources in self.language_sources.items()
            for source in sources
        ]

        # Try to use tqdm for overall progress
        try:
            from tqdm import tqdm
            sources_iterator = tqdm(
                all_sources,
                desc="Building manifest cache",
                unit=" datasets",
                ncols=100,
            )
            use_tqdm = True
        except ImportError:
            sources_iterator = all_sources
            use_tqdm = False

        if self.storage_type == "s3":
            manifest_loader = S3ManifestLoader(s3_client=self.s3_client)
            for idx, (lang, source) in enumerate(sources_iterator):
                if use_tqdm:
                    sources_iterator.set_postfix_str(f"{lang}/{source}")
                source_prefix = f"{self.s3_prefix.rstrip('/') + '/' if self.s3_prefix else ''}{source}/"
                manifest_key = f"{source_prefix}tarred_audio_manifest.json"
                try:
                    if sqlite_cache:
                        # Load full manifest and add to cache
                        manifest = manifest_loader.load_manifest(self.s3_bucket, manifest_key)
                        count = sqlite_cache.add_entries_batch(manifest, source)
                        total += count
                        # Also cache TAR file list to avoid S3 list calls in workers
                        tar_files = manifest_loader.list_tar_files(
                            self.s3_bucket, source_prefix,
                            sqlite_cache=sqlite_cache, source=source
                        )
                        logging.info(f"[{idx+1}/{total_sources}] [{lang}] {source}: {count} entries, {len(tar_files)} TARs")
                    else:
                        count = manifest_loader.count_manifest_entries(
                            self.s3_bucket, manifest_key
                        )
                        total += count
                        logging.info(f"[{idx+1}/{total_sources}] [{lang}] {source}: {count} manifest entries")
                except Exception as e:
                    logging.warning(f"[{lang}] Failed to load manifest for {source}: {e}")
        else:
            # Disk storage
            manifest_loader = DiskManifestLoader()
            for idx, (lang, source) in enumerate(sources_iterator):
                if use_tqdm:
                    sources_iterator.set_postfix_str(f"{lang}/{source}")
                if self.lang_subdir:
                    source_dir = os.path.join(self.data_root, lang, source)
                else:
                    source_dir = os.path.join(self.data_root, source)
                manifest_path = os.path.join(source_dir, "tarred_audio_manifest.json")
                try:
                    if sqlite_cache:
                        # Load full manifest and add to cache
                        manifest = manifest_loader.load_manifest(manifest_path)
                        count = sqlite_cache.add_entries_batch(manifest, source)
                        total += count
                        # Also cache TAR file list
                        tar_files = manifest_loader.list_tar_files(
                            source_dir,
                            sqlite_cache=sqlite_cache, source=source
                        )
                        logging.info(f"[{idx+1}/{total_sources}] [{lang}] {source}: {count} entries, {len(tar_files)} TARs")
                    else:
                        count = manifest_loader.count_manifest_entries(manifest_path)
                        total += count
                        logging.info(f"[{idx+1}/{total_sources}] [{lang}] {source}: {count} manifest entries")
                except Exception as e:
                    logging.warning(f"[{lang}] Failed to load manifest for {source}: {e}")

        if sqlite_cache:
            # Checkpoint to ensure all data is flushed before workers read
            sqlite_cache.checkpoint()
            sqlite_cache.close()
            logging.info(f"SQLite manifest cache built with {total} entries")
            logging.info("=" * 60)
            logging.info("CACHE BUILD COMPLETE - ready for training")
            logging.info("=" * 60)
            # Force flush to ensure logs appear before workers start
            import sys
            sys.stdout.flush()
            sys.stderr.flush()

        return total

    def _create_interleaver(self):
        """Create interleaver based on rotation_level (called once per worker)."""
        if self.rotation_level == "source":
            return self._create_source_interleaver()
        else:
            return self._create_language_interleaver()

    def _create_language_interleaver(self) -> RoundRobinInterleaver:
        """Create language-level round-robin interleaver."""
        language_managers = {}

        for lang, sources in self.language_sources.items():
            if self.storage_type == "s3":
                manager = LanguageSourceManager(
                    lang=lang,
                    sources=sources,
                    sample_filter=self.sample_filter,
                    token_augmenter=self.token_augmenter,
                    storage_type="s3",
                    s3_bucket=self.s3_bucket,
                    s3_prefix=self.s3_prefix,
                    s3_endpoint_url=self.s3_endpoint_url,
                    sqlite_cache_path=self._sqlite_cache_path,
                    prefetch_buffer_size=self.prefetch_buffer_size,
                    clean_sources=self.clean_sources,
                    clean_pnc_gate=self.clean_pnc_gate,
                )
            else:
                manager = LanguageSourceManager(
                    lang=lang,
                    sources=sources,
                    sample_filter=self.sample_filter,
                    token_augmenter=self.token_augmenter,
                    storage_type="disk",
                    data_root=self.data_root,
                    lang_subdir=self.lang_subdir,
                    sqlite_cache_path=self._sqlite_cache_path,
                    prefetch_buffer_size=self.prefetch_buffer_size,
                    clean_sources=self.clean_sources,
                    clean_pnc_gate=self.clean_pnc_gate,
                )
            language_managers[lang] = manager

        return RoundRobinInterleaver(
            language_managers=language_managers,
            languages_order=sorted(self.language_sources.keys()),
            stop_on_exhaustion=self._stop_on_lang_exhaustion,
        )

    def _create_source_interleaver(self) -> SourceRoundRobinInterleaver:
        """Create source-level round-robin interleaver."""
        source_managers = {}

        for lang, sources in self.language_sources.items():
            for source in sources:
                manager_key = f"{lang}:{source}"
                if self.storage_type == "s3":
                    manager = SingleSourceManager(
                        lang=lang,
                        source=source,
                        sample_filter=self.sample_filter,
                        token_augmenter=self.token_augmenter,
                        storage_type="s3",
                        s3_bucket=self.s3_bucket,
                        s3_prefix=self.s3_prefix,
                        s3_endpoint_url=self.s3_endpoint_url,
                        sqlite_cache_path=self._sqlite_cache_path,
                        prefetch_buffer_size=self.prefetch_buffer_size,
                        clean_sources=self.clean_sources,
                        clean_pnc_gate=self.clean_pnc_gate,
                    )
                else:
                    manager = SingleSourceManager(
                        lang=lang,
                        source=source,
                        sample_filter=self.sample_filter,
                        token_augmenter=self.token_augmenter,
                        storage_type="disk",
                        data_root=self.data_root,
                        lang_subdir=self.lang_subdir,
                        sqlite_cache_path=self._sqlite_cache_path,
                        prefetch_buffer_size=self.prefetch_buffer_size,
                        clean_sources=self.clean_sources,
                        clean_pnc_gate=self.clean_pnc_gate,
                    )
                source_managers[manager_key] = manager

        lang_target_samples = None
        if self.balance_mode in ("cap", "max") and self.balance_target_hours and self._lang_sample_counts:
            # Compute per-lang target samples from target_hours * samples_per_hour.
            # `_lang_hours` is populated alongside `_lang_sample_counts` when the
            # cache is loaded; fall back to a heuristic if it's missing.
            lang_hours = getattr(self, "_lang_hours", None) or {}
            lang_target_samples = {}
            for lang, samples in self._lang_sample_counts.items():
                hours = lang_hours.get(lang, 0)
                if hours > 0:
                    samples_per_hour = samples / hours
                    lang_target_samples[lang] = max(1, int(self.balance_target_hours * samples_per_hour))
                else:
                    # No hours info — leave the natural count (no cap)
                    lang_target_samples[lang] = samples
            logging.info(
                f"balance_mode='{self.balance_mode}' target={self.balance_target_hours}h/lang -> "
                f"per-lang target samples: {lang_target_samples}"
            )

        return SourceRoundRobinInterleaver(
            language_sources=self.language_sources,
            source_managers=source_managers,
            languages_order=sorted(self.language_sources.keys()),
            samples_per_source=self.samples_per_source,
            balance_mode=self.balance_mode,
            lang_sample_counts=self._lang_sample_counts,
            source_sample_counts=self._source_sample_counts,
            lang_target_samples=lang_target_samples,
            curriculum_buckets=self.curriculum_buckets,
        )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        """
        Iterate over training samples.

        Yields:
            Tuple of (audio, audio_len, tokens, tokens_len) or
            (audio, audio_len, tokens, tokens_len, sample_id) if return_sample_id=True
        """
        import time as _time
        _PROF = os.environ.get('DATASET_PROFILE', '0') == '1'
        try:
            _PROF_EVERY = int(os.environ.get('DATASET_PROFILE_EVERY', '2000'))
        except ValueError:
            _PROF_EVERY = 2000

        # Get worker info for multi-worker data loading
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        # Read distributed info live, so this works correctly even when the
        # caller didn't plumb global_rank/world_size through the constructor.
        # Lightning's use_distributed_sampler is a no-op for IterableDataset,
        # so each rank otherwise iterates the full sample stream → world_size×
        # duplicate work. Each rank now seeds independently and limits its own
        # quota to samples_per_epoch / world_size.
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            else:
                rank = self.global_rank
                world_size = max(1, self.world_size)
        except Exception:
            rank = self.global_rank
            world_size = max(1, self.world_size)

        logging.debug(f"[Worker {worker_id}] Starting iteration (rank={rank}/{world_size})")

        # Log epoch plan from worker 0 only
        if worker_id == 0 and rank == 0:
            self._log_epoch_plan()

        if worker_info is not None:
            # In multi-worker mode, give each (rank, worker) pair a distinct
            # random seed so the substreams diverge.
            random.seed(worker_info.id + rank * 1000)
        else:
            # Single-process: still seed per rank to diverge across ranks.
            if rank > 0:
                random.seed(rank * 1000)

        # Create interleaver for this worker
        interleaver = self._create_interleaver()

        # Shuffle buffer
        shuffle_buffer = ShuffleBuffer(buffer_size=self.shuffle_buffer_size)

        # Merge buffer for multi-utterance merging
        merge_buffer = MergeBuffer(self.audio_merger, self.merge_config)

        sample_idx = 0
        # Per-rank epoch budget, gated on a *cross-worker* shared counter rather
        # than a static per-worker slice. samples_per_epoch is the cluster-wide
        # budget; each rank takes 1/world_size of it. Within a rank, all
        # DataLoader workers share `self._raw_consumed` (an mp.Value inherited
        # across the fork), so the rank stops as a whole when the shared
        # raw-consumption count reaches `rank_budget`.
        #
        # Why not a static per-worker cap (samples_per_epoch // (world_size *
        # num_workers))? TAR files shard disjointly across workers
        # (lang_source_manager: tar[i] -> worker i % N), so a worker's partition
        # size depends entirely on how many tars it happens to own. A static cap
        # only yields full coverage when tars are many and uniform; with few or
        # uneven tars it truncates workers holding fat shards while empty-shard
        # workers contribute nothing — ending the epoch far below 100%. Gating
        # on the shared total makes coverage independent of tar layout and of
        # num_workers: workers with data keep pulling until the rank budget is
        # met (or their partition drains), absorbing slack from idle workers.
        num_workers = worker_info.num_workers if worker_info is not None else 1
        num_workers = max(1, num_workers)
        rank_budget = max(1, self.samples_per_epoch // world_size)
        if rank == 0 and worker_id == 0:
            logging.info(
                f"Sharded streaming: {self.samples_per_epoch:,} samples/epoch, "
                f"rank budget {rank_budget:,} ({world_size} ranks); "
                f"{num_workers} workers/rank share the budget via a cross-worker counter"
            )

        # Per-stage timing counters; only populated when DATASET_PROFILE=1.
        # Splits one consumer cycle into:
        #   t_fetch  : interleaver.next  (DiskTarStream pull + buffer/merge)
        #   t_proc   : _process_sample   (augment + tensor + tokenize)
        #   t_yield  : time the yield blocks (downstream backpressure: dynamic
        #              batch sort/pack + DataLoader IPC)
        _ds_t_fetch = 0.0
        _ds_t_proc = 0.0
        _ds_t_yield = 0.0
        _ds_n = 0
        _ds_t0 = _time.perf_counter() if _PROF else 0.0

        def _ds_emit():
            if not _PROF or _ds_n == 0:
                return
            tot = _ds_t_fetch + _ds_t_proc + _ds_t_yield
            logging.info(
                f"[DS_PROF wid={worker_id}] n={_ds_n} "
                f"fetch={_ds_t_fetch*1000:.0f}ms ({_ds_t_fetch/_ds_n*1000:.2f}ms/smp) "
                f"proc={_ds_t_proc*1000:.0f}ms ({_ds_t_proc/_ds_n*1000:.2f}ms/smp) "
                f"yield_block={_ds_t_yield*1000:.0f}ms ({_ds_t_yield/_ds_n*1000:.2f}ms/smp) "
                f"total={tot*1000:.0f}ms"
            )

        def process_and_yield(sample):
            """Process sample through merge buffer and yield results."""
            nonlocal sample_idx, _ds_t_proc

            # Try to add to merge buffer - may return merged sample, single sample, or None
            result_sample = merge_buffer.add(sample)
            if result_sample is not None:
                if _PROF:
                    _t = _time.perf_counter()
                result = self._process_sample(result_sample, sample_idx)
                if _PROF:
                    _ds_t_proc += _time.perf_counter() - _t
                if result is not None:
                    sample_idx += 1

                    # Log stats periodically
                    if sample_idx % 10000 == 0:
                        if self.merge_config.enabled:
                            self.audio_merger.log_stats()

                    return result
            return None

        # Stream samples through interleaver and shuffle buffer.
        #
        # Gate on the shared cross-worker raw-consumption counter. We count raw
        # inputs *pulled* (not yielded) because merging makes yielded ≪ pulled;
        # counting pulls keeps the progress bar honest and lets the rank stop at
        # exactly its budget regardless of merge ratio. Each worker breaks when
        # the shared counter reaches `rank_budget`, or earlier if its own tar
        # partition drains (StopIteration). Buffered samples are flushed in the
        # drain phase below; since they were already counted as pulls, the bar
        # is already at ~100% when the main loop ends and the drain is invisible.
        _last_yield_resume = _time.perf_counter() if _PROF else 0.0
        _interleaver_iter = iter(interleaver)
        # _raw_pulled: this worker's pull count (diagnostic only). _unflushed:
        # delta batched up to keep the shared mp.Value lock off the hot path;
        # we read the shared total at each flush to decide when to stop.
        _raw_pulled = 0
        _unflushed = 0
        _RAW_FLUSH_EVERY = 64
        # Raw pulls accumulated since the last yield, attributed to the next
        # yielded sample via self._cur_yield_weight so the batcher can weight
        # its emit-time bar counter. Sums to the rank budget over the epoch.
        _pending_raw = 0
        self._cur_yield_weight = 0
        # Generation-guarded reset of the shared counters: the first worker to
        # enter __iter__ for this pass zeroes both, others see the gen already
        # advanced and skip. Done under the lock before any counting, so no
        # worker loses increments and stale residue from a prior pass (e.g. the
        # sanity check's async prefetch) is wiped exactly once.
        with self._raw_consumed.get_lock():
            _gen = self._epoch_gen.value
            if self._counter_gen.value != _gen:
                self._raw_consumed.value = 0
                self._raw_emitted.value = 0
                self._counter_gen.value = _gen
        while True:
            if _PROF:
                _t = _time.perf_counter()
            try:
                sample = next(_interleaver_iter)
            except StopIteration:
                # This worker's tar partition is drained; stop contributing.
                # Other workers carry the remaining rank budget.
                break
            if _PROF:
                _ds_t_fetch += _time.perf_counter() - _t

            _raw_pulled += 1
            _unflushed += 1
            _pending_raw += 1
            if _unflushed >= _RAW_FLUSH_EVERY:
                with self._raw_consumed.get_lock():
                    self._raw_consumed.value += _unflushed
                    _cluster = self._raw_consumed.value
                _unflushed = 0
                if _cluster >= rank_budget:
                    break

            shuffle_buffer.add(sample)

            # Yield when buffer is full
            while shuffle_buffer.is_ready():
                buffered_sample = shuffle_buffer.get_random()
                if buffered_sample is not None:
                    result = process_and_yield(buffered_sample)
                    if result is not None:
                        if _PROF:
                            _yield_start = _time.perf_counter()
                        self._cur_yield_weight = _pending_raw
                        _pending_raw = 0
                        yield result
                        if _PROF:
                            _ds_t_yield += _time.perf_counter() - _yield_start
                            _ds_n += 1
                            if _ds_n % _PROF_EVERY == 0:
                                _ds_emit()
                                _ds_t_fetch = 0.0
                                _ds_t_proc = 0.0
                                _ds_t_yield = 0.0
                                _ds_n = 0

        # Flush any unflushed raw-input increments so the progress bar lands
        # at exactly the interleaver budget at epoch end.
        if _unflushed > 0:
            with self._raw_consumed.get_lock():
                self._raw_consumed.value += _unflushed
            _unflushed = 0

        # Drain remaining buffered samples. These were already pulled (and
        # counted toward the rank budget), so we do NOT re-increment the shared
        # counter — the progress bar is already at ~100% from the pull count.
        # They're real data, so always flush them even if the budget was met.
        for buffered_sample in shuffle_buffer.drain():
            result = process_and_yield(buffered_sample)
            if result is not None:
                self._cur_yield_weight = _pending_raw
                _pending_raw = 0
                yield result

        # Drain remaining samples from merge buffer
        for remaining_sample in merge_buffer.drain():
            result = self._process_sample(remaining_sample, sample_idx)
            if result is not None:
                self._cur_yield_weight = _pending_raw
                _pending_raw = 0
                yield result
                sample_idx += 1

    @staticmethod
    def _starts_with_capital(text: str) -> bool:
        """Whether the first alphabetical character in `text` is uppercase.
        Skips leading non-letter characters (quotes, digits, etc.) so
        '« Bonjour »' and '42 Is the answer' both pass."""
        for ch in text or '':
            if ch.isalpha():
                return ch.isupper()
        return False

    def _maybe_inject_early_lang(
        self,
        ids: List[int],
        lang_id: Optional[int],
        after_word: int,
    ) -> List[int]:
        """
        Splice a `<lang>` token into a token sequence after the N-th word.

        Args:
            ids:       Token IDs produced by `tokenizer.text_to_ids(text)` for
                       a single text span (one utterance / one merged segment).
            lang_id:   The lang token to insert. If None, nothing is inserted.
            after_word: 1-based word index after which to insert. 0 disables.

        Returns:
            New token list with the lang ID inserted, or the original list
            unchanged if there are not enough words (must have > after_word).

        Word boundaries are detected via the SentencePiece word-start convention:
        position 0 is always a word start, plus every piece whose string begins
        with U+2581 (the ▁ space marker). Holds for SP models built with
        `add_dummy_prefix=False`, which is what this codebase uses.
        """
        if after_word <= 0 or lang_id is None or not ids:
            return ids
        try:
            pieces = self.tokenizer.ids_to_tokens(ids)
        except Exception:
            return ids
        word_starts = [
            i for i, p in enumerate(pieces)
            if i == 0 or (isinstance(p, str) and p.startswith('▁'))
        ]
        # Total words = number of word-start positions. Need strictly more
        # words than `after_word` so that an N+1-th word actually exists.
        if len(word_starts) <= after_word:
            return ids
        insert_pos = word_starts[after_word]
        return ids[:insert_pos] + [lang_id] + ids[insert_pos:]

    def _clean_unk_chars(self, text: str) -> str:
        """Remove characters the tokenizer can't encode (would become <unk>),
        replacing each with a space and collapsing whitespace.

        Without byte_fallback, characters absent from the vocab — e.g. em-dash,
        colon, guillemets, ellipsis — tokenize to <unk>. Rather than drop the
        whole utterance (losing its audio), we strip those characters so the
        sample survives with clean text; the model simply never sees them. A
        character that is <unk> on its own is also <unk> in context (the
        tokenizer can't merge an unknown character into a known piece), so the
        per-character test is exact here.
        """
        if self._unk_id is None or not text:
            return text
        cache = self._unk_char_cache
        cleaned = []
        changed = False
        for ch in text:
            is_unk = cache.get(ch)
            if is_unk is None:
                is_unk = self._unk_id in self.tokenizer.text_to_ids(ch)
                cache[ch] = is_unk
            if is_unk:
                changed = True
                cleaned.append(' ')
            else:
                cleaned.append(ch)
        if not changed:
            return text
        return ' '.join(''.join(cleaned).split())

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

        # Record this sample's cleanliness on the per-yield side-channel so the
        # dynamic batcher (which only sees the collated tensor tuple, not the
        # dict) can group homogeneous batches. Merged samples carry a single
        # 'clean' flag set by the AudioMerger (merges are gated to be
        # homogeneous). Defaults to True when the feature is off.
        self._cur_yield_clean = bool(sample.get('clean', True))

        # Apply augmentation based on source cycle and configured probability
        source_cycle = sample.get('source_cycle', 0)
        if self.augmentor.should_augment(source_cycle):
            audio = self.augmentor(audio, sample_rate=16000, force=True)

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
            # Merged sample: tokenize each segment and add EOU where appropriate.
            # For romanize/romanize_word LID schemes the input text is romanized
            # downstream by `_build_lid_tokens`, so the per-char UNK strip would
            # wipe non-Latin scripts (e.g. uk Cyrillic) before romanization runs.
            # The romanizer already drops anything outside [a-z ] — let it do
            # the cleaning.
            skip_unk_clean = (
                self.lid_mode and self.lid_label_scheme in ("romanize", "romanize_word")
            )
            if not skip_unk_clean:
                original_texts = [self._clean_unk_chars(t) for t in original_texts]
            tokens = self._tokenize_merged(
                original_texts,
                eou_positions,
                sample_lang=sample.get('lang'),
                segment_langs=sample.get('langs'),
            )
        elif self.lid_mode:
            # LID mode (single utterance). Scheme = "word" or "syllable" — both
            # routed through `_build_lid_tokens`. No BOS/EOS/EOU/normalization.
            # romanize/romanize_word romanize the text downstream; skip the
            # UNK-char strip there or it wipes non-Latin scripts pre-romanize.
            text = sample.get('text', '') or ''
            if self.lid_label_scheme not in ("romanize", "romanize_word"):
                text = self._clean_unk_chars(text)
            sample_lang = sample.get('lang')
            if not sample_lang:
                return None
            tokens = self._build_lid_tokens(text, sample_lang)
            if tokens is None:
                return None
        else:
            # Regular sample
            text = sample.get('text', '')

            # Strip characters the tokenizer can't represent (keeps the sample
            # instead of dropping it via the <unk> guard below).
            text = self._clean_unk_chars(text)

            # Check if we should add EOU token BEFORE normalization (punctuation check)
            should_add_eou = (
                self.add_eou_token
                and self.eou_token_id is not None
                and self.token_augmenter.should_add_eou(text)
            )

            # If EOU mode is on, drop samples without terminal punctuation.
            # They produce inconsistent label patterns (no <eou>, but lang tag
            # may still follow) and dilute the meaning of <eou>. Allow them
            # only when EOU is disabled entirely (full-norm training).
            if self.add_eou_token and self.eou_token_id is not None and not should_add_eou:
                return None

            # Same gating: drop samples that don't start with a capital letter.
            # These are mid-sentence fragments that pollute merged outputs with
            # things like "...<eou>imitando um sinal..." (lowercase mid-segment).
            # Paired with the punct check, this ensures every single sample
            # (and every merge candidate) is a complete sentence.
            if self.add_eou_token and self.eou_token_id is not None:
                if not self._starts_with_capital(text):
                    return None

            # Normalize text if configured
            if self.normalize_text == "full":
                text = self._normalize_text(text)
            elif self.normalize_text == "soft":
                text = self._soft_normalize_text(text)

            # Tokenize text (without <eou> - we'll append the ID directly)
            tokens = self.tokenizer.text_to_ids(text)

            # Optional: inject a lang tag mid-text after the N-th word so the
            # model learns to commit to a language early (before <eou>).
            # Only active when add_lang_token=True AND text has > N words.
            if (
                self.early_lang_token_after_word > 0
                and self.add_lang_token
                and self.lang_token_ids
            ):
                sample_lang = sample.get('lang')
                early_lang_id = (
                    self.lang_token_ids.get(sample_lang) if sample_lang else None
                )
                tokens = self._maybe_inject_early_lang(
                    tokens, early_lang_id, self.early_lang_token_after_word
                )

            # Add BOS token if configured
            if self.bos_id is not None:
                tokens = [self.bos_id] + tokens

            # Add EOU token ID directly (before EOS, after text)
            if should_add_eou:
                tokens = tokens + [self.eou_token_id]

            # Add EOT token at end of turn (before EOS)
            if self.add_eot_token and self.eot_token_id is not None:
                tokens = tokens + [self.eot_token_id]

            # Append per-utterance lang tag (e.g. "...text <uk>") so the
            # model learns to identify the language at the end of speech.
            if self.add_lang_token and self.lang_token_ids:
                sample_lang = sample.get('lang')
                lang_id = self.lang_token_ids.get(sample_lang) if sample_lang else None
                if lang_id is not None and (
                    self.lang_token_prob >= 1.0 or random.random() < self.lang_token_prob
                ):
                    tokens = tokens + [lang_id]

            # Add EOS token if configured
            if self.eos_id is not None:
                tokens = tokens + [self.eos_id]

        # Guard: an empty target sequence has no signal to learn from (and
        # breaks RNNT loss). Can happen in LID mode when a merged sample's
        # segments all romanize to empty content.
        if len(tokens) == 0:
            return None

        # Guard: RNNT numba kernels launch with maxU = max(label_lengths) + 1 as
        # threads-per-block; CUDA hard-caps this at 1024.  Skip samples that
        # would breach the limit (typically corrupted data).
        if len(tokens) > 1023:
            return None

        # Drop samples whose tokenization contains <unk>. They surface in WER
        # logs as "⁇" and the model has no acoustic signal for them.
        if self._unk_id is not None and self._unk_id in tokens:
            return None

        tokens_tensor = torch.tensor(tokens).long()
        tokens_len = torch.tensor(len(tokens)).long()

        if self.return_sample_id:
            return audio_tensor, audio_len, tokens_tensor, tokens_len, sample_idx
        else:
            return audio_tensor, audio_len, tokens_tensor, tokens_len

    # Vowel sets for the syllable-count heuristic. Union of Latin (with common
    # diacritics) + Cyrillic vowels — works for all eu-23 languages.
    _LID_VOWELS = set(
        "aeiouy"
        "äöüß"
        "áéíóúýñ"
        "àèìòùâêîôûãõçœæ"
        "ąęųīē"
        # Cyrillic (lowercase)
        "аеиіоуюяїєё"
    )

    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count: number of contiguous vowel groups.
        Returns at least 1 so consonant-only tokens (acronyms, interjections)
        still get one `<syl>` marker."""
        word = word.lower()
        n, in_v = 0, False
        for ch in word:
            if ch in self._LID_VOWELS:
                if not in_v:
                    n += 1
                in_v = True
            else:
                in_v = False
        return max(1, n)

    def _build_lid_tokens(self, text: str, lang: str) -> Optional[List[int]]:
        """Return the LID label sequence for one utterance segment in `lang`.
        Returns None if the language isn't in the vocab (sample will be skipped)."""
        words = (text or "").split()
        if self.lid_label_scheme == "word":
            lang_id = self.lang_token_ids.get(lang)
            if lang_id is None:
                return None
            return [lang_id] * max(1, len(words))

        if self.lid_label_scheme == "romanize":
            # `<lang_S>` + BPE(romanized text) + `<lang_E>`. The romanized text
            # is a dense content target; the boundaries carry the LID label.
            s_id = self.lang_start_ids.get(lang)
            e_id = self.lang_end_ids.get(lang)
            if s_id is None or e_id is None:
                return None
            from .romanize import romanize as _romanize
            roman = _romanize(text or "", lang)
            inner = self.tokenizer.text_to_ids(roman) if roman else []
            return [s_id] + list(inner) + [e_id]

        if self.lid_label_scheme == "romanize_word":
            # Per-word: BPE(romanized word) + `<lang>`. The tag follows each
            # word, so the model commits the language only after hearing it
            # (authoritative per word, ~1-word latency, no provisional opening
            # tag to skew toward the majority class). Uses the single per-lang
            # `<lang>` token, same as the "word" scheme.
            lang_id = self.lang_token_ids.get(lang)
            if lang_id is None:
                return None
            from .romanize import romanize as _romanize
            out: List[int] = []
            for w in words:
                roman = _romanize(w, lang)
                if not roman:
                    continue  # word stripped to nothing — no bare content-free tag
                ids = self.tokenizer.text_to_ids(roman)
                if not ids:
                    continue
                out.extend(ids)
                out.append(lang_id)
            if not out:
                # Whole segment had no romanizable content. Drop the sample
                # rather than emit a content-free `<lang>` tag — bare tags in
                # the reference teach the model nothing and pollute WER logs.
                return None
            return out

        # syllable scheme
        s_id = self.lang_start_ids.get(lang)
        e_id = self.lang_end_ids.get(lang)
        if s_id is None or e_id is None or self.syllable_token_id is None:
            return None
        out: List[int] = []
        for w in words:
            n = self._count_syllables(w)
            out.append(s_id)
            out.extend([self.syllable_token_id] * n)
            out.append(e_id)
        if not words:  # empty text guard — emit one bracket pair
            out = [s_id, self.syllable_token_id, e_id]
        return out

    def _tokenize_merged(
        self,
        texts: List[str],
        eou_positions: List[int],
        sample_lang: Optional[str] = None,
        segment_langs: Optional[List[str]] = None,
    ) -> List[int]:
        """
        Tokenize merged multi-utterance/multi-turn sample.

        When EOT mode is active (add_eot_token=True), each segment is a turn
        and gets EOT appended. EOU is not used.
        Example: BOS + tokens(turn1) + EOT + tokens(turn2) + EOT + EOS

        When EOU mode is active (add_eou_token=True, add_eot_token=False),
        EOU is inserted after segments with sentence-ending punctuation.

        When lid_mode is active, the segment text is replaced by `[<lang_i>] *
        num_words_i` per segment, producing a label sequence like
        `<en> <en> <en> <fr> <fr>` for code-switch training.

        Args:
            texts: List of text segments
            eou_positions: Indices of segments that should have EOU appended
            sample_lang: Fallback language tag (used when segment_langs is not provided)
            segment_langs: Per-segment language tags, parallel to `texts`

        Returns:
            Combined token list with special tokens inserted appropriately
        """
        # LID mode: emit per-segment lang tokens (word or syllable scheme).
        # No BOS/EOS/EOU/EOT.
        if self.lid_mode:
            tokens: List[int] = []
            langs = segment_langs if segment_langs is not None else [sample_lang] * len(texts)
            for txt, lang in zip(texts, langs):
                if not lang:
                    continue
                seg = self._build_lid_tokens(txt or "", lang)
                if seg is not None:
                    tokens.extend(seg)
            return tokens

        tokens = []

        # Add BOS token if configured (only at the very beginning)
        if self.bos_id is not None:
            tokens.append(self.bos_id)

        # Decide once per sample whether to emit lang tokens at all (consistency
        # across segments — partial tagging would teach the model an ambiguous
        # signal). If we do tag, we emit one per segment using segment_langs.
        emit_lang_tags = (
            self.add_lang_token
            and self.lang_token_ids
            and (self.lang_token_prob >= 1.0 or random.random() < self.lang_token_prob)
        )
        if emit_lang_tags and segment_langs is None:
            segment_langs = [sample_lang] * len(texts)

        for i, text in enumerate(texts):
            # Normalize text if configured
            if self.normalize_text == "full":
                text = self._normalize_text(text)
            elif self.normalize_text == "soft":
                text = self._soft_normalize_text(text)

            # Tokenize this segment
            segment_tokens = self.tokenizer.text_to_ids(text)

            # Optional: inject this segment's lang tag mid-text, after the
            # N-th word. Gated by the same condition as end-of-segment lang
            # tagging (emit_lang_tags) so per-sample randomization stays
            # consistent within the same merged sample.
            if emit_lang_tags and self.early_lang_token_after_word > 0:
                seg_lang_for_early = (
                    segment_langs[i] if i < len(segment_langs) else sample_lang
                )
                early_lang_id = (
                    self.lang_token_ids.get(seg_lang_for_early)
                    if seg_lang_for_early else None
                )
                segment_tokens = self._maybe_inject_early_lang(
                    segment_tokens, early_lang_id, self.early_lang_token_after_word
                )

            tokens.extend(segment_tokens)

            if self.add_eot_token and self.eot_token_id is not None:
                # EOT mode: every segment is a turn, append EOT after each
                tokens.append(self.eot_token_id)
            elif self.add_eou_token and self.eou_token_id is not None and i in eou_positions:
                # EOU mode: append EOU after segments with sentence-ending
                # punctuation. Non-punctuated segments shouldn't reach here —
                # the merger now requires every merged sample to end with
                # terminal punctuation. The check is kept defensive.
                tokens.append(self.eou_token_id)

            # Append per-segment lang tag immediately after this segment's
            # boundary marker. Code-switch friendly: cross-lang merges produce
            # labels like  "<text_en> <eou> <en> <text_fr> <eou> <fr>" and the
            # inference layer can read each tag at every <eou> to know which
            # language the just-completed utterance was. Only emit when the
            # segment actually got a boundary marker — otherwise the lang
            # token would float without an <eou> in front of it.
            seg_got_boundary = (
                (self.add_eot_token and self.eot_token_id is not None)
                or (self.add_eou_token and self.eou_token_id is not None and i in eou_positions)
            )
            if emit_lang_tags and seg_got_boundary:
                seg_lang = segment_langs[i] if i < len(segment_langs) else None
                lang_id = self.lang_token_ids.get(seg_lang) if seg_lang else None
                if lang_id is not None:
                    tokens.append(lang_id)

        # Add EOS token if configured (only at the very end)
        if self.eos_id is not None:
            tokens.append(self.eos_id)

        return tokens

    # Word-internal apostrophe is orthography, not punctuation (uk м'ята,
    # fr l'école, en don't) — deleting it teaches the model unwritable words.
    # All variants fold to ASCII "'" (U+0027) so one tokenizer symbol covers them.
    _APOS_RE = re.compile(r'[\u2019\u02BC\u0060\u00B4\u2018]')
    # Hyphens/dashes become a space, never deleted: deletion merges compounds
    # ("будь-яких" -> "будьяких") and the model learns the merged spelling.
    _DASH_RE = re.compile(r'[\u002D\u2010-\u2015]+')

    # Pre-compiled regex for punctuation removal (all Unicode punctuation
    # categories except U+0027 "'" — apostrophes are folded and kept above)
    _PUNCT_RE = re.compile(r'[\u0021-\u0026\u0028-\u002F\u003A-\u003F\u005B-\u0060\u007B-\u007E'
                           r'\u00A1-\u00BF\u2000-\u206F\u2E00-\u2E7F'
                           r'\u3000-\u303F\uFE10-\uFE6F\uFF01-\uFF60'
                           r'\u0600-\u060F\u061B-\u061F\u066A-\u066D'
                           r'\u06D4\uFD3E\uFD3F]+')

    def _normalize_text(self, text: str) -> str:
        """
        Full normalization: remove all punctuation and lowercase.

        Produces "no PnC" (no Punctuation and Capitalization) text.
        Apostrophes are kept (folded to "'"); hyphens/dashes become spaces.
        """
        text = text.lower()
        text = self._APOS_RE.sub("'", text)
        text = self._DASH_RE.sub(' ', text)
        text = self._PUNCT_RE.sub('', text)
        text = ' '.join(text.split())
        return text.strip()

    # Regex for soft normalization: keep word chars, whitespace, . , ! ? @ and '
    _SOFT_PUNCT_RE = re.compile(r"[^\w\s.,!?@']", re.UNICODE)

    def _soft_normalize_text(self, text: str) -> str:
        """
        Soft normalization: keep capitalization, keep only . , ! ? punctuation.

        Apostrophes are kept (folded to "'"), hyphens/dashes become spaces, and
        all other punctuation (colons, semicolons, quotes, brackets, dashes
        etc.) is removed while preserving the original casing.
        """
        text = self._APOS_RE.sub("'", text)
        text = self._DASH_RE.sub(' ', text)
        text = self._SOFT_PUNCT_RE.sub('', text)
        text = ' '.join(text.split())
        return text.strip()

    def collate_fn(self, batch):
        """Collate function for DataLoader."""
        return _speech_collate_fn(batch, pad_id=0)

    def __len__(self):
        """
        Dataset length for epoch calculation.

        Returns samples_per_epoch, which is either:
        - Explicitly provided by user (always respected)
        - Auto-adjusted based on balance_mode if not explicitly set
        - Auto-calculated from manifest entry counts
        - Fallback estimate if manifest loading failed

        Source Balancing Behavior:
        If balance_mode="max":
        - Underrepresented sources are repeated during training with augmentation
        - If samples_per_epoch was NOT explicitly set, it's automatically adjusted
          to include all repeated data (expected_epoch_samples)
        - If samples_per_epoch WAS explicitly set, your value is respected
          (epochs may stop mid-repetition if set lower than needed)

        Example:
            Total unique samples: 100
            With auto-adjustment: 120 (includes repeats, __len__() returns 120)
            With explicit setting: samples_per_epoch=100 (stops at 100)
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
