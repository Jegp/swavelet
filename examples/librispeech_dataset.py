"""
PyTorch Dataset for LibriSpeech ASR Corpus.

LibriSpeech is a corpus of approximately 1000 hours of read English speech,
derived from audiobooks of the LibriVox project, and carefully segmented and aligned.

Dataset: https://www.openslr.org/12
"""

import tarfile
import urllib.request
import pickle
import hashlib
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import soundfile as sf
except ImportError:
    try:
        import torchaudio
        AUDIO_BACKEND = 'torchaudio'
    except ImportError:
        raise ImportError(
            "Either soundfile or torchaudio is required for LibriSpeech dataset. "
            "Install with: pip install soundfile  OR  pip install torchaudio"
        )
else:
    AUDIO_BACKEND = 'soundfile'


class LibriSpeechDataset(Dataset):
    """
    PyTorch Dataset for LibriSpeech ASR Corpus.

    The dataset loads audio files (.flac) and their transcripts from the LibriSpeech corpus.
    Audio files are organized as: subset/speaker_id/chapter_id/speaker_id-chapter_id-utterance_id.flac

    Chunk metadata is cached in root_dir/.cache/ to speed up subsequent dataset creation.

    Args:
        root_dir: Path to LibriSpeech root directory
        subsets: List of subsets to load (e.g., ['train-clean-100', 'dev-clean'])
        chunk_size: Number of samples per chunk. If None, returns full utterances.
        overlap: Number of samples to overlap between chunks (default: 0)
        transform: Optional transform to apply to each audio chunk
        include_transcripts: If True, returns (audio, transcript) tuples
        download: If True, download subsets from OpenSLR if not found locally
        device: Device to preload data on ('cpu' or 'cuda'). Note: preloading large datasets
                may consume significant memory. Set to None to load on-demand.
        split: Dataset split - 'train', 'val', or None for all data (default: None).
               When set, splits utterances within the loaded subsets.
        val_fraction: Fraction of utterances to use for validation (default: 0.1)
        seed: Random seed for reproducible train/val splits (default: 42)
    """

    # Available subsets
    AVAILABLE_SUBSETS = [
        'train-clean-100', 'train-clean-360', 'train-other-500',
        'dev-clean', 'dev-other',
        'test-clean', 'test-other'
    ]

    # Download URLs from OpenSLR
    DOWNLOAD_URLS = {
        'train-clean-100': 'https://www.openslr.org/resources/12/train-clean-100.tar.gz',
        'train-clean-360': 'https://www.openslr.org/resources/12/train-clean-360.tar.gz',
        'train-other-500': 'https://www.openslr.org/resources/12/train-other-500.tar.gz',
        'dev-clean': 'https://www.openslr.org/resources/12/dev-clean.tar.gz',
        'dev-other': 'https://www.openslr.org/resources/12/dev-other.tar.gz',
        'test-clean': 'https://www.openslr.org/resources/12/test-clean.tar.gz',
        'test-other': 'https://www.openslr.org/resources/12/test-other.tar.gz',
    }

    SAMPLING_RATE = 16000  # Hz

    def __init__(
        self,
        root_dir: Union[str, Path],
        subsets: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
        overlap: int = 0,
        transform: Optional[callable] = None,
        include_transcripts: bool = False,
        download: bool = True,
        device: Optional[str] = None,
        split: Optional[str] = None,
        val_fraction: float = 0.1,
        seed: int = 42
    ):
        self.root_dir = Path(root_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.transform = transform
        self.include_transcripts = include_transcripts
        self.device = device
        self.download = download
        self.split = split
        self.val_fraction = val_fraction
        self.seed = seed

        # Determine which subsets to load
        if subsets is None:
            # Find available subsets in the directory
            subsets = [s for s in self.AVAILABLE_SUBSETS
                      if (self.root_dir / s).exists()]
            print(self.root_dir.absolute())
            print(subsets)
            if not subsets:
                if download:
                    # Download the smallest subset by default
                    subsets = ['dev-clean']
                    print(f"No subsets found. Will download: {subsets}")
                else:
                    raise ValueError(f"No LibriSpeech subsets found in {root_dir}")
        self.subsets = subsets

        # Download missing subsets if requested
        if download:
            for subset in self.subsets:
                if not (self.root_dir / subset).exists():
                    self._download_subset(subset)

        # Load all utterance paths and transcripts
        self.utterances = []  # List of (audio_path, transcript, speaker_id, chapter_id)
        self._load_metadata()

        if len(self.utterances) == 0:
            raise ValueError("No utterances found in the specified subsets")

        # Apply train/val split if requested
        if split is not None:
            self.utterances = self._get_split_utterances(split, val_fraction, seed)

        # If chunk_size is specified, we'll split utterances into chunks
        if self.chunk_size is not None:
            self.stride = chunk_size - overlap
            # Try to load from cache first
            cache_path = self._get_cache_path()
            if cache_path.exists():
                print(f"Loading cached chunk metadata from {cache_path}")
                with open(cache_path, 'rb') as f:
                    self.chunks = pickle.load(f)
            else:
                print(f"Preparing chunks (this may take a while on first run)...")
                self._prepare_chunks()
                # Save to cache
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.chunks, f)
                print(f"Cached chunk metadata to {cache_path}")
        else:
            # Return full utterances
            self.chunks = [(i, 0, None) for i in range(len(self.utterances))]

        # Optionally preload audio data
        self.preloaded_audio = None
        if device is not None:
            print(f"Preloading audio to {device}... This may take a while.")
            self._preload_audio()

    def _get_cache_path(self) -> Path:
        """Generate a cache file path based on dataset configuration."""
        # Create a hash of the configuration to use as cache key
        split_str = self.split or "all"
        config_str = f"{sorted(self.subsets)}_{self.chunk_size}_{self.overlap}_{len(self.utterances)}_{split_str}_{self.val_fraction}_{self.seed}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        cache_filename = f"chunks_cache_{config_hash}.pkl"
        return self.root_dir / ".cache" / cache_filename

    def _get_split_utterances(self, split: str, val_fraction: float, seed: int) -> List[Dict]:
        """
        Split utterances into train/val sets by speaker to avoid data leakage.

        Args:
            split: 'train' or 'val'
            val_fraction: Fraction of speakers for validation
            seed: Random seed for reproducibility

        Returns:
            List of utterance dictionaries for the requested split
        """
        if split not in ('train', 'val'):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        # Group utterances by speaker (sorted for deterministic ordering)
        speakers = sorted(set(u['speaker_id'] for u in self.utterances))

        # Deterministic shuffle
        rng = np.random.RandomState(seed)
        shuffled_speakers = speakers.copy()
        rng.shuffle(shuffled_speakers)

        # Split speakers
        n_val = max(1, int(len(shuffled_speakers) * val_fraction))
        val_speakers = set(shuffled_speakers[:n_val])
        train_speakers = set(shuffled_speakers[n_val:])

        # Filter utterances
        if split == 'train':
            return [u for u in self.utterances if u['speaker_id'] in train_speakers]
        else:
            return [u for u in self.utterances if u['speaker_id'] in val_speakers]

    def _load_metadata(self):
        """Load all utterance paths and transcripts from the subsets."""
        for subset in self.subsets:
            subset_dir = self.root_dir / subset
            if not subset_dir.exists():
                print(f"Warning: Subset {subset} not found at {subset_dir}")
                continue

            # Find all transcript files (sorted for deterministic ordering)
            trans_files = sorted(subset_dir.glob("*/*/*.trans.txt"))

            for trans_file in trans_files:
                chapter_dir = trans_file.parent
                speaker_id = chapter_dir.parent.name
                chapter_id = chapter_dir.name

                # Read transcripts
                with open(trans_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split(' ', 1)
                        if len(parts) != 2:
                            continue

                        utterance_id, transcript = parts
                        audio_path = chapter_dir / f"{utterance_id}.flac"

                        if audio_path.exists():
                            self.utterances.append({
                                'path': audio_path,
                                'transcript': transcript,
                                'speaker_id': speaker_id,
                                'chapter_id': chapter_id,
                                'utterance_id': utterance_id,
                                'subset': subset
                            })

    def _download_subset(self, subset: str):
        """Download and extract a LibriSpeech subset from OpenSLR."""
        if subset not in self.DOWNLOAD_URLS:
            raise ValueError(f"Unknown subset: {subset}. Available: {list(self.DOWNLOAD_URLS.keys())}")

        url = self.DOWNLOAD_URLS[subset]
        tar_path = self.root_dir / f"{subset}.tar.gz"

        # Create root directory if it doesn't exist
        self.root_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {subset} from {url}...")
        print(f"This may take a while depending on your connection.")

        # Download with progress
        def _progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(downloaded * 100.0 / total_size, 100.0)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')

        try:
            urllib.request.urlretrieve(url, tar_path, reporthook=_progress_hook)
            print()  # New line after progress
            print(f"Downloaded to {tar_path}")

            # Extract the archive
            print(f"Extracting {subset}...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                # Extract to root_dir
                tar.extractall(path=self.root_dir)

            print(f"Extracted to {self.root_dir}")

            # Remove the tar file to save space
            tar_path.unlink()
            print(f"Removed archive {tar_path}")

            print(f"Successfully downloaded and extracted {subset}")

        except Exception as e:
            print(f"\nError downloading {subset}: {e}")
            # Clean up partial downloads
            if tar_path.exists():
                tar_path.unlink()
            raise

    def _prepare_chunks(self):
        """Prepare chunk indices for all utterances."""
        self.chunks = []  # List of (utterance_idx, chunk_idx, num_chunks)

        for i, utterance in enumerate(self.utterances):
            # Load audio to determine length
            audio = self._load_audio(utterance['path'])
            num_samples = len(audio)

            # Calculate number of chunks for this utterance
            if num_samples < self.chunk_size:
                # Skip utterances shorter than chunk_size or pad them
                # For now, we'll skip them
                continue

            num_chunks = (num_samples - self.chunk_size) // self.stride + 1

            for chunk_idx in range(num_chunks):
                self.chunks.append((i, chunk_idx, num_chunks))

    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """Load audio file and return as numpy array."""
        if AUDIO_BACKEND == 'soundfile':
            audio, sr = sf.read(audio_path)
            if sr != self.SAMPLING_RATE:
                raise ValueError(f"Expected sampling rate {self.SAMPLING_RATE}, got {sr}")
        else:  # torchaudio
            audio, sr = torchaudio.load(audio_path)
            if sr != self.SAMPLING_RATE:
                raise ValueError(f"Expected sampling rate {self.SAMPLING_RATE}, got {sr}")
            audio = audio.squeeze().numpy()  # Remove channel dimension

        return audio

    def _preload_audio(self):
        """Preload all audio data into memory."""
        self.preloaded_audio = []
        for utterance in self.utterances:
            audio = self._load_audio(utterance['path'])
            audio_tensor = torch.from_numpy(audio).float()
            if self.device:
                audio_tensor = audio_tensor.to(self.device)
            self.preloaded_audio.append(audio_tensor)

    def __len__(self) -> int:
        """Total number of chunks or utterances."""
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, str]]:
        """
        Get a single audio chunk or full utterance.

        Returns:
            If include_transcripts=False: torch.Tensor of shape (chunk_size,) or (num_samples,)
            If include_transcripts=True: (torch.Tensor, str) tuple of audio and transcript
        """
        utterance_idx, chunk_idx, num_chunks = self.chunks[idx]
        utterance = self.utterances[utterance_idx]

        # Load audio
        if self.preloaded_audio is not None:
            audio = self.preloaded_audio[utterance_idx]
        else:
            audio = self._load_audio(utterance['path'])
            audio = torch.from_numpy(audio).float()
            if self.device:
                audio = audio.to(self.device)

        # Extract chunk if chunk_size is specified
        if self.chunk_size is not None:
            start_idx = chunk_idx * self.stride
            end_idx = start_idx + self.chunk_size
            audio = audio[start_idx:end_idx]

        # Apply transform
        if self.transform is not None:
            audio = self.transform(audio)

        # Return with or without transcript
        if self.include_transcripts:
            return audio, utterance['transcript']
        else:
            return audio

    def get_utterance_info(self, idx: int) -> Dict:
        """
        Get metadata about the utterance containing the chunk at index idx.

        Returns:
            Dictionary with keys: path, transcript, speaker_id, chapter_id, utterance_id, subset
        """
        utterance_idx, chunk_idx, num_chunks = self.chunks[idx]
        return self.utterances[utterance_idx].copy()

    @property
    def sampling_rate(self) -> int:
        """Sampling rate of the LibriSpeech dataset (16 kHz)."""
        return self.SAMPLING_RATE

    @property
    def num_utterances(self) -> int:
        """Number of utterances in the dataset."""
        return len(self.utterances)

    @property
    def num_speakers(self) -> int:
        """Number of unique speakers in the dataset."""
        return len(set(u['speaker_id'] for u in self.utterances))

    def __repr__(self):
        chunk_info = f", chunk_size={self.chunk_size}" if self.chunk_size else ""
        split_info = f", split='{self.split}'" if self.split else ""
        return (
            f"LibriSpeechDataset(subsets={self.subsets}, "
            f"utterances={len(self.utterances):,}, "
            f"speakers={self.num_speakers}, "
            f"chunks={len(self):,}{chunk_info}{split_info})"
        )


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt

    print("Creating LibriSpeech dataset...")

    # Example 1: Load existing data (no download)
    # Create dataset without chunking (full utterances)
    dataset = LibriSpeechDataset(
        root_dir="LibriSpeech",
        include_transcripts=True,
        download=True
    )

    # Example 2: Auto-download if not available (uncomment to try)
    # dataset = LibriSpeechDataset(
    #     root_dir="LibriSpeech",
    #     subsets=['dev-clean'],  # Small subset for testing
    #     include_transcripts=True,
    #     download=True
    # )

    print(dataset)
    print(f"\nSampling rate: {dataset.sampling_rate} Hz")
    print(f"Number of utterances: {dataset.num_utterances}")
    print(f"Number of speakers: {dataset.num_speakers}")
    print(f"Audio backend: {AUDIO_BACKEND}")

    # Get first few utterances
    print(f"\nFirst 5 utterances:")
    for i in range(min(5, len(dataset))):
        audio, transcript = dataset[i]
        info = dataset.get_utterance_info(i)
        duration = len(audio) / dataset.sampling_rate
        print(f"  {i}: speaker={info['speaker_id']}, "
              f"duration={duration:.2f}s, "
              f"transcript={transcript[:50]}...")

    # Create chunked dataset
    print("\n\nCreating chunked dataset...")
    chunked_dataset = LibriSpeechDataset(
        root_dir="LibriSpeech",
        subsets=['train-clean-100'],
        chunk_size=16000,  # 1 second
        overlap=0,
        include_transcripts=False
    )

    print(chunked_dataset)
    print(f"Total chunks: {len(chunked_dataset):,}")

    # Get first few chunks
    print(f"\nFirst 5 chunks:")
    for i in range(min(5, len(chunked_dataset))):
        chunk = chunked_dataset[i]
        info = chunked_dataset.get_utterance_info(i)
        print(f"  Chunk {i}: speaker={info['speaker_id']}, "
              f"shape={chunk.shape}, "
              f"range=[{chunk.min():.6f}, {chunk.max():.6f}]")

    # Plot first 3 utterances
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    for i in range(min(3, len(dataset))):
        audio, transcript = dataset[i]
        info = dataset.get_utterance_info(i)
        time = np.arange(len(audio)) / dataset.sampling_rate

        axes[i].plot(time, audio.numpy() if isinstance(audio, torch.Tensor) else audio)
        axes[i].set_title(f"Speaker {info['speaker_id']} - {transcript[:60]}...")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("librispeech_example.png", dpi=150, bbox_inches="tight")
    print("\nSaved plot to librispeech_example.png")
