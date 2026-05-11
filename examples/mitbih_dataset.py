"""
PyTorch Dataset for MIT-BIH Arrhythmia Database.

The MIT-BIH Arrhythmia Database contains 48 half-hour excerpts of two-channel
ambulatory ECG recordings, digitized at 360 samples per second per channel
with 11-bit resolution over a 10 mV range.

Dataset: https://archive.physionet.org/physiobank/database/mitdb/
"""

from pathlib import Path
from typing import Optional, List, Union, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import wfdb
except ImportError:
    raise ImportError(
        "wfdb package is required for MIT-BIH dataset. "
        "Install it with: pip install wfdb"
    )


class MITBIHDataset(Dataset):
    """
    PyTorch Dataset for MIT-BIH Arrhythmia Database.

    Args:
        data_dir: Path to directory containing MIT-BIH records, or None to download from PhysioNet
        record_names: List of record names to load (e.g., ['100', '101']). If None, loads all available records.
        chunk_size: Number of samples per chunk (default: 360 for 1 second at 360Hz)
        channels: List of channels to load (0 or 1 for the two ECG leads, None = all channels)
        overlap: Number of samples to overlap between chunks (default: 0)
        transform: Optional transform to apply to each chunk
        download: If True, download records from PhysioNet if not found locally
        device: Device to store data on ('cpu' or 'cuda')
        split: Dataset split - 'train', 'val', or None for all data (default: None)
        val_fraction: Fraction of records to use for validation (default: 0.2)
        seed: Random seed for reproducible train/val splits (default: 42)
    """

    # MIT-BIH record names (48 half-hour recordings)
    ALL_RECORDS = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        #'111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        #'122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        #'209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        #'222', '223', '228', '230', '231', '232', '233', '234'
    ]

    SAMPLING_RATE = 360  # Hz

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        record_names: Optional[List[str]] = None,
        chunk_size: int = 360,
        channels: Optional[List[int]] = None,
        overlap: int = 0,
        transform: Optional[callable] = None,
        download: bool = True,
        device: str = "cpu",
        split: Optional[str] = None,
        val_fraction: float = 0.2,
        seed: int = 42
    ):
        self.data_dir = Path(data_dir) if data_dir else None
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.transform = transform
        self.device = device
        self.split = split
        self.val_fraction = val_fraction
        self.seed = seed

        # Determine which records to load
        if record_names is None:
            record_names = self.ALL_RECORDS.copy()

        # Apply train/val split if requested
        if split is not None:
            record_names = self._get_split_records(record_names, split, val_fraction, seed)

        self.record_names = record_names

        # Determine which channels to load
        if channels is None:
            self.channels = [0, 1]  # Both leads
        else:
            self.channels = channels
        self.num_channels = len(self.channels)

        # Load all records
        self.records = []
        self.annotations = []

        for record_name in self.record_names:
            try:
                record, annotation = self._load_record(record_name, download)
                self.records.append(record)
                self.annotations.append(annotation)
            except Exception as e:
                print(f"Warning: Failed to load record {record_name}: {e}")
                continue

        if len(self.records) == 0:
            raise ValueError("No records were successfully loaded")

        # Concatenate all records along time dimension for each channel
        # Each record has shape (num_samples, num_channels)
        all_signals = []
        for record in self.records:
            # Select requested channels and transpose to (num_channels, num_samples)
            signal = torch.from_numpy(record.p_signal[:, self.channels].T).float()
            all_signals.append(signal)

        # Concatenate along time dimension: (num_channels, total_samples)
        self.data = torch.cat(all_signals, dim=1).to(device)
        self.num_samples = self.data.shape[1]

        # Calculate number of chunks per channel
        self.stride = chunk_size - overlap
        self.num_chunks_per_channel = (self.num_samples - chunk_size) // self.stride + 1

    def _load_record(
        self, record_name: str, download: bool
    ) -> Tuple[wfdb.Record, wfdb.Annotation]:
        """Load a single MIT-BIH record and its annotations."""
        if self.data_dir and (self.data_dir / record_name).with_suffix('.hea').exists():
            # Load from local directory
            record_path = str(self.data_dir / record_name)
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
        elif download:
            # Download from PhysioNet and save to data_dir
            if self.data_dir is None:
                # If no data_dir specified, default to 'mitbih' in current directory
                self.data_dir = Path('mitbih')

            # Create data directory if it doesn't exist
            self.data_dir.mkdir(parents=True, exist_ok=True)

            # Download from PhysioNet
            record = wfdb.rdrecord(record_name, pn_dir='mitdb')
            annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')

            # Save to local directory
            wfdb.wrsamp(
                record_name=record_name,
                fs=record.fs,
                units=record.units,
                sig_name=record.sig_name,
                p_signal=record.p_signal,
                fmt=record.fmt,
                write_dir=str(self.data_dir)
            )
            # Save annotation
            wfdb.wrann(
                record_name=record_name,
                extension='atr',
                sample=annotation.sample,
                symbol=annotation.symbol,
                write_dir=str(self.data_dir)
            )

            print(f"Saved record {record_name} to {self.data_dir}")
        else:
            raise FileNotFoundError(
                f"Record {record_name} not found in {self.data_dir} and download=False"
            )

        return record, annotation

    def _get_split_records(
        self, record_names: List[str], split: str, val_fraction: float, seed: int
    ) -> List[str]:
        """
        Split records into train/val sets.

        Uses a deterministic shuffle based on the seed to ensure reproducibility.

        Args:
            record_names: List of all record names
            split: 'train' or 'val'
            val_fraction: Fraction of records for validation
            seed: Random seed for reproducibility

        Returns:
            List of record names for the requested split
        """
        if split not in ('train', 'val'):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        # Deterministic shuffle
        rng = np.random.RandomState(seed)
        shuffled_records = record_names.copy()
        rng.shuffle(shuffled_records)

        # Split
        n_val = max(1, int(len(shuffled_records) * val_fraction))
        val_records = shuffled_records[:n_val]
        train_records = shuffled_records[n_val:]

        if split == 'train':
            return train_records
        else:
            return val_records

    def __len__(self) -> int:
        """Total number of single-channel chunks across all channels."""
        return self.num_channels * self.num_chunks_per_channel

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single-channel chunk of ECG data.

        Returns:
            torch.Tensor: Shape (chunk_size,) - single channel ECG chunk
        """
        # Determine which channel and which chunk within that channel
        channel_idx = idx // self.num_chunks_per_channel
        chunk_idx = idx % self.num_chunks_per_channel

        start_idx = chunk_idx * self.stride
        end_idx = start_idx + self.chunk_size

        chunk = self.data[channel_idx, start_idx:end_idx]

        if self.transform is not None:
            chunk = self.transform(chunk)

        return chunk

    def get_annotations_in_range(
        self, start_sample: int, end_sample: int
    ) -> List[Tuple[int, str]]:
        """
        Get all annotations within a sample range.

        Args:
            start_sample: Starting sample index
            end_sample: Ending sample index

        Returns:
            List of (sample_index, annotation_symbol) tuples
        """
        annotations = []
        cumulative_samples = 0

        for record, annotation in zip(self.records, self.annotations):
            record_length = record.sig_len
            record_end = cumulative_samples + record_length

            # Check if this record overlaps with our range
            if record_end > start_sample and cumulative_samples < end_sample:
                # Find annotations in this record that fall within our range
                for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
                    global_idx = cumulative_samples + sample_idx
                    if start_sample <= global_idx < end_sample:
                        annotations.append((global_idx, symbol))

            cumulative_samples = record_end

            # Early exit if we've passed the end of our range
            if cumulative_samples >= end_sample:
                break

        return annotations

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of the MIT-BIH database (360 Hz)."""
        return self.SAMPLING_RATE

    def __repr__(self):
        split_info = f", split='{self.split}'" if self.split else ""
        return (
            f"MITBIHDataset(records={len(self.records)}, "
            f"channels={self.num_channels}, samples={self.num_samples:,}, "
            f"chunks_per_channel={self.num_chunks_per_channel:,}, "
            f"total_chunks={len(self):,}, chunk_size={self.chunk_size}{split_info})"
        )


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt

    print("Creating MIT-BIH dataset...")
    print("This will download data from PhysioNet on first run.\n")

    # Create dataset with first 3 records for quick testing
    dataset = MITBIHDataset(
        data_dir="mitbih",
        chunk_size=360,  # 1 second at 360 Hz
        download=True
    )

    print(dataset)
    print(f"\nSampling rate: {dataset.sampling_rate} Hz")
    print(f"Number of records loaded: {len(dataset.records)}")

    # Print info about each record
    print(f"\nRecord details:")
    for i, record in enumerate(dataset.records):
        print(f"  Record {dataset.record_names[i]}: "
              f"{record.sig_len:,} samples ({record.sig_len/dataset.sampling_rate:.1f} seconds), "
              f"{len(dataset.annotations[i].symbol)} annotations")

    # Get first few chunks
    print(f"\nFirst 5 chunks:")
    for i in range(5):
        chunk = dataset[i]
        channel_idx = i // dataset.num_chunks_per_channel
        chunk_idx = i % dataset.num_chunks_per_channel
        print(f"  Chunk {i}: channel={channel_idx}, chunk_in_channel={chunk_idx}, "
              f"shape={chunk.shape}, range=[{chunk.min():.6f}, {chunk.max():.6f}]")

    # Get annotations for first chunk
    start_idx = 0
    end_idx = dataset.chunk_size
    annotations = dataset.get_annotations_in_range(start_idx, end_idx)
    print(f"\nAnnotations in first chunk (samples {start_idx}-{end_idx}):")
    for sample, symbol in annotations[:10]:  # Show first 10
        print(f"  Sample {sample}: {symbol}")

    # Plot first chunk from both channels
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for i in range(2):
        chunk = dataset[i * dataset.num_chunks_per_channel]  # First chunk of channel i
        time = np.arange(len(chunk)) / dataset.sampling_rate
        axes[i].plot(time, chunk.numpy())
        axes[i].set_title(f"Channel {i} - First chunk (Record 100)")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Amplitude (mV)")
        axes[i].grid(True, alpha=0.3)

        # Mark annotations
        annotations = dataset.get_annotations_in_range(0, dataset.chunk_size)
        for sample, symbol in annotations:
            axes[i].axvline(x=sample/dataset.sampling_rate, color='r',
                           alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig("mitbih_example.png", dpi=150, bbox_inches="tight")
    print("\nSaved plot to mitbih_example.png")
