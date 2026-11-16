"""
Signal Preprocessing and Biosignal Epoching for Sleep Staging
Author: Vaibhav Mistry

Provides utilities for reading Polysomnography (PSG) European Data Format (EDF)
files, parsing hypnogram annotations, segmenting into standard 30-second epochs,
and normalizing multi-channel physiological data (EEG, EOG, EMG).
"""

import os
from typing import List, Tuple, Dict, Optional
import numpy as np
import mne
from sklearn.preprocessing import StandardScaler

# Standard AASM Sleep Staging mapping (Wake, N1, N2, N3, REM)
STAGE_MAP: Dict[str, int] = {
    'W': 0,   # Wake
    'N1': 1,  # Non-REM Stage 1 (Light sleep)
    'N2': 2,  # Non-REM Stage 2
    'N3': 3,  # Non-REM Stage 3 (Slow Wave Sleep / Deep sleep)
    'R': 4    # Rapid Eye Movement (REM)
}

STAGE_NAMES: List[str] = ['W', 'N1', 'N2', 'N3', 'R']
PREFERRED_CHANNELS: List[str] = ['EEG Fpz-Cz', 'EOG horizontal', 'EMG submental']


def load_raw_psg(
    psg_path: str,
    hypnogram_path: str,
    channels: Optional[List[str]] = None
) -> Tuple[mne.io.Raw, mne.Annotations]:
    """
    Load a raw EDF recording and its corresponding hypnogram annotations.

    Args:
        psg_path: Filepath to the PSG EDF file.
        hypnogram_path: Filepath to the Hypnogram EDF file.
        channels: List of target channels to select. Defaults to PREFERRED_CHANNELS.

    Returns:
        Tuple containing the filtered MNE Raw object and MNE Annotations.
    """
    if channels is None:
        channels = PREFERRED_CHANNELS

    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    annotations = mne.read_annotations(hypnogram_path)
    raw.set_annotations(annotations)

    available = [ch for ch in channels if ch in raw.ch_names]
    if not available:
        raise ValueError(f"None of preferred channels {channels} found in {raw.ch_names}")

    raw.pick_channels(available)
    return raw, annotations


def segment_into_epochs(
    raw: mne.io.Raw,
    epoch_duration_sec: int = 30
) -> np.ndarray:
    """
    Segment continuous multi-channel signal data into fixed-duration clinical epochs.

    Args:
        raw: Preloaded MNE Raw object.
        epoch_duration_sec: Length of each epoch in seconds (standard clinical length = 30s).

    Returns:
        Numpy array of shape (n_epochs, time_samples, n_channels).
    """
    sfreq = raw.info['sfreq']
    epoch_len = int(sfreq * epoch_duration_sec)
    data = raw.get_data()
    n_channels = data.shape[0]

    n_epochs = data.shape[1] // epoch_len
    truncated = data[:, :n_epochs * epoch_len]
    # Reshape to (n_epochs, epoch_len, n_channels)
    epochs = truncated.reshape(n_channels, n_epochs, epoch_len).transpose(1, 2, 0)
    return epochs


def extract_stage_labels(
    annotations: mne.Annotations,
    stage_map: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """
    Extract integer stage labels from MNE annotations matching the sleep stage map.

    Args:
        annotations: MNE Annotations extracted from hypnogram.
        stage_map: Dictionary mapping stage code to integer.

    Returns:
        1D numpy array of stage labels.
    """
    if stage_map is None:
        stage_map = STAGE_MAP

    labels = [
        stage_map[ann['description'].split()[-1]]
        for ann in annotations
        if 'Sleep stage' in ann['description'] and ann['description'].split()[-1] in stage_map
    ]
    return np.array(labels)


def normalize_epochs(epochs: np.ndarray) -> np.ndarray:
    """
    Apply standard Z-score normalization across all epochs.

    Args:
        epochs: Numpy array of shape (n_epochs, time_steps, n_channels).

    Returns:
        Normalized numpy array of the same shape.
    """
    original_shape = epochs.shape
    flattened = epochs.reshape(original_shape[0], -1)
    normalized = StandardScaler().fit_transform(flattened)
    return normalized.reshape(original_shape)


def load_multi_subject_dataset(
    data_dir: str,
    preferred_channels: Optional[List[str]] = None,
    epoch_duration_sec: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch load, preprocess, and concatenate PSG and hypnograms across multiple subjects.

    Args:
        data_dir: Path to directory containing PSG and Hypnogram .edf files.
        preferred_channels: Biosignal channels to retain.
        epoch_duration_sec: Epoch length in seconds (default: 30).

    Returns:
        Tuple of (X_all, y_all) ready for deep learning training.
    """
    if preferred_channels is None:
        preferred_channels = PREFERRED_CHANNELS

    psg_files = sorted([f for f in os.listdir(data_dir) if f.endswith("PSG.edf")])
    hypno_files = sorted([f for f in os.listdir(data_dir) if f.endswith("Hypnogram.edf")])

    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    print(f"Discovered {len(psg_files)} PSG recordings and {len(hypno_files)} Hypnogram files.")

    for psg, hyp in zip(psg_files, hypno_files):
        try:
            psg_path = os.path.join(data_dir, psg)
            hyp_path = os.path.join(data_dir, hyp)

            raw, annotations = load_raw_psg(psg_path, hyp_path, preferred_channels)
            epochs = segment_into_epochs(raw, epoch_duration_sec)
            labels = extract_stage_labels(annotations)

            min_len = min(len(epochs), len(labels))
            if min_len < 1:
                print(f"[Warning] Insufficient valid epochs in {psg}. Skipping.")
                continue

            epochs = epochs[:min_len]
            labels = labels[:min_len]

            epochs = normalize_epochs(epochs)
            X_all.append(epochs)
            y_all.append(labels)
            print(f"Loaded {psg}: {min_len} epochs processed.")

        except Exception as e:
            print(f"[Error] Processing {psg}: {e}")

    if not X_all:
        raise ValueError(f"No usable PSG data could be extracted from {data_dir}.")

    return np.vstack(X_all), np.concatenate(y_all)
