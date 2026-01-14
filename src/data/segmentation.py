"""
Module for signal segmentation into windows.
"""

from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


def segment_signal(
    data: np.ndarray,
    sfreq: float,
    window_length: float,
    window_step: float
) -> Generator[Tuple[np.ndarray, float, float], None, None]:
    """
    Segment signal into overlapping windows.
    
    Args:
        data: Signal data (n_channels, n_samples)
        sfreq: Sampling frequency
        window_length: Window length in seconds
        window_step: Step size in seconds
        
    Yields:
        Tuple of (window_data, t_start, t_end) for each window
    """
    n_channels, n_samples = data.shape
    
    window_samples = int(window_length * sfreq)
    step_samples = int(window_step * sfreq)
    
    if window_samples > n_samples:
        logger.warning(f"Window size {window_samples} > signal length {n_samples}")
        return
    
    start_idx = 0
    while start_idx + window_samples <= n_samples:
        end_idx = start_idx + window_samples
        
        window_data = data[:, start_idx:end_idx]
        t_start = start_idx / sfreq
        t_end = end_idx / sfreq
        
        yield window_data, t_start, t_end
        
        start_idx += step_samples


def create_windows(
    data: np.ndarray,
    sfreq: float,
    window_length: float,
    window_step: float,
    patient: str,
    edf_file: str,
    file_offset: float = 0.0
) -> List[Dict]:
    """
    Create list of window metadata with data references.
    
    Args:
        data: Signal data (n_channels, n_samples)
        sfreq: Sampling frequency
        window_length: Window length in seconds
        window_step: Step size in seconds
        patient: Patient ID
        edf_file: EDF filename
        file_offset: Time offset of file in continuous recording
        
    Returns:
        List of window dictionaries with metadata
    """
    windows = []
    
    for window_data, t_start, t_end in segment_signal(
        data, sfreq, window_length, window_step
    ):
        windows.append({
            'patient': patient,
            'edf_file': edf_file,
            't_start': t_start,
            't_end': t_end,
            't_start_global': file_offset + t_start,
            't_end_global': file_offset + t_end,
            'data': window_data,
            'sfreq': sfreq
        })
    
    return windows


def create_window_index(
    data: np.ndarray,
    sfreq: float,
    window_length: float,
    window_step: float,
    patient: str,
    edf_file: str
) -> pd.DataFrame:
    """
    Create DataFrame index of windows (without data).
    
    Args:
        data: Signal data (n_channels, n_samples)
        sfreq: Sampling frequency
        window_length: Window length in seconds
        window_step: Step size in seconds
        patient: Patient ID
        edf_file: EDF filename
        
    Returns:
        DataFrame with window metadata
    """
    n_samples = data.shape[-1]
    window_samples = int(window_length * sfreq)
    step_samples = int(window_step * sfreq)
    
    records = []
    window_id = 0
    start_idx = 0
    
    while start_idx + window_samples <= n_samples:
        end_idx = start_idx + window_samples
        
        records.append({
            'window_id': window_id,
            'patient': patient,
            'edf_file': edf_file,
            'start_sample': start_idx,
            'end_sample': end_idx,
            't_start': start_idx / sfreq,
            't_end': end_idx / sfreq
        })
        
        window_id += 1
        start_idx += step_samples
    
    return pd.DataFrame(records)


def get_window_data(
    data: np.ndarray,
    start_sample: int,
    end_sample: int
) -> np.ndarray:
    """
    Extract window data from full signal.
    
    Args:
        data: Full signal data (n_channels, n_samples)
        start_sample: Start sample index
        end_sample: End sample index
        
    Returns:
        Window data (n_channels, window_samples)
    """
    return data[:, start_sample:end_sample]


class WindowGenerator:
    """
    Generator class for creating windows from multiple EDF files.
    """
    
    def __init__(
        self,
        window_length: float,
        window_step: float,
        sfreq: float = 256.0
    ):
        self.window_length = window_length
        self.window_step = window_step
        self.sfreq = sfreq
        self.window_samples = int(window_length * sfreq)
        self.step_samples = int(window_step * sfreq)
    
    def generate_from_data(
        self,
        data: np.ndarray,
        patient: str,
        edf_file: str
    ) -> Generator[Dict, None, None]:
        """
        Generate windows from preprocessed data.
        
        Args:
            data: Preprocessed signal data
            patient: Patient ID
            edf_file: EDF filename
            
        Yields:
            Window dictionaries
        """
        for window_data, t_start, t_end in segment_signal(
            data, self.sfreq, self.window_length, self.window_step
        ):
            yield {
                'patient': patient,
                'edf_file': edf_file,
                't_start': t_start,
                't_end': t_end,
                'data': window_data
            }
    
    def count_windows(self, n_samples: int) -> int:
        """Calculate number of windows for given signal length."""
        if n_samples < self.window_samples:
            return 0
        return (n_samples - self.window_samples) // self.step_samples + 1


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    
    # Simulate 1 hour of 18-channel EEG at 256 Hz
    sfreq = 256
    duration = 3600  # 1 hour
    n_channels = 18
    n_samples = duration * sfreq
    
    data = np.random.randn(n_channels, n_samples)
    
    # Create windows
    window_length = 4.0
    window_step = 2.0
    
    windows = create_windows(
        data, sfreq, window_length, window_step,
        patient='chb01', edf_file='chb01_01.edf'
    )
    
    print(f"Created {len(windows)} windows")
    print(f"Window shape: {windows[0]['data'].shape}")
    print(f"First window: t={windows[0]['t_start']:.1f}-{windows[0]['t_end']:.1f}s")
    print(f"Last window: t={windows[-1]['t_start']:.1f}-{windows[-1]['t_end']:.1f}s")
