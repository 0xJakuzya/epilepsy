"""
Module for labeling windows as preictal, interictal, or excluded.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class WindowLabeler:
    """
    Labels windows based on their temporal relationship to seizures.
    
    Labels:
        - preictal (1): Window in [onset - P, onset - SPH]
        - interictal (0): Window far from any seizure (gap >= interictal_gap)
        - excluded (-1): Window in ictal, SPH buffer, postictal, or too close
    """
    
    def __init__(
        self,
        sph: float = 60.0,
        preictal_duration: float = 1800.0,
        postictal_excluded: float = 1800.0,
        interictal_gap: float = 14400.0
    ):
        """
        Args:
            sph: Seizure Prediction Horizon in seconds (default 60s)
            preictal_duration: Preictal period P in seconds (default 30 min)
            postictal_excluded: Postictal excluded period in seconds (default 30 min)
            interictal_gap: Minimum gap from seizures for interictal in seconds (default 4h)
        """
        self.sph = sph
        self.preictal_duration = preictal_duration
        self.postictal_excluded = postictal_excluded
        self.interictal_gap = interictal_gap
    
    def get_window_label(
        self,
        window_start: float,
        window_end: float,
        seizures: List[Dict]
    ) -> Tuple[int, str]:
        """
        Determine label for a single window.
        
        Args:
            window_start: Window start time in seconds (relative to file)
            window_end: Window end time in seconds
            seizures: List of seizure dicts with 'onset_sec' and 'offset_sec'
            
        Returns:
            Tuple of (label, reason)
            - label: 1 (preictal), 0 (interictal), -1 (excluded)
            - reason: String explaining the label
        """
        window_mid = (window_start + window_end) / 2
        
        for seizure in seizures:
            onset = seizure['onset_sec']
            offset = seizure['offset_sec']
            
            # Check if window is during seizure (ictal)
            if window_start < offset and window_end > onset:
                return -1, 'ictal'
            
            # Check if window is in SPH buffer [onset - SPH, onset]
            sph_start = onset - self.sph
            if window_start < onset and window_end > sph_start:
                return -1, 'sph_buffer'
            
            # Check if window is in postictal period [offset, offset + postictal]
            postictal_end = offset + self.postictal_excluded
            if window_start < postictal_end and window_end > offset:
                return -1, 'postictal'
            
            # Check if window is in preictal period [onset - P, onset - SPH]
            preictal_start = onset - self.preictal_duration
            preictal_end = onset - self.sph
            
            if window_start >= preictal_start and window_end <= preictal_end:
                return 1, 'preictal'
            
            # Partial overlap with preictal - use midpoint
            if window_mid >= preictal_start and window_mid < preictal_end:
                return 1, 'preictal_partial'
        
        # Check interictal gap requirement
        min_distance = float('inf')
        for seizure in seizures:
            onset = seizure['onset_sec']
            offset = seizure['offset_sec']
            
            # Distance to onset
            dist_to_onset = abs(window_mid - onset)
            # Distance to offset
            dist_to_offset = abs(window_mid - offset)
            
            min_distance = min(min_distance, dist_to_onset, dist_to_offset)
        
        if min_distance >= self.interictal_gap:
            return 0, 'interictal'
        else:
            return -1, f'too_close_{min_distance:.0f}s'
    
    def label_windows_for_file(
        self,
        window_times: List[Tuple[float, float]],
        seizures: List[Dict],
        patient: str,
        edf_file: str
    ) -> pd.DataFrame:
        """
        Label all windows from a single EDF file.
        
        Args:
            window_times: List of (t_start, t_end) tuples
            seizures: List of seizure dicts for this file
            patient: Patient ID
            edf_file: EDF filename
            
        Returns:
            DataFrame with window labels
        """
        records = []
        
        for i, (t_start, t_end) in enumerate(window_times):
            label, reason = self.get_window_label(t_start, t_end, seizures)
            
            records.append({
                'window_id': i,
                'patient': patient,
                'edf_file': edf_file,
                't_start': t_start,
                't_end': t_end,
                'label': label,
                'label_reason': reason
            })
        
        return pd.DataFrame(records)


def label_windows(
    windows_df: pd.DataFrame,
    seizure_index: pd.DataFrame,
    sph: float = 60.0,
    preictal_duration: float = 1800.0,
    postictal_excluded: float = 1800.0,
    interictal_gap: float = 14400.0
) -> pd.DataFrame:
    """
    Label all windows based on seizure index.
    
    Args:
        windows_df: DataFrame with window metadata (patient, edf_file, t_start, t_end)
        seizure_index: DataFrame with seizure info (patient, edf_file, onset_sec, offset_sec)
        sph: Seizure Prediction Horizon
        preictal_duration: Preictal period P
        postictal_excluded: Postictal excluded period
        interictal_gap: Minimum gap for interictal
        
    Returns:
        DataFrame with added 'label' and 'label_reason' columns
    """
    labeler = WindowLabeler(
        sph=sph,
        preictal_duration=preictal_duration,
        postictal_excluded=postictal_excluded,
        interictal_gap=interictal_gap
    )
    
    labels = []
    reasons = []
    
    for _, row in windows_df.iterrows():
        # Get seizures for this file
        file_seizures = seizure_index[
            (seizure_index['patient'] == row['patient']) &
            (seizure_index['edf_file'] == row['edf_file'])
        ]
        
        seizures = file_seizures[['onset_sec', 'offset_sec']].to_dict('records')
        
        label, reason = labeler.get_window_label(
            row['t_start'], row['t_end'], seizures
        )
        
        labels.append(label)
        reasons.append(reason)
    
    result = windows_df.copy()
    result['label'] = labels
    result['label_reason'] = reasons
    
    return result


def get_window_label(
    window_start: float,
    window_end: float,
    seizures: List[Dict],
    sph: float = 60.0,
    preictal_duration: float = 1800.0,
    postictal_excluded: float = 1800.0,
    interictal_gap: float = 14400.0
) -> Tuple[int, str]:
    """
    Convenience function to get label for a single window.
    """
    labeler = WindowLabeler(
        sph=sph,
        preictal_duration=preictal_duration,
        postictal_excluded=postictal_excluded,
        interictal_gap=interictal_gap
    )
    return labeler.get_window_label(window_start, window_end, seizures)


def get_label_statistics(labeled_df: pd.DataFrame) -> Dict:
    """
    Get statistics about labeled windows.
    
    Args:
        labeled_df: DataFrame with 'label' column
        
    Returns:
        Dictionary with label counts and ratios
    """
    total = len(labeled_df)
    
    preictal = (labeled_df['label'] == 1).sum()
    interictal = (labeled_df['label'] == 0).sum()
    excluded = (labeled_df['label'] == -1).sum()
    
    return {
        'total_windows': total,
        'preictal': preictal,
        'interictal': interictal,
        'excluded': excluded,
        'preictal_ratio': preictal / total if total > 0 else 0,
        'interictal_ratio': interictal / total if total > 0 else 0,
        'excluded_ratio': excluded / total if total > 0 else 0,
        'class_ratio': interictal / preictal if preictal > 0 else float('inf')
    }


if __name__ == "__main__":
    # Test with synthetic data
    
    # Simulate a file with one seizure at 2000 seconds
    seizures = [{'onset_sec': 2000, 'offset_sec': 2040}]
    
    labeler = WindowLabeler(
        sph=60,
        preictal_duration=1800,  # 30 min
        postictal_excluded=1800,  # 30 min
        interictal_gap=14400  # 4 hours
    )
    
    # Test various window positions
    test_windows = [
        (0, 4, "far before - should be excluded (< 4h gap)"),
        (100, 104, "early - should be excluded"),
        (200, 204, "preictal start - should be preictal"),
        (1000, 1004, "mid preictal - should be preictal"),
        (1900, 1904, "late preictal - should be preictal"),
        (1940, 1944, "SPH buffer - should be excluded"),
        (1980, 1984, "SPH buffer - should be excluded"),
        (2000, 2004, "ictal - should be excluded"),
        (2020, 2024, "ictal - should be excluded"),
        (2040, 2044, "postictal - should be excluded"),
        (2100, 2104, "postictal - should be excluded"),
        (3900, 3904, "after postictal but < 4h - should be excluded"),
    ]
    
    print("Window labeling test:")
    print("-" * 60)
    for t_start, t_end, description in test_windows:
        label, reason = labeler.get_window_label(t_start, t_end, seizures)
        label_name = {1: 'preictal', 0: 'interictal', -1: 'excluded'}[label]
        print(f"t={t_start:5.0f}-{t_end:5.0f}s: {label_name:10s} ({reason:20s}) | {description}")
