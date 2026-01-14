"""
Module for parsing CHB-MIT summary files and building seizure index.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


def parse_summary_file(summary_path: Path) -> Dict:
    """
    Parse a CHB-MIT summary file and extract seizure information.
    
    Args:
        summary_path: Path to the *-summary.txt file
        
    Returns:
        Dictionary with:
            - sampling_rate: int
            - channels: List[str]
            - files: List[Dict] with file info and seizures
    """
    with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    result = {
        'sampling_rate': None,
        'channels': [],
        'files': []
    }
    
    # Extract sampling rate
    sr_match = re.search(r'Data Sampling Rate:\s*(\d+)\s*Hz', content)
    if sr_match:
        result['sampling_rate'] = int(sr_match.group(1))
    
    # Extract channels
    channel_pattern = re.compile(r'Channel\s+\d+:\s*(\S+)')
    result['channels'] = channel_pattern.findall(content)
    
    # Split by file sections
    file_sections = re.split(r'\n(?=File Name:)', content)
    
    for section in file_sections:
        if 'File Name:' not in section:
            continue
            
        file_info = parse_file_section(section)
        if file_info:
            result['files'].append(file_info)
    
    return result


def parse_file_section(section: str) -> Optional[Dict]:
    """
    Parse a single file section from summary.
    
    Args:
        section: Text section for one EDF file
        
    Returns:
        Dictionary with file info and seizures, or None if parsing fails
    """
    # Extract file name
    name_match = re.search(r'File Name:\s*(\S+)', section)
    if not name_match:
        return None
    
    file_info = {
        'file_name': name_match.group(1),
        'start_time': None,
        'end_time': None,
        'num_seizures': 0,
        'seizures': []
    }
    
    # Extract times
    start_match = re.search(r'File Start Time:\s*(\S+)', section)
    end_match = re.search(r'File End Time:\s*(\S+)', section)
    if start_match:
        file_info['start_time'] = start_match.group(1)
    if end_match:
        file_info['end_time'] = end_match.group(1)
    
    # Extract number of seizures
    num_match = re.search(r'Number of Seizures in File:\s*(\d+)', section)
    if num_match:
        file_info['num_seizures'] = int(num_match.group(1))
    
    # Extract seizure times (can have multiple seizures per file)
    # Pattern handles both "Seizure Start Time:" and "Seizure 1 Start Time:"
    seizure_pattern = re.compile(
        r'Seizure\s*\d*\s*Start Time:\s*(\d+)\s*seconds\s*\n'
        r'Seizure\s*\d*\s*End Time:\s*(\d+)\s*seconds',
        re.IGNORECASE
    )
    
    for match in seizure_pattern.finditer(section):
        file_info['seizures'].append({
            'onset_sec': int(match.group(1)),
            'offset_sec': int(match.group(2))
        })
    
    return file_info


def build_seizure_index(
    data_root: Path,
    patients: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Build a complete seizure index from CHB-MIT dataset.
    
    Args:
        data_root: Root path to CHB-MIT dataset (containing chbXX folders)
        patients: Optional list of patient IDs to process (e.g., ['chb01', 'chb02'])
                  If None, processes all available patients
                  
    Returns:
        DataFrame with columns:
            - patient: str (e.g., 'chb01')
            - edf_file: str (e.g., 'chb01_03.edf')
            - seizure_id: int (global seizure ID for patient)
            - onset_sec: int (seizure start in seconds from file start)
            - offset_sec: int (seizure end in seconds from file start)
            - duration_sec: int (seizure duration)
            - file_start_time: str (file start time)
    """
    data_root = Path(data_root)
    records = []
    
    # Find all patient directories
    if patients is None:
        patient_dirs = sorted([d for d in data_root.iterdir() 
                               if d.is_dir() and d.name.startswith('chb')])
    else:
        patient_dirs = [data_root / p for p in patients if (data_root / p).exists()]
    
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        
        # Find summary file
        summary_files = list(patient_dir.glob('*-summary.txt'))
        if not summary_files:
            logger.warning(f"No summary file found for {patient_id}")
            continue
        
        summary_path = summary_files[0]
        logger.info(f"Processing {patient_id}: {summary_path.name}")
        
        try:
            summary_data = parse_summary_file(summary_path)
        except Exception as e:
            logger.error(f"Error parsing {summary_path}: {e}")
            continue
        
        # Build seizure records
        seizure_counter = 0
        for file_info in summary_data['files']:
            edf_path = patient_dir / file_info['file_name']
            
            for seizure in file_info['seizures']:
                seizure_counter += 1
                records.append({
                    'patient': patient_id,
                    'edf_file': file_info['file_name'],
                    'edf_path': str(edf_path),
                    'seizure_id': seizure_counter,
                    'onset_sec': seizure['onset_sec'],
                    'offset_sec': seizure['offset_sec'],
                    'duration_sec': seizure['offset_sec'] - seizure['onset_sec'],
                    'file_start_time': file_info['start_time'],
                    'sampling_rate': summary_data['sampling_rate']
                })
    
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        logger.info(f"Built index with {len(df)} seizures from {df['patient'].nunique()} patients")
    else:
        logger.warning("No seizures found in dataset")
    
    return df


def build_file_index(
    data_root: Path,
    patients: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Build an index of all EDF files (with and without seizures).
    
    Args:
        data_root: Root path to CHB-MIT dataset
        patients: Optional list of patient IDs
        
    Returns:
        DataFrame with all EDF files and their metadata
    """
    data_root = Path(data_root)
    records = []
    
    if patients is None:
        patient_dirs = sorted([d for d in data_root.iterdir() 
                               if d.is_dir() and d.name.startswith('chb')])
    else:
        patient_dirs = [data_root / p for p in patients if (data_root / p).exists()]
    
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        
        summary_files = list(patient_dir.glob('*-summary.txt'))
        if not summary_files:
            continue
        
        try:
            summary_data = parse_summary_file(summary_files[0])
        except Exception as e:
            logger.error(f"Error parsing summary for {patient_id}: {e}")
            continue
        
        for file_info in summary_data['files']:
            edf_path = patient_dir / file_info['file_name']
            
            records.append({
                'patient': patient_id,
                'edf_file': file_info['file_name'],
                'edf_path': str(edf_path),
                'file_exists': edf_path.exists(),
                'start_time': file_info['start_time'],
                'end_time': file_info['end_time'],
                'num_seizures': file_info['num_seizures'],
                'seizures': file_info['seizures'],
                'sampling_rate': summary_data['sampling_rate'],
                'channels': summary_data['channels']
            })
    
    return pd.DataFrame(records)


def save_index(df: pd.DataFrame, output_path: Path) -> None:
    """Save seizure index to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved index to {output_path}")


if __name__ == "__main__":
    # Quick test
    import yaml
    
    config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_root = Path(config['paths']['data_root'])
    output_dir = Path(config['paths']['output_dir'])
    
    # Build and save seizure index
    seizure_index = build_seizure_index(data_root)
    save_index(seizure_index, output_dir / "seizure_index.csv")
    
    print(seizure_index)
