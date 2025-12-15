#!/usr/bin/env python3
"""
Extract specific timepoint(s) from all .npy files in processed_data_biweekly_12 subfolders
and save them in a new folder with the same structure.
Copy the yield_geotiffs subfolder as is.

Usage:
    # Extract single timepoint
    python extract_timepoint_9.py --timepoints 9
    
    # Extract multiple timepoints
    python extract_timepoint_9.py --timepoints 8 9 10
    
    # Specify custom source and target directories
    python extract_timepoint_9.py --timepoints 9 --source processed_data_biweekly_12 --target processed_data_t9
"""

import os
import numpy as np
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_timepoints(timepoints, source_dir, target_dir):
    """
    Extract specific timepoint(s) from timeseries data.
    
    Args:
        timepoints: List of timepoint indices to extract (0-indexed)
        source_dir: Path to source directory
        target_dir: Path to target directory
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Create target directory
    target_dir.mkdir(exist_ok=True)
    
    # List of subfolders that contain timeseries data
    data_subfolders = ['CDL', 'DEM', 'S1RTC', 'S2L2A', 'SOIL', 'WEATHER']
    
    # Process each subfolder
    for subfolder in data_subfolders:
        source_subfolder = source_dir / subfolder
        target_subfolder = target_dir / subfolder
        
        if not source_subfolder.exists():
            print(f"Skipping {subfolder} - not found")
            continue
            
        # Create target subfolder
        target_subfolder.mkdir(exist_ok=True)
        
        # Get all .npy files
        npy_files = list(source_subfolder.glob('*.npy'))
        
        print(f"\nProcessing {subfolder}: {len(npy_files)} files")
        
        # Process each file
        for npy_file in tqdm(npy_files, desc=subfolder):
            try:
                # Load the data
                data = np.load(npy_file)
                
                # Check if it has the expected shape (T, C, H, W)
                if len(data.shape) != 4:
                    print(f"  Warning: {npy_file.name} has unexpected shape {data.shape}, skipping")
                    continue
                
                # Extract specified timepoint(s) and maintain (T, C, H, W) shape
                # Always keep the time dimension, even for single timepoint
                data_extracted = data[timepoints]
                
                # Save to target folder
                target_file = target_subfolder / npy_file.name
                np.save(target_file, data_extracted)
                
            except Exception as e:
                print(f"  Error processing {npy_file.name}: {e}")
    
    # Copy yield_geotiffs folder as is
    source_yield = source_dir / 'yield_geotiffs'
    target_yield = target_dir / 'yield_geotiffs'
    
    if source_yield.exists():
        print(f"\nCopying yield_geotiffs folder...")
        if target_yield.exists():
            shutil.rmtree(target_yield)
        shutil.copytree(source_yield, target_yield)
        print(f"Copied yield_geotiffs folder with all contents")
    else:
        print(f"\nWarning: yield_geotiffs folder not found")
    
    # Copy text files (train.txt, test.txt, val.txt, etc.)
    print(f"\nCopying text files...")
    for txt_file in source_dir.glob('*.txt'):
        target_file = target_dir / txt_file.name
        shutil.copy2(txt_file, target_file)
        print(f"  Copied {txt_file.name}")
    
    print(f"\n✓ Extraction complete! Data saved to: {target_dir}")
    print(f"\nSummary:")
    print(f"  - Extracted timepoint(s) {timepoints} from all .npy files")
    print(f"  - New data shape: ({len(timepoints)}, C, H, W) maintaining (T, C, H, W) format")
    print(f"  - Copied yield_geotiffs folder as is")
    print(f"  - Copied all text files")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract specific timepoint(s) from timeseries .npy files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract single timepoint 9
  python extract_timepoint_9.py --timepoints 9
  
  # Extract multiple timepoints
  python extract_timepoint_9.py --timepoints 8 9 10
  
  # Specify custom directories
  python extract_timepoint_9.py --timepoints 9 --source processed_data_biweekly_12 --target processed_data_t9
        """
    )
    
    parser.add_argument(
        '--timepoints', 
        type=int, 
        nargs='+', 
        required=True,
        help='Timepoint index/indices to extract (0-indexed). Can specify one or multiple.'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='processed_data_biweekly_12',
        help='Source directory name (default: processed_data_biweekly_12)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Target directory name (default: auto-generated based on timepoints)'
    )
    
    args = parser.parse_args()
    
    # Generate default target directory name if not specified
    if args.target is None:
        timepoints_str = '_'.join(map(str, args.timepoints))
        args.target = f'processed_biweek_{timepoints_str}'
    
    # Build full paths
    base_dir = Path('/work/mech-ai-scratch/aapowadi/ISA_Yield')
    source_path = base_dir / args.source
    target_path = base_dir / args.target
    
    # Validate source exists
    if not source_path.exists():
        print(f"Error: Source directory '{source_path}' does not exist!")
        exit(1)
    
    print(f"Extracting timepoint(s): {args.timepoints}")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    
    extract_timepoints(args.timepoints, source_path, target_path)
