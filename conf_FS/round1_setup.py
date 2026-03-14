"""
Forward Selection Step 1: Generate Single Week dataset
- Generate folder for only Week 1, Week 2, ... Week 24
"""

import numpy as np
import os
import shutil
from tqdm import tqdm
from multiprocessing import Pool

def create_single_week_dataset(args):
    week, input_dir, output_base_dir = args
    output_dir = f"{output_base_dir}_{week}"
    
    modalities = ['S2L2A', 'S1GRD', 'DEM', 'WEATHER', 'CDL', 'SOIL']
    
    for modality in modalities:
        src_folder = os.path.join(input_dir, modality)
        if not os.path.exists(src_folder):
            continue
        dst_folder = os.path.join(output_dir, modality)
        os.makedirs(dst_folder, exist_ok=True)
        
        src_files = set(os.listdir(src_folder))
        dst_files = set(os.listdir(dst_folder)) if os.path.exists(dst_folder) else set()
        remaining = src_files - dst_files  # 아직 안 된 파일만!
        
        for filename in remaining:
            data = np.load(os.path.join(src_folder, filename))
            idx = week - 1
            selected_data = data[idx:idx+1, ...]
            np.save(os.path.join(dst_folder, filename), selected_data)
            

def main():
    input_dir = "/scratch/bepk/bkim2/ISA_Yield/processed_data_weekly_24"
    output_base = "processed_data_fs"
    weeks_to_create = list(range(1, 25))
    
    print("=" * 50)
    print("Forward Selection Step 1: Single Week Datasets")
    print("=" * 50)
    print(f"Input: {input_dir}")
    print(f"Weeks: {weeks_to_create}")
    print("=" * 50 + "\n")
    
    args = [(w, input_dir, output_base) for w in weeks_to_create]
    
    with Pool(processes=16) as pool:
        pool.map(create_single_week_dataset, args)
    
    print("=" * 50)
    print("🎉 All done!")
    print("=" * 50)


if __name__ == "__main__":
    main()