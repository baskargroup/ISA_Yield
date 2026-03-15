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
    os.makedirs(output_dir, exist_ok=True)
    
    modalities = ['S2L2A', 'S1GRD', 'DEM', 'WEATHER', 'CDL', 'SOIL']
    
    for modality in modalities:
        src_folder = os.path.join(input_dir, modality)
        if not os.path.exists(src_folder):
            continue
        dst_folder = os.path.join(output_dir, modality)
        os.makedirs(dst_folder, exist_ok=True)
        
        for filename in os.listdir(src_folder):
            data = np.load(os.path.join(src_folder, filename))
            idx = week - 1
            selected_data = data[idx:idx+1, ...]
            np.save(os.path.join(dst_folder, filename), selected_data)
    
    # yield_geotiffs 복사
    yield_src = os.path.join(input_dir, "yield_geotiffs")
    yield_dst = os.path.join(output_dir, "yield_geotiffs")
    if os.path.exists(yield_src):
        if os.path.exists(yield_dst):
            shutil.rmtree(yield_dst)
        shutil.copytree(yield_src, yield_dst)
    
    # txt 파일 복사
    for txt_file in os.listdir(input_dir):
        if txt_file.endswith('.txt'):
            shutil.copy(os.path.join(input_dir, txt_file),
                       os.path.join(output_dir, txt_file))
    
    print(f"✅ Week {week} 완료!")


def main():
    input_dir = "processed_data_weekly_24"
    output_base = "processed_data_fs"
    weeks_to_create = list(range(1, 25))
    
    print("=" * 50)
    print("Forward Selection Step 1: Single Week Datasets")
    print("=" * 50)
    print(f"Input: {input_dir}")
    print(f"Weeks: {weeks_to_create}")
    print("=" * 50 + "\n")
    
    args = [(w, input_dir, output_base) for w in weeks_to_create]
    
    with Pool(processes=8) as pool:
        pool.map(create_single_week_dataset, args)
    
    print("=" * 50)
    print("🎉 All done!")
    print("=" * 50)


if __name__ == "__main__":
    main()