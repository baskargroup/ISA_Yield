"""
Forward Selection Step 1: Generate Single Week dataset
- Generate folder for only Week 1, Week 2, ... Week 24
"""

import numpy as np
import os
import shutil
from tqdm import tqdm

def create_single_week_dataset(week, input_dir, output_base_dir):
    """
    Extract specific week data and create a new folder.
    
    Args:
        week: 1-24 (1-indexed)
        input_dir: processed_data_weekly_24 path
        output_base_dir: output folder base (ex: processed_data_fs)
    """
    output_dir = f"{output_base_dir}_{week}"
    os.makedirs(output_dir, exist_ok=True)
    
    # modalities list
    modalities = ['S2L2A', 'S1GRD', 'DEM', 'WEATHER', 'CDL', 'SOIL']
    
    for modality in modalities:
        src_folder = os.path.join(input_dir, modality)
        if not os.path.exists(src_folder):
            print(f"  {modality}: not found, skipping")
            continue
        
        dst_folder = os.path.join(output_dir, modality)
        os.makedirs(dst_folder, exist_ok=True)
        
        files = [f for f in os.listdir(src_folder) if f.endswith('.npy')]
        
        for filename in tqdm(files, desc=f"  {modality}", leave=False):
            data = np.load(os.path.join(src_folder, filename))
            idx = week - 1
            selected_data = data[idx:idx+1, ...]  # Keep shape (1, C, H, W) 
            np.save(os.path.join(dst_folder, filename), selected_data)
        
        print(f"  {modality}: {len(files)} files")
    
    # copy txt files
    for txt_file in os.listdir(input_dir):
        if txt_file.endswith('.txt'):
            shutil.copy(
                os.path.join(input_dir, txt_file),
                os.path.join(output_dir, txt_file)
            )
    
    print(f"✅ Created: {output_dir}\n")
    return output_dir


def main():
    # ============ setting ============
    input_dir = "processed_data/weekly_24/processed_data_weekly_24"  # input folder
    output_base = "processed_data_fs"       # ourput folder prefix
    
    # Generate Week 1-24 separate datasets
    weeks_to_create = list(range(1, 25))
    # ==============================
    
    print("=" * 50)
    print("Forward Selection Step 1: Single Week Datasets")
    print("=" * 50)
    print(f"Input: {input_dir}")
    print(f"Output: {output_base}_1 ~ {output_base}_24")
    print(f"Weeks: {weeks_to_create}")
    print("=" * 50 + "\n")
    
    for week in weeks_to_create:
        print(f"Creating Week {week}...")
        create_single_week_dataset(week, input_dir, output_base)
    
    print("=" * 50)
    print("✅ All done!")
    print("=" * 50)


if __name__ == "__main__":
    main()