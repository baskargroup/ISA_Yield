import numpy as np
import os
import glob


selected_weeks = [2, 3, 6, 7, 8, 11, 13, 14, 15, 17, 18, 19, 21, 23] # for 14
# selected_weeks = [3, 8, 13, 14, 15, 17, 18, 19, 21, 23]  # for 10
# selected_weeks = [3, 8, 13, 15, 17, 18, 21, 23]
# selected_weeks = [8, 13, 15, 17, 18, 23]

src = 'processed_data_weekly_24'
dst = 'processed_data_weekly_selected_14'

for modality in os.listdir(src):
    src_path = f'{src}/{modality}'
    dst_path = f'{dst}/{modality}'
    
    if not os.path.isdir(src_path): 
        continue
        
    os.makedirs(dst_path, exist_ok=True)
    
    for f in glob.glob(f'{src_path}/*.npy'):
        data = np.load(f)
        
        if data.ndim == 4 and data.shape[0] == 24:  # (T=24, C, H, W) 
            data = data[selected_weeks]  # → (T=10, C, H, W)
        
        
        np.save(f'{dst_path}/{os.path.basename(f)}', data)
        
    print(f'{modality}: {len(glob.glob(f"{dst_path}/*.npy"))} files saved')

print(f'\nDone! T=24 → T={len(selected_weeks)}')