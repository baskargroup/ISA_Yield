import numpy as np
import os
import glob

selected_biweeks = [0, 3, 5, 6, 7, 8, 9, 11]  

src = 'processed_data/biweekly_12/processed_data_biweekly_12'
dst = 'processed_data/biweekly_12/processed_data_biweekly_selected'

for modality in os.listdir(src):
    src_path = f'{src}/{modality}'
    dst_path = f'{dst}/{modality}'
    
    if not os.path.isdir(src_path): 
        continue
        
    os.makedirs(dst_path, exist_ok=True)
    
    for f in glob.glob(f'{src_path}/*.npy'):
        data = np.load(f)
        
        if data.ndim == 4 and data.shape[0] == 12:  # (T=12, C, H, W)
            data = data[selected_biweeks]  # → (T=8, C, H, W)
        # yield is just (C, H, W) -> save
        
        np.save(f'{dst_path}/{os.path.basename(f)}', data)
        
    print(f'{modality}: {len(glob.glob(f"{dst_path}/*.npy"))} files saved')

print(f'\nDone! T=12 → T={len(selected_biweeks)}')