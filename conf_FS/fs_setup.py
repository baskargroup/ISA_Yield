import numpy as np
import os
import shutil
from multiprocessing import Pool
 
# ✅ modify only here!
SELECTED_WEEKS = [16, 18]

MODALITIES = ['S2L2A', 'S1GRD', 'CDL', 'DEM', 'WEATHER', 'SOIL']
selected_str = "_".join(map(str, SELECTED_WEEKS))
BASE_DIR = f"processed_data_fs_{selected_str}"  # processed_data_fs_16_20_1_
REMAINING = [w for w in range(1, 25) if w not in SELECTED_WEEKS]

def process_week(week):
    combo_name = f"processed_data_fs_{selected_str}_{week}"
    print(f"Creating {combo_name}...")
    
    os.makedirs(combo_name, exist_ok=True)
    
    for mod in MODALITIES:
        os.makedirs(f"{combo_name}/{mod}", exist_ok=True)
        
        files = os.listdir(f"{BASE_DIR}/{mod}")
        
        for fname in files:
            data_base = np.load(f"{BASE_DIR}/{mod}/{fname}")       # (2, 12, 224, 224)
            data_new  = np.load(f"processed_data_fs_{week}/{mod}/{fname}")  # (1, 12, 224, 224)
            merged = np.concatenate([data_base, data_new], axis=0) # (3, 12, 224, 224)
            np.save(f"{combo_name}/{mod}/{fname}", merged)
    
    for txt in ['train_soybean.txt', 'val_soybean.txt', 'test_soybean.txt']:
        shutil.copy(f"{BASE_DIR}/{txt}", f"{combo_name}/{txt}")
    
    if os.path.exists(f"{BASE_DIR}/yield_geotiffs"):
        if os.path.exists(f"{combo_name}/yield_geotiffs"):
            shutil.rmtree(f"{combo_name}/yield_geotiffs")
        shutil.copytree(f"{BASE_DIR}/yield_geotiffs", f"{combo_name}/yield_geotiffs")
    
    print(f"✅ {combo_name} Complete!")

if __name__ == '__main__':
    with Pool(processes=8) as pool:
        pool.map(process_week, REMAINING)
    print("🎉 Round 4 Dataset preparation complete!")