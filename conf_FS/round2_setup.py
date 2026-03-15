import numpy as np
import os
import shutil
from multiprocessing import Pool

BEST_WEEKS = [16]
MODALITIES = ['S2L2A', 'S1GRD', 'DEM', 'WEATHER', 'CDL', 'SOIL']
# REMAINING = [w for w in range(1, 25) if w not in BEST_WEEKS]
REMAINING = [20]  # 16+20만 생성

def process_week(week):
    combo_name = f"processed_data_fs_16_{week}"
    print(f"Creating {combo_name}...")
    
    os.makedirs(combo_name, exist_ok=True)
    
    for mod in MODALITIES:
        os.makedirs(f"{combo_name}/{mod}", exist_ok=True)
        
        files = os.listdir(f"processed_data_fs_16/{mod}")
        
        for fname in files:
            data16 = np.load(f"processed_data_fs_16/{mod}/{fname}")
            data_w = np.load(f"processed_data_fs_{week}/{mod}/{fname}")
            merged = np.concatenate([data16, data_w], axis=0)
            np.save(f"{combo_name}/{mod}/{fname}", merged)
    
    for txt in ['train_corn.txt', 'val_corn.txt', 'test_corn.txt']:
        shutil.copy(f"processed_data_fs_16/{txt}", f"{combo_name}/{txt}")
    
    if not os.path.exists(f"{combo_name}/yield_geotiffs"):
        shutil.copytree(f"processed_data_fs_16/yield_geotiffs", 
                        f"{combo_name}/yield_geotiffs")
    
    print(f"✅ Week16+{week} 완료!")

if __name__ == '__main__':
    with Pool(processes=8) as pool:  # 8개 병렬
        pool.map(process_week, REMAINING)
    print("🎉 Round 2 데이터 준비 완료!")