import numpy as np
import os
import shutil
import sys

# ✅ modify only here!
SELECTED_WEEKS = [18, 5, 12, 8, 7, 11, 17, 16, 9]  # soybean
MODALITIES = ['S2L2A', 'S1GRD', 'WEATHER']  # soybean

selected_str = "_".join(map(str, SELECTED_WEEKS))
BASE_DIR = f"processed_data_fs_{selected_str}"
REMAINING = [w for w in range(1, 25) if w not in SELECTED_WEEKS]

if __name__ == '__main__':
    week = int(sys.argv[1])
    if week not in REMAINING:
        print(f"Skipping week {week}")
        exit(0)

    combo_name = f"processed_data_fs_{selected_str}_{week}"
    print(f"Creating {combo_name}...")
    os.makedirs(combo_name, exist_ok=True)

    for mod in MODALITIES:
        os.makedirs(f"{combo_name}/{mod}", exist_ok=True)
        files = os.listdir(f"{BASE_DIR}/{mod}")
        for fname in files:
            data_base = np.load(f"{BASE_DIR}/{mod}/{fname}")
            data_new = np.load(f"processed_data_fs_{week}/{mod}/{fname}")
            merged = np.concatenate([data_base, data_new], axis=0)
            np.save(f"{combo_name}/{mod}/{fname}", merged)

    for txt in ['train_soybean.txt', 'val_soybean.txt', 'test_soybean.txt']:
        shutil.copy(f"{BASE_DIR}/{txt}", f"{combo_name}/{txt}")

    if os.path.exists(f"{BASE_DIR}/yield_geotiffs"):
        if os.path.exists(f"{combo_name}/yield_geotiffs"):
            shutil.rmtree(f"{combo_name}/yield_geotiffs")
        shutil.copytree(f"{BASE_DIR}/yield_geotiffs", f"{combo_name}/yield_geotiffs")

    print(f"✅ {combo_name} Complete!")