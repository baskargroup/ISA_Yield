import numpy as np
import os
import shutil
import sys
import functools
from multiprocessing import Pool

START_WEEKS = list(range(1, 17))
# ✅ modify only here!
REMOVED_WEEKS = [9, 11, 5, 4, 14, 16, 6, 3, 13, 10, 12]  # Round 1: [], Round 2: [removed week], ...
MODALITIES = ['S2L2A', 'S1GRD', 'CDL', 'DEM', 'WEATHER']  # corn + soybean 공용
FS_BASE = "/work/mech-ai-scratch/bgekim/project/ISA_Yield_Anirudha/ISA_Yield/conf_FS"
WEEKLY_BASE = f"{FS_BASE}/processed_data_fs"

start_str = str(max(START_WEEKS))
removed_str = "_".join(map(str, REMOVED_WEEKS))
round_num = len(REMOVED_WEEKS) + 1
CURRENT_WEEKS = [w for w in START_WEEKS if w not in REMOVED_WEEKS]


def process_file(fname, mod, remaining, combo_name):
    arrays = []
    for w in remaining:
        fpath = f"{WEEKLY_BASE}_{w}/{mod}/{fname}"
        if os.path.exists(fpath):
            arrays.append(np.load(fpath))
    if arrays:
        merged = np.concatenate(arrays, axis=0)
        np.save(f"{combo_name}/{mod}/{fname}", merged)


if __name__ == '__main__':
    remove_week = int(sys.argv[1])
    remaining = [w for w in CURRENT_WEEKS if w != remove_week]

    if REMOVED_WEEKS:
        combo_name = f"processed_data_bs{start_str}_rm_{removed_str}_{remove_week}"
    else:
        combo_name = f"processed_data_bs{start_str}_rm_{remove_week}"

    print(f"[Week {remove_week}] Creating {combo_name}...")
    os.makedirs(combo_name, exist_ok=True)

    ref_dir = f"{WEEKLY_BASE}_{remaining[0]}"

    for mod in MODALITIES:
        os.makedirs(f"{combo_name}/{mod}", exist_ok=True)
        files = os.listdir(f"{ref_dir}/{mod}")
        print(f"  [{mod}] Processing {len(files)} files with 8 workers...")
        fn = functools.partial(process_file, mod=mod, remaining=remaining, combo_name=combo_name)
        with Pool(processes=8) as pool:
            pool.map(fn, files)
        print(f"  [{mod}] ✅ Done")

    # Copy txt files - both corn and soybean
    ref_txt_dir = f"{WEEKLY_BASE}_{remaining[0]}"
    for txt in ['train_soybean.txt', 'val_soybean.txt', 'test_soybean.txt',
                'train_corn.txt', 'val_corn.txt', 'test_corn.txt']:
        shutil.copy(f"{ref_txt_dir}/{txt}", f"{combo_name}/{txt}")

    # Copy yield geotiffs
    ref_yield_dir = f"{WEEKLY_BASE}_{remaining[0]}/yield_geotiffs"
    if os.path.exists(ref_yield_dir):
        if os.path.exists(f"{combo_name}/yield_geotiffs"):
            shutil.rmtree(f"{combo_name}/yield_geotiffs")
        shutil.copytree(ref_yield_dir, f"{combo_name}/yield_geotiffs")

    print(f"[Week {remove_week}] ✅ Done!")