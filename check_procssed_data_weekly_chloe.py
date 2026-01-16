# import numpy as np
# from pathlib import Path

# modalities = ['DEM', 'S1GRD', 'S2L2A', 'SOIL', 'WEATHER', 'yield_geotiffs']

# print("Checking shapes\n")

# for mod in modalities:
#     mod_path = Path(mod)
#     npy_files = list(mod_path.glob('*.npy'))
    
#     if npy_files:
#         sample = np.load(npy_files[0])
#         print(f"{mod}:")
#         print(f"  Files: {len(npy_files)}")
#         print(f"  Shape: {sample.shape}")
#         print(f"  Dtype: {sample.dtype}")
#         print()

import numpy as np
from pathlib import Path
from collections import Counter

modalities = ["DEM", "S1GRD", "S1RTC", "S2L2A", "SOIL", "WEATHER"]  # 둘 다 넣어두자

def scan(mod):
    p = Path(mod)
    files = sorted(p.glob("*.npy"))
    if not files:
        print(f"{mod}: (no npy files)")
        return

    t_counter = Counter()
    shape_counter = Counter()
    bad = []

    for f in files:
        arr = np.load(f, mmap_mode="r")
        shape = tuple(arr.shape)
        shape_counter[shape] += 1

        # 너 저장 포맷이 (T,C,H,W) 라고 가정
        if len(shape) == 4:
            T = shape[0]
        elif len(shape) == 3:
            T = 1
        else:
            T = None

        t_counter[T] += 1
        if T != 24:  # 24 아닌 것만 추적
            bad.append((f.name, shape))

    print(f"\n{mod}: files={len(files)}")
    print("  T distribution:", dict(t_counter))
    print("  top shapes:", shape_counter.most_common(5))
    if bad:
        print(f"  !!! non-24 files: {len(bad)}")
        for name, shape in bad[:20]:
            print("   ", name, shape)

for m in modalities:
    scan(m)
