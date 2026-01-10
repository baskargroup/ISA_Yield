import numpy as np
from pathlib import Path

modalities = ['DEM', 'S1RTC', 'S2L2A', 'SOIL', 'WEATHER', 'yield_geotiffs']

print("Checking shapes\n")

for mod in modalities:
    mod_path = Path(mod)
    npy_files = list(mod_path.glob('*.npy'))
    
    if npy_files:
        sample = np.load(npy_files[0])
        print(f"{mod}:")
        print(f"  Files: {len(npy_files)}")
        print(f"  Shape: {sample.shape}")
        print(f"  Dtype: {sample.dtype}")
        print()