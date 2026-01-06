import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box
import numpy as np

weather_file = '/work/mech-ai-scratch/geospatial-data/iowa-soybean-association/Weather_10m/ST2021IASW036_2021-06-24_Daymet.tif'  # 1km 파일
yield_file = '/work/mech-ai-scratch/bgekim/project/ISA_Yield/yld_proc/unprocessed_data/yield_geotiffs/ST2021IASW036_Soybean.tif'

print("=== 1. Check CRS ===")
with rasterio.open(weather_file) as src:
    print(f"Weather CRS: {src.crs}")
    print(f"Weather Bounds: {src.bounds}")
    print(f"Weather Shape: {src.shape}")
    weather_data = src.read()
    print(f"Weather Data Range: {np.nanmin(weather_data)} ~ {np.nanmax(weather_data)}")
    print(f"Weather has NaN? {np.isnan(weather_data).any()}")

with rasterio.open(yield_file) as src:
    print(f"\nYield CRS: {src.crs}")
    print(f"Yield Bounds: {src.bounds}")
    print(f"Yield Shape: {src.shape}")

print("\n=== 2. Check Overlap ===")
with rasterio.open(weather_file) as weather_src:
    with rasterio.open(yield_file) as yield_src:
        # Yield boundary
        yield_bounds = yield_src.bounds
        field_bbox = box(yield_bounds.left, yield_bounds.bottom, 
                        yield_bounds.right, yield_bounds.top)
        
        print(f"Field bbox: {field_bbox.bounds}")
        
        # Create Mask 
        field_mask_1km = geometry_mask(
            [field_bbox],
            out_shape=(weather_src.height, weather_src.width),
            transform=weather_src.transform,
            invert=True
        )
        
        print(f"Mask has True? {field_mask_1km.any()}")
        print(f"Mask True count: {field_mask_1km.sum()}")
        
        # Extract Value
        weather_data = weather_src.read(1)
        field_values = weather_data[field_mask_1km]
        print(f"Field values: {field_values}")
        print(f"Field values mean: {np.nanmean(field_values)}")