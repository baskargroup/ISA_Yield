import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box
import numpy as np
import glob
import os

from rasterio.features import rasterize

def process_weather_field(weather_1km_file, yield_file, output_file):
    """Convert 1km weather to 10m field-level weather."""
    
    # 1. Load yield file
    with rasterio.open(yield_file) as yield_src:
        yield_bounds = yield_src.bounds
        yield_transform = yield_src.transform
        yield_shape = yield_src.shape
        yield_crs = yield_src.crs
        
        field_bbox = box(yield_bounds.left, yield_bounds.bottom, 
                        yield_bounds.right, yield_bounds.top)
    
    # 2. Load weather data
    with rasterio.open(weather_1km_file) as weather_src:
        weather_data = weather_src.read()
        weather_crs = weather_src.crs
        
        # CRS 변환
        if yield_crs != weather_crs:
            import geopandas as gpd
            gdf = gpd.GeoDataFrame({'geometry': [field_bbox]}, crs=yield_crs)
            gdf_reprojected = gdf.to_crs(weather_crs)
            field_bbox_weather_crs = gdf_reprojected.geometry.iloc[0]
        else:
            field_bbox_weather_crs = field_bbox
        
        # ✅ all_touched=True로 mask 생성!
        field_mask_1km = rasterize(
            [(field_bbox_weather_crs, 1)],
            out_shape=(weather_src.height, weather_src.width),
            transform=weather_src.transform,
            fill=0,
            all_touched=True,
            dtype='uint8'
        ).astype(bool)
        
        print(f"  Mask overlap: {field_mask_1km.sum()} pixels")
        
        if field_mask_1km.sum() == 0:
            print(f"  ⚠️ No overlap! Skipping")
            return
        
        # Calculate mean
        mean_values = []
        for band_idx in range(weather_data.shape[0]):
            band_data = weather_data[band_idx]
            field_values = band_data[field_mask_1km]
            mean_val = np.nanmean(field_values)
            mean_values.append(mean_val)
        
        mean_values = np.array(mean_values)
    
    # 3. Create output
    num_bands = len(mean_values)
    output_data = np.zeros((num_bands, yield_shape[0], yield_shape[1]), dtype=np.float32)
    
    for band_idx in range(num_bands):
        output_data[band_idx, :, :] = mean_values[band_idx]
    
    # 4. Save
    out_meta = {
        'driver': 'GTiff',
        'height': yield_shape[0],
        'width': yield_shape[1],
        'count': num_bands,
        'dtype': np.float32,
        'crs': yield_crs,
        'transform': yield_transform
    }
    
    with rasterio.open(output_file, 'w', **out_meta) as dst:
        dst.write(output_data)
    
    print(f"✅ Saved: {os.path.basename(output_file)}")


# ========== Batch Processing ==========

weather_1km_dir = '/work/mech-ai-scratch/geospatial-data/iowa-soybean-association/Weather_1km'
yield_dir = '/work/mech-ai-scratch/bgekim/project/ISA_Yield_Anirudha/ISA_Yield/yld_proc/unprocessed_data/yield_geotiffs'
output_dir = '/work/mech-ai-scratch/geospatial-data/iowa-soybean-association/Weather_processed_10m'

os.makedirs(output_dir, exist_ok=True)

# Get all yield files
yield_files = glob.glob(os.path.join(yield_dir, 'ST*IA*_*.tif'))

for yield_file in yield_files:
    base_name = os.path.basename(yield_file)
    field_id = base_name.split('_')[0]  # ST2021IA0013
    year = field_id[2:6]  # 2021
    
    print(f"\nProcessing: {field_id}")
    
    # Find all weather files for this field
    weather_pattern = f'{field_id}_{year}-*_Daymet.tif'
    weather_files = glob.glob(os.path.join(weather_1km_dir, weather_pattern))
    
    for weather_file in weather_files:
        weather_base = os.path.basename(weather_file)
        output_file = os.path.join(output_dir, weather_base)
        
        if os.path.exists(output_file):
            print(f"  Skip: {weather_base} (exists)")
            continue
        
        try:
            process_weather_field(weather_file, yield_file, output_file)
        except Exception as e:
            print(f"  ❌ Error: {weather_base} - {e}")

print("\n🎉 Complete!")