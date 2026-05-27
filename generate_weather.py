import rasterio
import numpy as np
import glob
import os

def process_weather_field(weather_4km_file, template_file, output_file):
    """Convert 4km weather to 10m field-level weather."""
    
    # 1. Load template file (DEM) for shape/transform
    with rasterio.open(template_file) as template:
        out_shape = template.shape        # (224, 224) 등
        out_transform = template.transform
        out_crs = template.crs
    
    # 2. Load weather data
    with rasterio.open(weather_4km_file) as weather_src:
        weather_data = weather_src.read()
        num_bands = weather_src.count
    
    # 3. Calculate mean and broadcast to 10m grid
    out_data = np.zeros((num_bands, out_shape[0], out_shape[1]), dtype=np.float32)
    for b in range(num_bands):
        mean_val = np.nanmean(weather_data[b])
        out_data[b, :, :] = mean_val
    
    # 4. Save
    out_meta = {
        'driver': 'GTiff',
        'height': out_shape[0],
        'width': out_shape[1],
        'count': num_bands,
        'dtype': np.float32,
        'crs': out_crs,
        'transform': out_transform
    }
    
    with rasterio.open(output_file, 'w', **out_meta) as dst:
        dst.write(out_data)
    
    print(f"✅ Saved: {os.path.basename(output_file)}")


# ========== Batch Processing ==========
weather_4km_dir = '/work/mech-ai-scratch/bgekim/satellite/soybean_association/Weather-Gridmet-2017-2024'
template_dir = '/work/mech-ai-scratch/bgekim/project/ISA_Yield_Anirudha/ISA_Yield/chloe_dataset/DEM'
output_dir = '/work/mech-ai-scratch/bgekim/satellite/soybean_association/Weather-Gridmet-10m'

os.makedirs(output_dir, exist_ok=True)

# Get all template (DEM) files
template_files = glob.glob(os.path.join(template_dir, 'ST*IA*_*.tif'))

for template_file in template_files:
    base_name = os.path.basename(template_file)
    field_id = base_name.split('_')[0]  # ST2021IA0013
    year = field_id[2:6]  # 2021
    
    print(f"\nProcessing: {field_id}")
    
    # Find all weather files for this field
    weather_pattern = f'{field_id}_{year}-*_Gridmet.tif'
    weather_files = glob.glob(os.path.join(weather_4km_dir, weather_pattern))
    
    if not weather_files:
        print(f"  ⚠️ No weather files found")
        continue
    
    for weather_file in weather_files:
        weather_base = os.path.basename(weather_file)
        output_file = os.path.join(output_dir, weather_base)
        
        if os.path.exists(output_file):
            print(f"  Skip: {weather_base} (exists)")
            continue
        
        try:
            process_weather_field(weather_file, template_file, output_file)
        except Exception as e:
            print(f"  ❌ Error: {weather_base} - {e}")

print("\n🎉 Complete!")