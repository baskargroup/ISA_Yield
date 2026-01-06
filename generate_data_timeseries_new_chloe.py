import os
import rasterio
from rasterio.warp import reproject, Resampling
import glob
import numpy as np
import gc


# ========== Define paths ==========
ext = "IA"
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Base directories 
dem_dir = '/work/mech-ai-scratch/geospatial-data/iowa-soybean-association/Elevation_10m'  
soil_dir = '/work/mech-ai-scratch/geospatial-data/iowa-soybean-association/Soil_10m'  
s2_dir = '/work/mech-ai-scratch/geospatial-data/iowa-soybean-association/S2_10m'
s1_dir = '/work/mech-ai-scratch/geospatial-data/iowa-soybean-association/S1_10m'
weather_dir = '/work/mech-ai-scratch/geospatial-data/iowa-soybean-association/Weather_processed_10m'  
yield_path = '/work/mech-ai-scratch/bgekim/project/ISA_Yield/yld_proc/unprocessed_data/yield_geotiffs'

output_dir = '/work/mech-ai-scratch/bgekim/project/ISA_Yield/chloe_dataset'

# Modalities
modalities = ['DEM', 'SOIL', 'S2L2A', 'S1RTC', 'WEATHER']

# Ensure output directories exist
for m in modalities:
    os.makedirs(os.path.join(output_dir, m), exist_ok=True)

# Get yield files
yield_files = []
for year in years:
    yield_files.extend(glob.glob(os.path.join(yield_path, f'ST{year}IA*_*.tif')))


# ========== Functions ==========

def resample_to_reference(src_data, src_transform, src_crs, ref_transform, ref_crs, ref_shape):
    """
    Resample source data to match reference grid (CRS, transform, shape).
    
    Args:
        src_data: Source array (bands, H, W)
        src_transform: Source affine transform
        src_crs: Source CRS
        ref_transform: Reference affine transform
        ref_crs: Reference CRS
        ref_shape: Reference shape (height, width)
    
    Returns:
        Resampled array (bands, ref_height, ref_width)
    """
    try:
        num_bands = src_data.shape[0]
        ref_height, ref_width = ref_shape
        
        # Create output array
        dst_data = np.zeros((num_bands, ref_height, ref_width), dtype=src_data.dtype)
        
        # Reproject each band
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )
        
        return dst_data
        
    except Exception as e:
        print(f"    ❌ Resampling error: {e}")
        return None

def load_static_data(field_id, data_dir, pattern_suffix, ref_transform, ref_crs, ref_shape, repeat_times=24):
    """
    Load static data, align to reference grid, and repeat for temporal dimension.
    
    Returns:
        numpy array (repeat_times, bands, H, W)
    """
    try:
        file_path = os.path.join(data_dir, f'{field_id}{pattern_suffix}')
        
        if not os.path.exists(file_path):
            print(f"  ⚠️ File not found: {file_path}")
            return None
        
        with rasterio.open(file_path) as src:
            src_data = src.read()  # (bands, H, W)
            src_transform = src.transform
            src_crs = src.crs
        
        # Resample to reference grid
        aligned_data = resample_to_reference(
            src_data, src_transform, src_crs,
            ref_transform, ref_crs, ref_shape
        )
        
        if aligned_data is None:
            return None
        
        # ✅ Repeat for temporal dimension
        repeated = np.stack([aligned_data for _ in range(repeat_times)], axis=0)
        # Shape: (repeat_times, bands, H, W)
        
        return repeated
        
    except Exception as e:
        print(f"  ❌ Error loading static data: {e}")
        return None


def load_dynamic_data(field_id, year, data_dir, pattern_middle, pattern_suffix, ref_transform, ref_crs, ref_shape):
    """
    Load dynamic (time-series) data and align to reference grid.
    
    Args:
        field_id: e.g., 'ST2021IA0013'
        year: e.g., '2021'
        data_dir: Directory containing files
        pattern_middle: e.g., '_*_S2' or '_*_Daymet' or just '_*'
        pattern_suffix: e.g., '.tif'
        ref_transform: Reference affine transform
        ref_crs: Reference CRS
        ref_shape: Reference shape (height, width)
    
    Returns:
        tuple: (numpy array (dates, bands, H, W), list of date strings)
    """
    try:
        # Find files for this field
        pattern = f'{field_id}{pattern_middle}{pattern_suffix}'
        files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        
        if not files:
            print(f"  ⚠️ No files found with pattern: {pattern}")
            return None, None
        
        # Filter by growing season (April-September)
        valid_files = []
        valid_dates = []
        
        for f in files:
            filename = os.path.basename(f)
            # Extract date from filename
            # ST2021IA0013_2021-04-15_S2.tif → 2021-04-15
            parts = filename.split('_')
            if len(parts) >= 2:
                date_str = parts[1]
                try:
                    # Check year matches
                    file_year = date_str.split('-')[0]
                    if file_year != year:
                        continue
                    
                    # Check month (April-September)
                    month = int(date_str.split('-')[1])
                    if 4 <= month <= 9:
                        valid_files.append(f)
                        valid_dates.append(date_str)
                except:
                    continue
        
        if not valid_files:
            print(f"  ⚠️ No valid dates found (April-Sept {year})")
            return None, None
        
        print(f"    Found {len(valid_files)} dates")
        
        # Load all files and align to reference grid
        arrays = []
        for file_path in sorted(valid_files):
            with rasterio.open(file_path) as src:
                src_data = src.read()  # (bands, H, W)
                src_transform = src.transform
                src_crs = src.crs
            
            # Resample to reference grid
            aligned_data = resample_to_reference(
                src_data, src_transform, src_crs,
                ref_transform, ref_crs, ref_shape
            )
            
            if aligned_data is not None:
                arrays.append(aligned_data)
        
        if not arrays:
            return None, None
        
        # Stack: (dates, bands, H, W)
        stacked = np.stack(arrays, axis=0)
        
        return stacked, sorted(valid_dates)
        
    except Exception as e:
        print(f"  ❌ Error loading dynamic data: {e}")
        return None, None

def load_dynamic_data_with_dates(field_id, data_dir, selected_dates, pattern_suffix, ref_transform, ref_crs, ref_shape):
    """
    Load dynamic data for specific dates only and align to reference grid.
    
    Args:
        field_id: e.g., 'ST2021IA0013'
        data_dir: Directory containing files
        selected_dates: List of dates to load (e.g., ['2021-04-15', '2021-05-01', ...])
        pattern_suffix: e.g., '_S2.tif' or '.tif' or '_Daymet.tif'
        ref_transform: Reference affine transform
        ref_crs: Reference CRS
        ref_shape: Reference shape (height, width)
    
    Returns:
        numpy array (dates, bands, H, W)
    """
    try:
        arrays = []
        loaded_dates = []
        
        for date in selected_dates:
            # Construct filename
            if pattern_suffix == '.tif':
                # S1: ST2021IA0013_2021-04-15.tif
                filename = f'{field_id}_{date}.tif'
            else:
                # S2: ST2021IA0013_2021-04-15_S2.tif
                # Weather: ST2021IA0013_2021-04-15_Daymet.tif
                filename = f'{field_id}_{date}{pattern_suffix}'
            
            file_path = os.path.join(data_dir, filename)
            
            if not os.path.exists(file_path):
                print(f"    ⚠️ Missing: {filename}")
                continue
            
            with rasterio.open(file_path) as src:
                src_data = src.read()  # (bands, H, W)
                src_transform = src.transform
                src_crs = src.crs
            
            # Resample to reference grid
            aligned_data = resample_to_reference(
                src_data, src_transform, src_crs,
                ref_transform, ref_crs, ref_shape
            )
            
            if aligned_data is not None:
                arrays.append(aligned_data)
                loaded_dates.append(date)
        
        if not arrays:
            return None
        
        # Stack: (dates, bands, H, W)
        stacked = np.stack(arrays, axis=0)
        
        return stacked
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None

# ========== Process ==========

log_file = os.path.join(output_dir, "processing_log.txt")
dates_log_file = os.path.join(output_dir, "dates_log.txt")

with open(log_file, "w") as logf, open(dates_log_file, "w") as datesf:
    
    for yield_file in yield_files:
        base_name = os.path.basename(yield_file)
        field_id = base_name.split('_')[0]
        year = field_id[2:6]
        
        print(f"\n{'='*70}")
        print(f"Processing: {base_name}")
        print(f"  Field ID: {field_id}, Year: {year}")
        
        # ========== Step 1: Load yield geotiff as reference grid ==========
        
        try:
            with rasterio.open(yield_file) as ref_src:
                ref_transform = ref_src.transform
                ref_crs = ref_src.crs
                ref_shape = (ref_src.height, ref_src.width)
                print(f"\n  Reference Grid (from yield):")
                print(f"    Shape: {ref_shape}")
                print(f"    CRS: {ref_crs}")
                print(f"    Transform: {ref_transform}")
        except Exception as e:
            print(f"  ❌ Error reading yield file: {e}")
            continue
        
        # ========== Step 2: Identify dates for each dynamic modality (NO INTERSECTION) ==========
        
        dynamic_dates = {}
        
        # S2L2A dates
        s2_pattern = f'{field_id}_{year}-*_S2.tif'
        s2_files = glob.glob(os.path.join(s2_dir, s2_pattern))
        s2_dates = []
        for f in s2_files:
            filename = os.path.basename(f)
            date_str = filename.split('_')[1]
            try:
                month = int(date_str.split('-')[1])
                if 4 <= month <= 9:
                    s2_dates.append(date_str)
            except:
                continue
        dynamic_dates['S2L2A'] = sorted(s2_dates)
        
        # S1RTC dates
        s1_pattern = f'{field_id}_{year}-*.tif'
        s1_files = glob.glob(os.path.join(s1_dir, s1_pattern))
        s1_dates = []
        for f in s1_files:
            filename = os.path.basename(f)
            date_str = filename.split('_')[1].replace('.tif', '')
            try:
                month = int(date_str.split('-')[1])
                if 4 <= month <= 9:
                    s1_dates.append(date_str)
            except:
                continue
        dynamic_dates['S1RTC'] = sorted(s1_dates)
        
        # WEATHER dates
        weather_pattern = f'{field_id}_{year}-*_Daymet.tif'
        weather_files = glob.glob(os.path.join(weather_dir, weather_pattern))
        weather_dates = []
        for f in weather_files:
            filename = os.path.basename(f)
            date_str = filename.split('_')[1]
            try:
                month = int(date_str.split('-')[1])
                if 4 <= month <= 9:
                    weather_dates.append(date_str)
            except:
                continue
        dynamic_dates['WEATHER'] = sorted(weather_dates)
        
        print(f"\n  Date counts (ALL valid dates per modality):")
        for mod, dates in dynamic_dates.items():
            print(f"    {mod}: {len(dates)} dates")
        
        # Log dates for this field
        datesf.write(f"\n{'='*70}\n")
        datesf.write(f"{base_name} (Field: {field_id}, Year: {year})\n")
        for mod, dates in dynamic_dates.items():
            datesf.write(f"  {mod}: {len(dates)} dates\n")
            datesf.write(f"    {', '.join(dates)}\n")
        
        if all(len(dates) == 0 for dates in dynamic_dates.values()):
            print(f"  ⚠️ No valid dates found, skipping {base_name}")
            continue
        
        # ========== Step 3: Static modalities (aligned to yield grid) ==========
        
        # Calculate repeat_times based on longest dynamic modality
        max_dates = max(len(dates) for dates in dynamic_dates.values() if len(dates) > 0)
        repeat_times = max_dates if max_dates > 0 else 1
        
        print(f"\n  [DEM] (repeating {repeat_times} times to match dynamic data)")
        dem_data = load_static_data(field_id, dem_dir, '_DEM_10m.tif', ref_transform, ref_crs, ref_shape, repeat_times=repeat_times)
        if dem_data is not None:
            output_path = os.path.join(output_dir, 'DEM', base_name.replace('.tif', '.npy'))
            np.save(output_path, dem_data, allow_pickle=False)
            print(f"    ✅ Saved: {dem_data.shape}")
        
        print(f"\n  [SOIL] (repeating {repeat_times} times to match dynamic data)")
        soil_data = load_static_data(field_id, soil_dir, '_Static_Soil_gNATSGO.tif', ref_transform, ref_crs, ref_shape, repeat_times=repeat_times)
        if soil_data is not None:
            output_path = os.path.join(output_dir, 'SOIL', base_name.replace('.tif', '.npy'))
            np.save(output_path, soil_data, allow_pickle=False)
            print(f"    ✅ Saved: {soil_data.shape}")
        
        # ========== Step 4: Dynamic modalities (ALL dates, aligned to yield grid) ==========
        
        print(f"\n  [S2L2A] (loading ALL {len(dynamic_dates['S2L2A'])} dates)")
        s2_data = load_dynamic_data_with_dates(field_id, s2_dir, dynamic_dates['S2L2A'], '_S2.tif', ref_transform, ref_crs, ref_shape)
        if s2_data is not None:
            output_path = os.path.join(output_dir, 'S2L2A', base_name.replace('.tif', '.npy'))
            np.save(output_path, s2_data, allow_pickle=False)
            print(f"    ✅ Saved: {s2_data.shape}")
        
        print(f"\n  [S1RTC] (loading ALL {len(dynamic_dates['S1RTC'])} dates)")
        s1_data = load_dynamic_data_with_dates(field_id, s1_dir, dynamic_dates['S1RTC'], '.tif', ref_transform, ref_crs, ref_shape)
        if s1_data is not None:
            output_path = os.path.join(output_dir, 'S1RTC', base_name.replace('.tif', '.npy'))
            np.save(output_path, s1_data, allow_pickle=False)
            print(f"    ✅ Saved: {s1_data.shape}")
        
        print(f"\n  [WEATHER] (loading ALL {len(dynamic_dates['WEATHER'])} dates)")
        weather_data = load_dynamic_data_with_dates(field_id, weather_dir, dynamic_dates['WEATHER'], '_Daymet.tif', ref_transform, ref_crs, ref_shape)
        if weather_data is not None:
            output_path = os.path.join(output_dir, 'WEATHER', base_name.replace('.tif', '.npy'))
            np.save(output_path, weather_data, allow_pickle=False)
            print(f"    ✅ Saved: {weather_data.shape}")
        
        # Log
        logf.write(f"{base_name}: S2L2A={len(dynamic_dates['S2L2A'])}, S1RTC={len(dynamic_dates['S1RTC'])}, WEATHER={len(dynamic_dates['WEATHER'])}\n")
        
        gc.collect()

print("\n🎉 Processing complete!")
print("✅ All modalities saved with their own valid dates (no intersection)")
print("✅ All data aligned to yield geotiff reference grids (CRS, transform, shape)")
print(f"📄 Processing log: {log_file}")
print(f"📄 Dates log: {dates_log_file}")
