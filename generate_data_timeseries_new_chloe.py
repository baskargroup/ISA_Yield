import os
import rasterio
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

def load_static_data(field_id, data_dir, pattern_suffix, repeat_times=12):
    """
    Load static data and repeat for temporal dimension.
    
    Returns:
        numpy array (repeat_times, bands, H, W)
    """
    try:
        file_path = os.path.join(data_dir, f'{field_id}{pattern_suffix}')
        
        if not os.path.exists(file_path):
            print(f"  ⚠️ File not found: {file_path}")
            return None
        
        with rasterio.open(file_path) as src:
            data = src.read()  # (bands, H, W)
        
        # ✅ Repeat for temporal dimension
        repeated = np.stack([data for _ in range(repeat_times)], axis=0)
        # Shape: (repeat_times, bands, H, W)
        
        return repeated
        
    except Exception as e:
        print(f"  ❌ Error loading static data: {e}")
        return None


def load_dynamic_data(field_id, year, data_dir, pattern_middle, pattern_suffix):
    """
    Load dynamic (time-series) data.
    
    Args:
        field_id: e.g., 'ST2021IA0013'
        year: e.g., '2021'
        data_dir: Directory containing files
        pattern_middle: e.g., '_*_S2' or '_*_Daymet' or just '_*'
        pattern_suffix: e.g., '.tif'
    
    Returns:
        numpy array (dates, bands, H, W)
    """
    try:
        # Find files for this field
        pattern = f'{field_id}{pattern_middle}{pattern_suffix}'
        files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        
        if not files:
            print(f"  ⚠️ No files found with pattern: {pattern}")
            return None
        
        # Filter by growing season (April-September)
        valid_files = []
        
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
                except:
                    continue
        
        if not valid_files:
            print(f"  ⚠️ No valid dates found (April-Sept {year})")
            return None
        
        print(f"    Found {len(valid_files)} dates")
        
        # Load all files
        arrays = []
        for file_path in sorted(valid_files):
            with rasterio.open(file_path) as src:
                data = src.read()  # (bands, H, W)
                arrays.append(data)
        
        # Stack: (dates, bands, H, W)
        stacked = np.stack(arrays, axis=0)
        
        return stacked
        
    except Exception as e:
        print(f"  ❌ Error loading dynamic data: {e}")
        return None

def load_dynamic_data_with_dates(field_id, data_dir, selected_dates, pattern_suffix):
    """
    Load dynamic data for specific dates only.
    
    Args:
        field_id: e.g., 'ST2021IA0013'
        data_dir: Directory containing files
        selected_dates: List of dates to load (e.g., ['2021-04-15', '2021-05-01', ...])
        pattern_suffix: e.g., '_S2.tif' or '.tif' or '_Daymet.tif'
    
    Returns:
        numpy array (dates, bands, H, W)
    """
    try:
        arrays = []
        
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
                data = src.read()  # (bands, H, W)
                arrays.append(data)
        
        if not arrays:
            return None
        
        # Stack: (dates, bands, H, W)
        stacked = np.stack(arrays, axis=0)
        
        return stacked
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None

# ========== Process ==========

num_times = 24  # ← 고정! (bi-weekly 4-9월 = 24 weeks)

log_file = os.path.join(output_dir, "processing_log.txt")
with open(log_file, "w") as logf:
    
    for yield_file in yield_files:
        base_name = os.path.basename(yield_file)
        field_id = base_name.split('_')[0]
        year = field_id[2:6]
        
        print(f"\n{'='*70}")
        print(f"Processing: {base_name}")
        print(f"  Field ID: {field_id}, Year: {year}")
        
        # ========== Step 1: Identify dates for each dynamic modality ==========
        
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
        
        print(f"\n  Date counts:")
        for mod, dates in dynamic_dates.items():
            print(f"    {mod}: {len(dates)} dates")
        
        # ========== Step 2: Find common dates (intersection) ==========
        
        if all(len(dates) == 0 for dates in dynamic_dates.values()):
            print(f"  ⚠️ No valid dates found, skipping {base_name}")
            continue
        
        sets_of_dates = [set(dates) for dates in dynamic_dates.values() if len(dates) > 0]
        common_dates = sorted(set.intersection(*sets_of_dates))
        
        if not common_dates:
            print(f"  ⚠️ No common dates across modalities, skipping {base_name}")
            continue
        
        print(f"\n  ✅ Common dates: {len(common_dates)} dates")
        print(f"    {common_dates}")
        print(f"  ⚠️ Target after imputation: {num_times} dates")
        
        # ========== Step 3: Static modalities (now fixed set of 24) ==========
        
        print(f"\n  [DEM]")
        dem_data = load_static_data(field_id, dem_dir, '_DEM_10m.tif', repeat_times=num_times)
        if dem_data is not None:
            output_path = os.path.join(output_dir, 'DEM', base_name.replace('.tif', '.npy'))
            np.save(output_path, dem_data, allow_pickle=False)
            print(f"    ✅ Saved: {dem_data.shape}")
        
        print(f"\n  [SOIL]")
        soil_data = load_static_data(field_id, soil_dir, '_Static_Soil_gNATSGO.tif', repeat_times=num_times)
        if soil_data is not None:
            output_path = os.path.join(output_dir, 'SOIL', base_name.replace('.tif', '.npy'))
            np.save(output_path, soil_data, allow_pickle=False)
            print(f"    ✅ Saved: {soil_data.shape}")
        
        # ========== Step 4: Dynamic modalities (common dates only, variable-length) ==========
        
        print(f"\n  [S2L2A]")
        s2_data = load_dynamic_data_with_dates(field_id, s2_dir, common_dates, '_S2.tif')
        if s2_data is not None:
            output_path = os.path.join(output_dir, 'S2L2A', base_name.replace('.tif', '.npy'))
            np.save(output_path, s2_data, allow_pickle=False)
            print(f"    ✅ Saved: {s2_data.shape}")
            print(f"    ⚠️ Needs imputation: {s2_data.shape[0]} → {num_times}")
        
        print(f"\n  [S1RTC]")
        s1_data = load_dynamic_data_with_dates(field_id, s1_dir, common_dates, '.tif')
        if s1_data is not None:
            output_path = os.path.join(output_dir, 'S1RTC', base_name.replace('.tif', '.npy'))
            np.save(output_path, s1_data, allow_pickle=False)
            print(f"    ✅ Saved: {s1_data.shape}")
            print(f"    ⚠️ Needs imputation: {s1_data.shape[0]} → {num_times}")
        
        print(f"\n  [WEATHER]")
        weather_data = load_dynamic_data_with_dates(field_id, weather_dir, common_dates, '_Daymet.tif')
        if weather_data is not None:
            output_path = os.path.join(output_dir, 'WEATHER', base_name.replace('.tif', '.npy'))
            np.save(output_path, weather_data, allow_pickle=False)
            print(f"    ✅ Saved: {weather_data.shape}")
            print(f"    ⚠️ Needs imputation: {weather_data.shape[0]} → {num_times}")
        
        # Log
        logf.write(f"{base_name}: {len(common_dates)} dates (target: {num_times}) - {','.join(common_dates)}\n")
        
        gc.collect()

print("\n🎉 Processing complete!")
print("⚠️ Next step: Run imputation script to make all dynamic data 24 time steps")
