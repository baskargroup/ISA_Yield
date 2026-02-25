import geopandas as gpd
import pandas as pd
import glob
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm

# Find all Yield_YYYY_filtered.parquet files
parquet_files = sorted(glob.glob('Yield_*_filtered.parquet'))
print(f"Found {len(parquet_files)} parquet files: {parquet_files}")


def process_layer_group(args):
    """Process a single layer group and extract bounds."""
    layer_id, layer_data, year = args
    
    # Extract coordinates from POINT geometry
    coords = [(point.x, point.y) for point in layer_data.geometry]
    
    if coords:
        # Get min/max coordinates
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Top-left: (min_x, max_y), Bottom-right: (max_x, min_y)
        top_left = (min_x, max_y)
        bottom_right = (max_x, min_y)
        
        return (layer_id, {
            'year': year,
            'top_left': top_left,
            'bottom_right': bottom_right,
            'num_points': len(coords)
        })
    
    return None


def process_file(file):
    """Process a single parquet file and return layer bounds."""
    year = file.split('_')[1]
    
    # Read the parquet file
    gdf = gpd.read_parquet(file)
    
    # Ensure CRS is set
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    
    # Prepare layer data for parallel processing
    layer_tasks = [
        (layer_id, gdf[gdf['Layer_ID'] == layer_id], year)
        for layer_id in gdf['Layer_ID'].unique()
    ]
    
    # Use ThreadPoolExecutor for inner parallelism (avoids daemon process issues)
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_layer_group, task) for task in layer_tasks]
        for future in tqdm(futures, desc=f"  {file}", leave=False):
            result = future.result()
            if result is not None:
                results.append(result)
    
    return results


# Dictionary to store bounding boxes for each Layer_ID across all years
layer_bounds = {}

# Process each file in parallel
print("\nProcessing parquet files...")
num_file_workers = min(cpu_count() - 1, 4)  # Use up to 4 workers for files
with Pool(num_file_workers) as file_pool:
    file_results = list(tqdm(
        file_pool.imap_unordered(process_file, parquet_files),
        total=len(parquet_files),
        desc="Files",
        unit="file"
    ))

# Aggregate results
for file_result in file_results:
    for layer_id, bounds in file_result:
        layer_bounds[layer_id] = bounds


# Convert to DataFrame for better visualization
bounds_df = pd.DataFrame.from_dict(layer_bounds, orient='index')
bounds_df.index.name = 'Layer_ID'
bounds_df = bounds_df.reset_index()

print(f"\n{'='*80}")
print(f"Summary: Found bounding boxes for {len(bounds_df)} unique Layer_IDs")
print(f"{'='*80}")

# Display first few rows
bounds_df.to_csv('field_coordinates_summary.csv', index=False)