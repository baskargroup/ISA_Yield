import geopandas as gpd
import os 
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import concurrent.futures

def read_geoparquet(file):
    try:
        return gpd.read_parquet(file)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        return None

def combine_geoparquets(input_folder, output_file):
    # Ensure input folder exists
    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        raise ValueError(f"Input folder {input_folder} does not exist")

    # Get list of all .parquet files in the folder
    parquet_files = list(input_folder.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No GeoParquet files found in {input_folder}")

    # Read all GeoParquet files in parallel with progress bar
    gdfs = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for gdf in tqdm(executor.map(read_geoparquet, parquet_files), total=len(parquet_files), desc="Processing GeoParquet files"):
            if gdf is not None:
                gdfs.append(gdf)

    if not gdfs:
        raise ValueError("No valid GeoParquet files could be read")

    # Concatenate all GeoDataFrames
    full_data = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

    # Write to output GeoParquet file
    output_file = Path(output_file)
    full_data.to_parquet(output_file, index=False)
    print(f"Combined GeoParquet saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    input_folder = "./ISA_2024_Raw_Yields_chunks"
    output_file = f"./Yield_{input_folder.split('_')[1]}.parquet"
    combine_geoparquets(input_folder, output_file)