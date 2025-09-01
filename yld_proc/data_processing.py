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

def filter_data(year):
    full_data = gpd.read_parquet(f'Yield_{year}.parquet')
    full_data = full_data.iloc[:,1:]
    full_data = full_data.drop_duplicates(subset='geometry', keep='first')
    crop_class = pd.read_csv(f'yld_proc/Crop_Classification_{year}.csv')
    # Renaming specific columns
    crop_class = crop_class.rename(columns={'X_cent': 'x', 'Y_cent': 'y'})
    # Using map (efficient, handles duplicates in df1)
    crop_mapping = dict(zip(crop_class['Layer_ID'], crop_class['Crop']))
    full_data['Crop'] = full_data['Layer_ID'].map(crop_mapping)
    full_data.dropna(subset = ['Crop'],inplace=True)
    # Filter df1 based on the conditions
    full_data_filtered = full_data[
        # Keep rows where Crop is not Soybean or Yld_Vol_Dr is in range 0-150 for Soybean
        ((full_data['Crop'] != 'Soybean') | ((full_data['Yld_Vol_Dr'] >= 0) & (full_data['Yld_Vol_Dr'] <= 150))) &
        # Keep rows where Crop is not Corn or Yld_Vol_Dr is in range 0-550 for Corn
        ((full_data['Crop'] != 'Corn') | ((full_data['Yld_Vol_Dr'] >= 0) & (full_data['Yld_Vol_Dr'] <= 550)))
    ]
    # full_data_filtered = full_data_filtered.set_index(['x','y'])
    full_data_filtered.to_parquet(f'Yield_{year}_filtered.parquet')
    print(f'Saved the processed data as Yield_{year}_filtered.parquet')

if __name__ == "__main__":
    # Example usage
    year = '2024'
    input_folder = f"./ISA_{year}_raw_yields_chunks"
    output_file = f"./Yield_{input_folder.split('_')[1]}.parquet"
    combine_geoparquets(input_folder, output_file)
    filter_data(year)