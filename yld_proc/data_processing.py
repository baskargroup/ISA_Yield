import geopandas as gpd
import os 
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import concurrent.futures
import glob
import re
import pdb
def read_geoparquet(file):
    try:
        return gpd.read_parquet(file)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        return None

def combine_geoparquets(input_folder, output_file):
    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        raise ValueError(f"Input folder {input_folder} does not exist")
    parquet_files = list(input_folder.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No GeoParquet files found in {input_folder}")
    gdfs = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for gdf in tqdm(executor.map(read_geoparquet, parquet_files), total=len(parquet_files), desc=f"Processing GeoParquet files in {input_folder}"):
            if gdf is not None:
                gdfs.append(gdf)
    if not gdfs:
        raise ValueError("No valid GeoParquet files could be read")
    full_data = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    output_file = Path(output_file)
    full_data.to_parquet(output_file, index=False)
    print(f"Combined GeoParquet saved to {output_file}")

def filter_data(year):
    full_data = gpd.read_parquet(f'Yield_{year}.parquet')
    full_data = full_data.iloc[:,1:]
    full_data = full_data.drop_duplicates(subset='geometry', keep='first')
    crop_class = pd.read_csv(f'./raw_yield/Crop_Classification_{year}.csv')
    crop_class = crop_class.rename(columns={'X_cent': 'x', 'Y_cent': 'y'})
    crop_mapping = dict(zip(crop_class['Layer_ID'], crop_class['Crop']))
    full_data['Crop'] = full_data['Layer_ID'].map(crop_mapping)
    full_data.dropna(subset = ['Crop'],inplace=True)
    full_data_filtered = full_data[
        ((full_data['Crop'] != 'Soybean') | ((full_data['Yield'] >= 0) & (full_data['Yield'] <= 150))) &
        ((full_data['Crop'] != 'Corn') | ((full_data['Yield'] >= 0) & (full_data['Yield'] <= 550)))
    ]
    full_data_filtered.to_parquet(f'Yield_{year}_filtered.parquet')
    print(f'Saved the processed data as Yield_{year}_filtered.parquet')

def process_year(year):
    input_folder = f"./raw_yield/ISA_{year}_raw_yields_chunks"
    output_file = f"./Yield_{year}.parquet"
    combine_geoparquets(input_folder, output_file)
    filter_data(year)

if __name__ == "__main__":
    # Find all chunk folders matching the pattern
    chunk_dirs = glob.glob("./raw_yield/ISA_*_raw_yields_chunks")
    years = []
    
    for d in chunk_dirs:
        m = re.match(r"\./raw_yield/ISA_(\d{4})_raw_yields_chunks", d)
        if m:
            years.append(m.group(1))
    print(f"Found years: {years}")

    # Process all years in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_year, years), total=len(years), desc="Processing all years"))