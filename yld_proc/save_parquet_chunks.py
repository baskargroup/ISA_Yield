import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import HeatMap
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import csv
import glob

# Directory containing all ISA_YYYY_raw_yields.csv files
input_dir = 'raw_yield/2025'
csv_files = glob.glob(os.path.join(input_dir, 'ISA_*_raw_yields.csv'))
chunk_size = 1000

def process_csv_file(input_file):
    year = input_file.split('_')[1]
    output_dir = f"{input_file.split('.csv')[0]}_chunks"
    os.makedirs(output_dir, exist_ok=True)

    # Precompute total number of chunks for progress bar
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        row_count = sum(1 for _ in reader) - 1  # Subtract 1 for header
    total_chunks = (row_count // chunk_size) + (1 if row_count % chunk_size else 0)

    def process_chunk(chunk, i, output_dir):
        gdf_chunk = gpd.GeoDataFrame(chunk, geometry=gpd.points_from_xy(chunk['x'], chunk['y']), crs="EPSG:4326")
        file_path = os.path.join(output_dir, f'chunk_{i}.parquet')
        gdf_chunk.to_parquet(file_path, index=False)

    def worker(queue, output_dir):
        while True:
            item = queue.get()
            if item is None:
                break
            chunk, i = item
            process_chunk(chunk, i, output_dir)

    queue = mp.Queue()
    num_workers = min(mp.cpu_count(), 4)  # Limit workers per file to avoid overloading
    workers = [mp.Process(target=worker, args=(queue, output_dir)) for _ in range(num_workers)]
    for w in workers:
        w.start()

    chunk_iter = pd.read_csv(input_file, dtype={'x': 'float32', 'y': 'float32'}, chunksize=chunk_size)
    for i, chunk in tqdm(enumerate(chunk_iter), total=total_chunks, desc=f"Queuing chunks {year}"):
        queue.put((chunk, i))

    for _ in workers:
        queue.put(None)
    for w in workers:
        w.join()

if __name__ == "__main__":
    for csv_file in tqdm(csv_files, total=len(csv_files), desc="Processing all years"):
        process_csv_file(csv_file)