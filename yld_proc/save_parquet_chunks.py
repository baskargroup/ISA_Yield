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
input_dir = 'raw_yield'
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

    # Optional: Combine and visualize for each year
    # parquet_files = [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir)) if f.endswith('.parquet')]
    # with ProcessPoolExecutor() as executor:
    #     future_to_file = {executor.submit(gpd.read_parquet, file): file for file in parquet_files}
    #     gdf_list = []
    #     for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc=f"Loading chunks {year}"):
    #         gdf_list.append(future.result())
    # full_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs="EPSG:4326")
    # print(f"[{year}] Reconstructed dataset with {len(full_gdf)} rows")

    # # Visualize with heatmap
    # map_center = [full_gdf['geometry'].y.mean(), full_gdf['geometry'].x.mean()]
    # mymap = folium.Map(location=map_center, zoom_start=5)
    # heat_data = []
    # for point, value in tqdm(zip(full_gdf.geometry, full_gdf['Yield']), total=len(full_gdf), desc=f"Preparing heat data {year}"):
    #     heat_data.append([point.y, point.x, value])
    # HeatMap(heat_data, radius=15, blur=10).add_to(mymap)
    # map_file = f'{input_file.split(".csv")[0]}_map.html'
    # mymap.save(map_file)
    # print(f"[{year}] Map saved as '{map_file}'")

if __name__ == "__main__":
    for csv_file in tqdm(csv_files, total=len(csv_files), desc="Processing all years"):
        process_csv_file(csv_file)