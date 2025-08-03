import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import HeatMap
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import csv

# Step 1: Convert CSV to GeoParquet in chunks of 1000 using parallel processing
input_file = 'ISA_2024_Raw_Yields.csv'  # Replace with your CSV file path
chunk_size = 1000
output_dir = f"{input_file.split('.')[0]}_chunks"
os.makedirs(output_dir, exist_ok=True)  # Create directory for chunks

# Precompute total number of chunks for progress bar
with open(input_file, 'r') as f:
    reader = csv.reader(f)
    row_count = sum(1 for _ in reader) - 1  # Subtract 1 for header
total_chunks = (row_count // chunk_size) + (1 if row_count % chunk_size else 0)

# Function to process and save a chunk
def process_chunk(chunk, i, output_dir):
    gdf_chunk = gpd.GeoDataFrame(chunk, geometry=gpd.points_from_xy(chunk['x'], chunk['y']), crs="EPSG:4326")
    file_path = os.path.join(output_dir, f'chunk_{i}.parquet')
    gdf_chunk.to_parquet(file_path,index=False)
    print(f"Saved chunk {i} to {file_path}")

# Worker function for multiprocessing
def worker(queue, output_dir):
    while True:
        item = queue.get()
        if item is None:
            break
        chunk, i = item
        process_chunk(chunk, i, output_dir)

# Set up queue and workers
queue = mp.Queue()
num_workers = mp.cpu_count()  # Use available CPU cores; adjust if needed
workers = [mp.Process(target=worker, args=(queue, output_dir)) for _ in range(num_workers)]
for w in workers:
    w.start()

# Read CSV in chunks and put into queue with progress bar
chunk_iter = pd.read_csv(input_file, dtype={'x': 'float32', 'y': 'float32'}, chunksize=chunk_size)
for i, chunk in tqdm(enumerate(chunk_iter), total=total_chunks, desc="Queuing chunks"):
    queue.put((chunk, i))

# Signal workers to stop
for _ in workers:
    queue.put(None)

# Wait for workers to finish
for w in workers:
    w.join()

# Step 2: Read all GeoParquet chunks in parallel and combine
def load_chunk(file_path):
    return gpd.read_parquet(file_path)

parquet_files = [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir)) if f.endswith('.parquet')]

with ProcessPoolExecutor() as executor:
    future_to_file = {executor.submit(load_chunk, file): file for file in parquet_files}
    gdf_list = []
    for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Loading chunks"):
        gdf_list.append(future.result())

# Concatenate all chunks into full GeoDataFrame
full_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs="EPSG:4326")
print(f"Reconstructed dataset with {len(full_gdf)} rows")

# Step 3: Visualize with heatmap
map_center = [full_gdf['geometry'].y.mean(), full_gdf['geometry'].x.mean()]
mymap = folium.Map(location=map_center, zoom_start=5)

# Prepare data for heatmap with progress bar
heat_data = []
for point, value in tqdm(zip(full_gdf.geometry, full_gdf['Yld_Vol_Dr']), total=len(full_gdf), desc="Preparing heat data"):
    heat_data.append([point.y, point.x, value])

HeatMap(heat_data, radius=15, blur=10).add_to(mymap)

# Save map
mymap.save('full_geoparquet_map.html')
print("Map saved as 'full_geoparquet_map.html'")