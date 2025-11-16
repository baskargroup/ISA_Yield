import os
import glob
import numpy as np
import rasterio
import pdb
import sys
import re
from collections import defaultdict
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

def resize_nearest(data, new_h, new_w, is_reference=False):
    if not is_reference:
        t, c, old_h, old_w = data.shape
    else:
        c, old_h, old_w = data.shape
    ii, jj = np.meshgrid(np.arange(new_h), np.arange(new_w), indexing='ij')
    orig_i = np.floor(ii * old_h / new_h).astype(int)
    orig_j = np.floor(jj * old_w / new_w).astype(int)
    orig_i = np.clip(orig_i, 0, old_h - 1)
    orig_j = np.clip(orig_j, 0, old_w - 1)
    if not is_reference:
        return data[:, :, orig_i, orig_j]
    else:
        return data[:, orig_i, orig_j]

def rescale_to_224x224(data, is_reference=False):
    return resize_nearest(data, 224, 224, is_reference)

def rescale_data(data, ref_h, ref_w):
    return resize_nearest(data, ref_h, ref_w)

def crop_to_largest_square(data, is_reference = False):
    if is_reference == False:
        H = data.shape[2]
        W = data.shape[3]
    else:
        H = data.shape[1]
        W = data.shape[2]
    side = min(H, W)
    if H == side and W == side:
        return data
    if is_reference == False:
        if H > W:
            start_row = (H - side) // 2
            data = data[:, :, start_row:start_row + side, :]
        else:
            start_col = (W - side) // 2
            data = data[:, :, :, start_col:start_col + side]
    else:
        if H > W:
            start_row = (H - side) // 2
            data = data[:, start_row:start_row + side, :]
        else:
            start_col = (W - side) // 2
            data = data[:, :, start_col:start_col + side]
    return data

def tile_to_min_size(data, min_size=224, is_reference=False):
    """
    Tile the spatial dimensions (H, W) of the data until both are at least `min_size`.
    - For non-reference tensors: (T, C, H, W)
    - For reference tensors: (C, H, W)
    Returns a new array with repeated tiles along H and/or W.
    """
    if not is_reference:
        T, C, H, W = data.shape
    else:
        C, H, W = data.shape

    reps_h = 1 if H >= min_size else int(np.ceil(min_size / H))
    reps_w = 1 if W >= min_size else int(np.ceil(min_size / W))

    if reps_h == 1 and reps_w == 1:
        return data

    if not is_reference:
        # (T, C, H, W) -> repeat along H and W
        tiled = np.tile(data, (1, 1, reps_h, reps_w))
    else:
        # (C, H, W) -> repeat along H and W
        tiled = np.tile(data, (1, reps_h, reps_w))

    return tiled

def get_monthly_indices(file, used_dates_dict):
    """
    For a given file, return a list of lists, each containing indices of dates for each month (April to September).
    """
    base = os.path.basename(file)
    # Remove .npy or .tif.npy extension for matching
    base = base.split('.')[0] + '.tif' 
    # Get available dates for this file
    dates = used_dates_dict.get(base, [])
    # Group indices by month (April=4, ..., September=9)
    month_to_indices = defaultdict(list)
    for idx, d in enumerate(dates):
        m = int(d[5:7])
        if 4 <= m <= 9:
            month_to_indices[m].append(idx)
    # For each month, get the indices (if none, will be empty)
    monthly_indices = []
    for m in range(4, 10):
        monthly_indices.append(month_to_indices[m])
    return monthly_indices, dates

def aggregate_monthly(data, monthly_indices):
    """
    Given data of shape (T, C, H, W) and monthly_indices (list of lists of indices),
    return (6, C, H, W) where each timepoint is the average of that month's available data.
    If no data for a month, fill with zeros.
    """
    T, C, H, W = data.shape
    monthly_data = []
    for inds in monthly_indices:
        if inds:
            monthly_data.append(np.mean(data[inds], axis=0, keepdims=True))
        else:
            monthly_data.append(np.zeros((1, C, H, W), dtype=data.dtype))
    return np.concatenate(monthly_data, axis=0)

def get_biweekly_indices(file, used_dates_dict):
    """
    For a given file, return a list of lists, each containing indices of dates for each biweek (April 1 to September 30).
    Each month is split into two biweeks: 1-15, 16-end.
    Returns (biweekly_indices, dates)
    """
    base = os.path.basename(file)
    # Remove .npy or .tif.npy extension for matching
    base = base.split('.')[0] + '.tif'
    # Get available dates for this file
    dates = used_dates_dict.get(base, [])

    # Group indices by biweek (April 1 to Sept 30, 2 per month)
    biweek_to_indices = defaultdict(list)
    for idx, d in enumerate(dates):
        m = int(d[5:7])
        day = int(d[8:10])
        if 4 <= m <= 9:
            # biweek 0: Apr 1-15, 1: Apr 16-30, ..., 10: Sep 1-15, 11: Sep 16-30
            biweek = (m - 4) * 2
            if day > 15:
                biweek += 1
            biweek_to_indices[biweek].append(idx)
    # For each biweek, get the indices (if none, will be empty)
    biweekly_indices = []
    for b in range(12):  # 6 months * 2
        biweekly_indices.append(biweek_to_indices[b])
    return biweekly_indices, dates

def aggregate_biweekly(data, biweekly_indices):
    """
    Given data of shape (T, C, H, W) and biweekly_indices (list of lists of indices),
    return (12, C, H, W) where each timepoint is the average of that biweek's available data.
    If no data for a biweek, fill with mean of previous and next valid biweek (if available), else zeros.
    Ignores data slices that are all zeros when aggregating.
    """
    T, C, H, W = data.shape
    biweekly_data = []
    valid_means = [None] * len(biweekly_indices)
    # First, compute means for valid biweeks
    for i, inds in enumerate(biweekly_indices):
        if inds:
            valid_slices = [data[idx] for idx in inds if not np.all(data[idx] == 0)]
            if valid_slices:
                stacked = np.stack(valid_slices, axis=0)
                valid_means[i] = np.mean(stacked, axis=0)
    # Now, fill missing biweeks with mean of previous and next valid biweek
    for i in range(len(biweekly_indices)):
        if valid_means[i] is not None:
            biweekly_data.append(valid_means[i][None])
        else:
            print(f"Biweek {i} has no valid data, filling...")
            # Find previous valid
            prev = None
            for j in range(i-1, i-2, -1):
                if valid_means[j] is not None:
                    prev = valid_means[j]
                    break
            # Find next valid
            next_ = None
            if i < len(biweekly_indices) - 1:
                for j in range(i+1, i+2):
                    if valid_means[j] is not None:
                        next_ = valid_means[j]
                        break
            if prev is not None and next_ is not None:
                fill = ((prev + next_) / 2)[None]
            elif prev is not None:
                fill = prev[None]
            elif next_ is not None:
                fill = next_[None]
            else:
                fill = np.zeros((1, C, H, W), dtype=data.dtype)
            biweekly_data.append(fill)
    return np.concatenate(biweekly_data, axis=0)

def process_single_file(file, dest_subfolder_path, modality_name, is_reference, used_dates_dict, ref_dims, tsave):
    """
    Process a single file and save it. This function is designed to be called in parallel.
    """
    static_modalities = {"CDL", "DEM", "SOIL"}
    norm_max_corn = 370.0
    norm_max_soybean = 150.0
    
    try:
        if is_reference:
            with rasterio.open(file) as ds:
                data = ds.read()
                C, H, W = data.shape
        else:
            data = np.load(file)
            # Only aggregate if not static modality
            if used_dates_dict is not None and modality_name not in static_modalities:
                biweekly_indices, dates = get_biweekly_indices(file, used_dates_dict)
                data = aggregate_biweekly(data, biweekly_indices[:tsave+1])
            # After potential aggregation, capture dims
            T, C, H, W = data.shape
        
        if not is_reference and ref_dims:
            ref_h, ref_w = ref_dims.get(os.path.basename(file).replace('.tif.npy', '.npy'), (H, W))
            data = rescale_data(data, ref_h, ref_w)
            H, W = ref_h, ref_w
        
        # Tile to reach at least 224x224, then crop to largest square
        data = tile_to_min_size(data, min_size=224, is_reference=is_reference)
        # Crop to largest square
        data = crop_to_largest_square(data, is_reference)
        # Rescale to 224x224
        if data.shape[-1] < 10 or data.shape[-2] < 10:
            print(f"Warning: {file} has very small dimensions after cropping: {data.shape} Skipping.")
            return None
        data = rescale_to_224x224(data, is_reference)
        # Only keep the first tsave timepoints (for non-reference)
        if not is_reference:
            data = data[:tsave]
        # Save with .npy extension in destination subfolder
        if is_reference:
            save_path = os.path.join(dest_subfolder_path, os.path.splitext(os.path.basename(file))[0] + '.npy')
            data[data < 30] = 0.0  # Set values below 30 to 0
            data = data.clip(0, norm_max_corn if 'corn' in file.lower() else norm_max_soybean)
            data = data / (norm_max_corn if 'corn' in file.lower() else norm_max_soybean)
        else:
            save_path = os.path.join(dest_subfolder_path, os.path.splitext(os.path.splitext(os.path.basename(file))[0])[0] + '.npy')
        # Ensure float32 before saving
        data = data.astype(np.float32)
        np.save(save_path, data)
        return file
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

def process_subfolder(subfolder_path, dest_subfolder_path, ref_dims=None, is_reference=False, used_dates_dict=None, tsave=12, num_workers=None):
    modality_name = os.path.basename(subfolder_path)
    if is_reference:
        files = sorted(glob.glob(os.path.join(subfolder_path, '*.tif')) + glob.glob(os.path.join(subfolder_path, '*.tiff')))
    else:
        files = sorted(glob.glob(os.path.join(subfolder_path, '*.tif.npy')) + glob.glob(os.path.join(subfolder_path, '*.npy')))
    
    # Create destination subfolder if it doesn't exist
    os.makedirs(dest_subfolder_path, exist_ok=True)
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    # Process all images with parallel processing
    process_func = partial(
        process_single_file,
        dest_subfolder_path=dest_subfolder_path,
        modality_name=modality_name,
        is_reference=is_reference,
        used_dates_dict=used_dates_dict,
        ref_dims=ref_dims,
        tsave=tsave
    )
    
    if len(files) > 0:
        print(f"Processing {len(files)} files using {num_workers} workers...")
        with Pool(num_workers) as pool:
            results = pool.map(process_func, files)
        processed = sum(1 for r in results if r is not None)
        print(f"Successfully processed {processed}/{len(files)} files")

def parse_used_dates_log(log_path):
    """
    Parse used_dates_log.txt and return a dict: {filename: [date1, date2, ...]}
    """
    used_dates_dict = {}
    with open(log_path, "r") as f:
        for line in f:
            if ':' not in line:
                continue
            fname, dates_str = line.strip().split(':', 1)
            dates = [d.strip() for d in dates_str.split(',') if d.strip()]
            used_dates_dict[fname] = dates
    return used_dates_dict

def process_root_folder(root_path, dest_root_path, used_dates_dict=None, tsave=12, all=True, num_workers=None):
    def compute_assigned_iterations(total_iters=12, num_nodes=1, node_rank=0):
        """Return list of iteration indices (1-based) assigned to this node.
        Distribute as evenly as possible: first (total_iters % num_nodes) nodes get one extra.
        """
        if num_nodes <= 1:
            return list(range(1, total_iters + 1))
        base = total_iters // num_nodes
        extra = total_iters % num_nodes
        # Node ranks < extra get (base+1) items
        if node_rank < extra:
            start = node_rank * (base + 1) + 1
            end = start + (base + 1) - 1
        else:
            start = extra * (base + 1) + (node_rank - extra) * base + 1
            end = start + base - 1
        return list(range(start, end + 1))

    # Obtain node distribution info from environment or args
    env_num_nodes = int(os.environ.get('NUM_NODES', os.environ.get('SLURM_JOB_NUM_NODES', '1')))
    env_node_rank = int(os.environ.get('NODE_RANK', os.environ.get('SLURM_NODEID', '0')))
    # Allow overriding via function parameters by passing in a tuple via dest_root_path (not used) or environment.
    num_nodes = env_num_nodes
    node_rank = env_node_rank

    if all == True:
        assigned_iters = compute_assigned_iterations(total_iters=12, num_nodes=num_nodes, node_rank=node_rank)
        print(f"Node {node_rank}/{num_nodes - 1 if num_nodes>0 else 0} assigned iterations: {assigned_iters}")
        for i in assigned_iters:
            dest_root_path = f'processed_data_biweekly_{i}'
            os.makedirs(dest_root_path, exist_ok=True)
            yield_folder = os.path.join(root_path, 'yield_geotiffs')
            if not os.path.exists(yield_folder):
                print("yield_geotiffs subfolder not found!")
                return
            ref_files = sorted(glob.glob(os.path.join(yield_folder, '*.tif')) + glob.glob(os.path.join(yield_folder, '*.tiff')))
            ref_dims = {}
            for ref_file in ref_files:
                with rasterio.open(ref_file) as ds:
                    ref_dims[os.path.splitext(os.path.basename(ref_file))[0] + '.npy'] = (ds.height, ds.width)
            for subfolder in sorted(os.listdir(root_path)):
                subfolder_path = os.path.join(root_path, subfolder)
                dest_subfolder_path = os.path.join(dest_root_path, subfolder)
                if os.path.isdir(subfolder_path):
                    if subfolder == 'yield_geotiffs':
                        print(f"Processing reference folder: {subfolder}")
                        process_subfolder(subfolder_path, 
                                          dest_subfolder_path, 
                                          is_reference=True, 
                                          used_dates_dict=used_dates_dict, 
                                          tsave=i, 
                                          num_workers=num_workers)
                    else:
                        print(f"Processing folder: {subfolder}")
                        process_subfolder(subfolder_path, 
                                          dest_subfolder_path, 
                                          ref_dims=ref_dims, 
                                          used_dates_dict=used_dates_dict, 
                                          tsave=i, 
                                          num_workers=num_workers)
    else:
        yield_folder = os.path.join(root_path, 'yield_geotiffs')
        if not os.path.exists(yield_folder):
            print("yield_geotiffs subfolder not found!")
            return
        ref_files = sorted(glob.glob(os.path.join(yield_folder, '*.tif')) + glob.glob(os.path.join(yield_folder, '*.tiff')))
        ref_dims = {}
        for ref_file in ref_files:
            with rasterio.open(ref_file) as ds:
                ref_dims[os.path.splitext(os.path.basename(ref_file))[0] + '.npy'] = (ds.height, ds.width)
        for subfolder in sorted(os.listdir(root_path)):
            subfolder_path = os.path.join(root_path, subfolder)
            dest_subfolder_path = os.path.join(dest_root_path, subfolder)
            if os.path.isdir(subfolder_path):
                if subfolder == 'yield_geotiffs':
                    print(f"Processing reference folder: {subfolder}")
                    process_subfolder(subfolder_path, dest_subfolder_path, is_reference=True, used_dates_dict=used_dates_dict, tsave=tsave, num_workers=num_workers)
                else:
                    print(f"Processing folder: {subfolder}")
                    process_subfolder(subfolder_path, dest_subfolder_path, ref_dims=ref_dims, used_dates_dict=used_dates_dict, tsave=tsave, num_workers=num_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ts', type=int, default=12, help='Number of timepoints to save (default: 12)')
    parser.add_argument('--all', action='store_true', default=False, help='Process all timepoints (1-12)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: cpu_count-1)')
    parser.add_argument('--num-nodes', type=int, default=None, help='Total number of nodes participating (overrides env NUM_NODES/SLURM_JOB_NUM_NODES)')
    parser.add_argument('--node-rank', type=int, default=None, help='Rank of this node (overrides env NODE_RANK/SLURM_NODEID)')
    args = parser.parse_args()

    log_path = './unprocessed_data/used_dates_log.txt'
    used_dates_dict = parse_used_dates_log(log_path)
    dst_folder = f'processed_data_biweekly_{args.ts}'
    os.makedirs(dst_folder, exist_ok=True)
    # Inject environment overrides if CLI args provided
    if args.num_nodes is not None:
        os.environ['NUM_NODES'] = str(args.num_nodes)
    if args.node_rank is not None:
        os.environ['NODE_RANK'] = str(args.node_rank)
    process_root_folder('./unprocessed_data', dst_folder, used_dates_dict=used_dates_dict, tsave=args.ts, all=args.all, num_workers=args.workers)