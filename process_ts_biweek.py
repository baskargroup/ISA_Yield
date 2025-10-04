import os
import glob
import numpy as np
import rasterio
import pdb
import sys
import re
from collections import defaultdict

def get_aspect_ratio(H, W):
    return max(H, W) / min(H, W)

def is_rectangular(H, W, threshold=1.2):
    return get_aspect_ratio(H, W) > threshold

def is_almost_square(H, W, lower_threshold=1.0, upper_threshold=1.2):
    aspect = get_aspect_ratio(H, W)
    return lower_threshold < aspect <= upper_threshold and H != W

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

def get_monthly_indices(file, used_dates_dict):
    """
    For a given file, return a list of lists, each containing indices of dates for each month (April to September).
    """
    base = os.path.basename(file)
    # Remove .npy or .tif.npy extension for matching
    if base.endswith('.tif.npy'):
        base = base[:-8] + '.tif'
    elif base.endswith('.npy'):
        base = base[:-4] + '.tif'
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
    if base.endswith('.tif.npy'):
        base = base[:-8] + '.tif'
    elif base.endswith('.npy'):
        base = base[:-4] + '.tif'
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
    If no data for a biweek, fill with zeros.
    """
    T, C, H, W = data.shape
    biweekly_data = []
    for inds in biweekly_indices:
        if inds:
            biweekly_data.append(np.mean(data[inds], axis=0, keepdims=True))
        else:
            biweekly_data.append(np.zeros((1, C, H, W), dtype=data.dtype))
    return np.concatenate(biweekly_data, axis=0)

def process_subfolder(subfolder_path, dest_subfolder_path, ref_dims=None, is_reference=False, used_dates_dict=None):
    if is_reference:
        files = sorted(glob.glob(os.path.join(subfolder_path, '*.tif')) + glob.glob(os.path.join(subfolder_path, '*.tiff')))
    else:
        files = sorted(glob.glob(os.path.join(subfolder_path, '*.tif.npy')) + glob.glob(os.path.join(subfolder_path, '*.npy')))
    rectangular_files = []
    almost_square_files = []
    square_files = []
    for file in files:
        if is_reference:
            with rasterio.open(file) as ds:
                data = ds.read()
                C, H, W = data.shape
        else:
            data = np.load(file)
            T, C, H, W = data.shape
            # Aggregate biweekly before filtering by T
            if used_dates_dict is not None:
                biweekly_indices, dates = get_biweekly_indices(file, used_dates_dict)
                data = aggregate_biweekly(data, biweekly_indices)
                T = data.shape[0]
            if T < 12:
                continue
        if is_rectangular(H, W, threshold=1.2):
            rectangular_files.append(file)
        elif is_almost_square(H, W, lower_threshold=1.0, upper_threshold=1.2):
            almost_square_files.append(file)
        else:
            square_files.append(file)
    # Create destination subfolder if it doesn't exist
    os.makedirs(dest_subfolder_path, exist_ok=True)
    # Process rectangular images (concatenate with self)
    for file in rectangular_files:
        if is_reference:
            with rasterio.open(file) as ds:
                data = ds.read()
                C, H, W = data.shape
        else:
            data = np.load(file)
            # Aggregate biweekly
            if used_dates_dict is not None:
                biweekly_indices, dates = get_biweekly_indices(file, used_dates_dict)
                data = aggregate_biweekly(data, biweekly_indices)
                T = data.shape[0]
        if not is_reference and ref_dims:
            ref_h, ref_w = ref_dims.get(os.path.basename(file).replace('.tif.npy', '.npy'), (H, W))
            data = rescale_data(data, ref_h, ref_w)
            H, W = ref_h, ref_w
        # Concatenate the same image
        if is_reference:
            if W > H:
                reps = int(np.round(W / H))
                new_h = H * reps
                new_w = W
                new_data = np.tile(data, (1, reps, 1))[:, :new_h, :]
            else:
                reps = int(np.round(H / W))
                new_h = H
                new_w = W * reps
                new_data = np.tile(data, (1, 1, reps))[:, :, :new_w]
        else:
            if W > H:
                reps = int(np.round(W / H))
                new_h = H * reps
                new_w = W
                new_data = np.tile(data, (1, 1, reps, 1))[:, :, :new_h, :]
            else:
                reps = int(np.round(H / W))
                new_h = H
                new_w = W * reps
                new_data = np.tile(data, (1, 1, 1, reps))[:, :, :, :new_w]
            data = new_data
        # Crop to largest square
        data = crop_to_largest_square(data, is_reference)
        # Rescale to 224x224
        if data.shape[-1] < 10 or data.shape[-2] < 10:
            print(f"Warning: {file} has very small dimensions after cropping: {data.shape} Skipping.")
            continue
        data = rescale_to_224x224(data, is_reference)
        # Save with .npy extension in destination subfolder
        if is_reference:
            save_path = os.path.join(dest_subfolder_path, os.path.splitext(os.path.basename(file))[0] + '.npy')
        else:
            save_path = os.path.join(dest_subfolder_path, os.path.splitext(os.path.splitext(os.path.basename(file))[0])[0] + '.npy')
        np.save(save_path, data)
    # Process almost square images (crop to perfect square)
    for file in almost_square_files:
        if is_reference:
            with rasterio.open(file) as ds:
                data = ds.read()
                C, H, W = data.shape
        else:
            data = np.load(file)
            if used_dates_dict is not None:
                biweekly_indices, dates = get_biweekly_indices(file, used_dates_dict)
                data = aggregate_biweekly(data, biweekly_indices)
                T = data.shape[0]
        if not is_reference and ref_dims:
            ref_h, ref_w = ref_dims.get(os.path.basename(file).replace('.tif.npy', '.npy'), (H, W))
            data = rescale_data(data, ref_h, ref_w)
            H, W = ref_h, ref_w
        # Crop to largest square
        data = crop_to_largest_square(data, is_reference)
        # Rescale to 224x224
        if data.shape[-1] < 10 or data.shape[-2] < 10:
            print(f"Warning: {file} has very small dimensions after cropping: {data.shape} Skipping.")
            continue        
        data = rescale_to_224x224(data, is_reference)
        # Save with .npy extension in destination subfolder
        if is_reference:
            save_path = os.path.join(dest_subfolder_path, os.path.splitext(os.path.basename(file))[0] + '.npy')
        else:
            save_path = os.path.join(dest_subfolder_path, os.path.splitext(os.path.splitext(os.path.basename(file))[0])[0] + '.npy')
        np.save(save_path, data)
    # Process square images
    for file in square_files:
        if is_reference:
            with rasterio.open(file) as ds:
                data = ds.read()
                C, H, W = data.shape
        else:
            data = np.load(file)
            if used_dates_dict is not None:
                biweekly_indices, dates = get_biweekly_indices(file, used_dates_dict)
                data = aggregate_biweekly(data, biweekly_indices)
                T = data.shape[0]
        if not is_reference and ref_dims:
            ref_h, ref_w = ref_dims.get(os.path.basename(file).replace('.tif.npy', '.npy'), (H, W))
            data = rescale_data(data, ref_h, ref_w)
            H, W = ref_h, ref_w
        # Crop to largest square (noop if perfect square)
        data = crop_to_largest_square(data, is_reference)
        # Rescale to 224x224
        if data.shape[-1] < 10 or data.shape[-2] < 10:
            print(f"Warning: {file} has very small dimensions after cropping: {data.shape} Skipping.")
            continue
        data = rescale_to_224x224(data, is_reference)
        # Save with .npy extension in destination subfolder
        if is_reference:
            save_path = os.path.join(dest_subfolder_path, os.path.splitext(os.path.basename(file))[0] + '.npy')
        else:
            save_path = os.path.join(dest_subfolder_path, os.path.splitext(os.path.splitext(os.path.basename(file))[0])[0] + '.npy')
        np.save(save_path, data)

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

def process_root_folder(root_path, dest_root_path, used_dates_dict=None):
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
                process_subfolder(subfolder_path, dest_subfolder_path, is_reference=True, used_dates_dict=used_dates_dict)
            else:
                print(f"Processing folder: {subfolder}")
                process_subfolder(subfolder_path, dest_subfolder_path, ref_dims=ref_dims, used_dates_dict=used_dates_dict)

if __name__ == "__main__":
    log_path = './unprocessed_data/used_dates_log.txt'
    used_dates_dict = parse_used_dates_log(log_path)
    dst_folder = f'processed_data_biweekly'
    os.makedirs(dst_folder, exist_ok=True)
    process_root_folder('./unprocessed_data', dst_folder, used_dates_dict=used_dates_dict)