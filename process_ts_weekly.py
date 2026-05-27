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
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

ignore_list = []

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
        tiled = np.tile(data, (1, 1, reps_h, reps_w))
    else:
        tiled = np.tile(data, (1, reps_h, reps_w))

    return tiled


# ✅ NEW: force static modalities to fixed T
def force_static_T(data, target_T=24):
    """
    Accepts (T,C,H,W) or (C,H,W) and returns (target_T,C,H,W).
    If T < target_T: repeats in time.
    If T > target_T: truncates.
    """
    if data.ndim == 3:
        data = data[None, ...]  # (1,C,H,W)

    if data.ndim != 4:
        raise ValueError(f"Unsupported static data shape: {data.shape}")

    T = data.shape[0]
    if T == target_T:
        return data
    if T > target_T:
        return data[:target_T]

    reps = int(np.ceil(target_T / T))
    data_rep = np.tile(data, (reps, 1, 1, 1))
    return data_rep[:target_T]


def get_monthly_indices(file, used_dates_dict, modality_name):
    base = os.path.basename(file)
    base = base.split('.')[0] + '.tif'
    file_dates = used_dates_dict.get(base, {})
    dates = file_dates.get(modality_name.upper(), [])
    month_to_indices = defaultdict(list)
    for idx, d in enumerate(dates):
        m = int(d[5:7])
        if 4 <= m <= 9:
            month_to_indices[m].append(idx)
    monthly_indices = []
    for m in range(4, 10):
        monthly_indices.append(month_to_indices[m])
    return monthly_indices, dates

def aggregate_monthly(data, monthly_indices):
    T, C, H, W = data.shape
    monthly_data = []
    for inds in monthly_indices:
        if inds:
            monthly_data.append(np.mean(data[inds], axis=0, keepdims=True))
        else:
            monthly_data.append(np.zeros((1, C, H, W), dtype=data.dtype))
    return np.concatenate(monthly_data, axis=0)

def get_weekly_indices(file, used_dates_dict, modality_name):
    base = os.path.basename(file)
    base = base.split('.')[0] + '.tif'
    file_dates = used_dates_dict.get(base, {})
    dates = file_dates.get(modality_name.upper(), [])
    weekly_indices = [[] for _ in range(24)]

    for idx, d in enumerate(dates):
        m = int(d[5:7])
        day = int(d[8:10])
        if 4 <= m <= 9:
            if m == 4:
                days_since_start = day - 1
            elif m == 5:
                days_since_start = 30 + (day - 1)
            elif m == 6:
                days_since_start = 30 + 31 + (day - 1)
            elif m == 7:
                days_since_start = 30 + 31 + 30 + (day - 1)
            elif m == 8:
                days_since_start = 30 + 31 + 30 + 31 + (day - 1)
            elif m == 9:
                days_since_start = 30 + 31 + 30 + 31 + 31 + (day - 1)

            week = int(days_since_start / 7.625)
            week = min(week, 23)
            weekly_indices[week].append(idx)

    return weekly_indices, dates

def aggregate_weekly(data, weekly_indices):
    T, C, H, W = data.shape
    weekly_data = []
    valid_means = [None] * len(weekly_indices)

    for i, inds in enumerate(weekly_indices):
        if inds:
            valid_slices = [data[idx] for idx in inds if not np.all(data[idx] == 0)]
            if valid_slices:
                stacked = np.stack(valid_slices, axis=0)
                valid_means[i] = np.mean(stacked, axis=0)

    for i in range(len(weekly_indices)):
        if valid_means[i] is not None:
            weekly_data.append(valid_means[i][None])
        else:
            prev = None
            for j in range(i-1, -1, -1):
                if valid_means[j] is not None:
                    prev = valid_means[j]
                    break

            next_ = None
            for j in range(i+1, len(weekly_indices)):
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

            weekly_data.append(fill)

    return np.concatenate(weekly_data, axis=0)

def polynomial_curve(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def fill_remaining_zeros_with_nearest(data):
    if data.ndim == 4:
        T, C, H, W = data.shape
        has_time = True
    elif data.ndim == 3:
        C, H, W = data.shape
        has_time = False
    else:
        return data

    data_filled = data.copy()

    if has_time:
        for t in range(T):
            for c in range(C):
                channel_data = data_filled[t, c, :, :]
                zero_mask = channel_data == 0
                if np.any(zero_mask):
                    non_zero_mask = ~zero_mask
                    if np.any(non_zero_mask):
                        zero_coords = np.argwhere(zero_mask)
                        non_zero_coords = np.argwhere(non_zero_mask)
                        for zh, zw in zero_coords:
                            distances = np.sqrt(np.sum((non_zero_coords - np.array([zh, zw]))**2, axis=1))
                            nearest_idx = np.argmin(distances)
                            nh, nw = non_zero_coords[nearest_idx]
                            data_filled[t, c, zh, zw] = data_filled[t, c, nh, nw]
    else:
        for c in range(C):
            channel_data = data_filled[c, :, :]
            zero_mask = channel_data == 0
            if np.any(zero_mask):
                non_zero_mask = ~zero_mask
                if np.any(non_zero_mask):
                    zero_coords = np.argwhere(zero_mask)
                    non_zero_coords = np.argwhere(non_zero_mask)
                    for zh, zw in zero_coords:
                        distances = np.sqrt(np.sum((non_zero_coords - np.array([zh, zw]))**2, axis=1))
                        nearest_idx = np.argmin(distances)
                        nh, nw = non_zero_coords[nearest_idx]
                        data_filled[c, zh, zw] = data_filled[c, nh, nw]

    return data_filled

def fit_and_fill_zeros(data, method='polynomial', order=3):
    T, C, H, W = data.shape
    data_filled = data.copy()
    time_indices = np.arange(T)

    for c in range(C):
        for h in range(H):
            for w in range(W):
                temporal_profile = data[:, c, h, w]
                non_zero_mask = temporal_profile != 0
                non_zero_indices = time_indices[non_zero_mask]
                non_zero_values = temporal_profile[non_zero_mask]

                if len(non_zero_indices) >= 2:
                    zero_indices = time_indices[~non_zero_mask]
                    if len(zero_indices) > 0:
                        try:
                            if method == 'polynomial':
                                poly_order = min(order, len(non_zero_indices) - 1)
                                coeffs = np.polyfit(non_zero_indices, non_zero_values, poly_order)
                                poly = np.poly1d(coeffs)
                                filled_values = poly(zero_indices)
                            elif method == 'spline':
                                if len(non_zero_indices) > order:
                                    interp_func = interp1d(non_zero_indices, non_zero_values, kind=order, fill_value='extrapolate')
                                else:
                                    interp_func = interp1d(non_zero_indices, non_zero_values, kind='linear', fill_value='extrapolate')
                                filled_values = interp_func(zero_indices)
                            elif method == 'linear':
                                interp_func = interp1d(non_zero_indices, non_zero_values, kind='linear', fill_value='extrapolate')
                                filled_values = interp_func(zero_indices)

                            filled_values = np.maximum(filled_values, 0)
                            data_filled[zero_indices, c, h, w] = filled_values

                        except Exception:
                            try:
                                interp_func = interp1d(non_zero_indices, non_zero_values, kind='linear', fill_value='extrapolate')
                                filled_values = interp_func(zero_indices)
                                filled_values = np.maximum(filled_values, 0)
                                data_filled[zero_indices, c, h, w] = filled_values
                            except:
                                pass
    return data_filled

def process_single_file(file, dest_subfolder_paths, modality_name, is_reference, used_dates_dict, ref_dims, tsave_list, fill_zeros=False, interp_method='polynomial'):
    static_modalities = {"CDL", "DEM", "SOIL"}

    if file in ignore_list:
        return None

    try:
        if is_reference:
            with rasterio.open(file) as ds:
                data = ds.read()
                C, H, W = data.shape
        else:
            data = np.load(file)

            # ✅ NEW: force static modalities to T=24 right here
            if modality_name.upper() in static_modalities:
                data = force_static_T(data, target_T=24)

            # ✅ aggregate only dynamic modalities
            if used_dates_dict is not None and modality_name.upper() not in static_modalities:
                weekly_indices, dates = get_weekly_indices(file, used_dates_dict, modality_name)
                data = aggregate_weekly(data, weekly_indices)
                if data is None:
                    print(f"Warning: {file} could not be aggregated due to lack of valid data. Skipping.")
                    ignore_list.append(file)
                    return None

            # ✅ After (possible) force/aggregate, capture dims safely
            T, C, H, W = data.shape

        if not is_reference and ref_dims:
            ref_h, ref_w = ref_dims.get(os.path.basename(file).replace('.tif.npy', '.npy'), (H, W))
            data = rescale_data(data, ref_h, ref_w)
            H, W = ref_h, ref_w

        data = tile_to_min_size(data, min_size=224, is_reference=is_reference)
        data = crop_to_largest_square(data, is_reference)
        if data.shape[-1] < 10 or data.shape[-2] < 10:
            print(f"Warning: {file} has very small dimensions after cropping: {data.shape} Skipping.")
            return None
        data = rescale_to_224x224(data, is_reference)

        if not is_reference:
            if modality_name.lower() == 's2l2a':
                data[data < 0] = 0
                data = data * 10000
            elif modality_name.lower() == 'soil':
                data[data < 0] = 0

            if modality_name.upper() not in static_modalities:
                if fill_zeros:
                    data = fit_and_fill_zeros(data, method=interp_method)

        for tsave in tsave_list:
            dest_subfolder_path = dest_subfolder_paths[tsave]

            if not is_reference:
                data_to_save = data[:tsave]
            else:
                data_to_save = data

            if is_reference:
                save_path = os.path.join(dest_subfolder_path, os.path.splitext(os.path.basename(file))[0] + '.npy')
                # normalized_data = data_to_save / 370.0

                # Min-max normalization (crop-specific) -Anirudha 
                norm_max_corn = 370.0
                norm_max_soybean = 120.0
                data_min = 50.0 if 'corn' in file.lower() else 30.0
                data_max = norm_max_corn if 'corn' in file.lower() else norm_max_soybean
                normalized_data = (data_to_save - data_min) / (data_max - data_min)

                normalized_data[normalized_data > 1] = 1
                normalized_data[normalized_data < 0] = -1
                normalized_data = normalized_data.astype(np.float32)
                np.save(save_path, normalized_data)
            else:
                save_path = os.path.join(dest_subfolder_path, os.path.splitext(os.path.splitext(os.path.basename(file))[0])[0] + '.npy')
                data_to_save = data_to_save.astype(np.float32)
                np.save(save_path, data_to_save)

        return file

    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

def process_subfolder(subfolder_path, dest_subfolder_paths, ref_dims=None, is_reference=False, used_dates_dict=None, tsave_list=[24], num_workers=None, fill_zeros=False, interp_method='polynomial'):
    modality_name = os.path.basename(subfolder_path)
    if is_reference:
        files = sorted(glob.glob(os.path.join(subfolder_path, '*.tif')) + glob.glob(os.path.join(subfolder_path, '*.tiff')))
    else:
        files = sorted(glob.glob(os.path.join(subfolder_path, '*.tif.npy')) + glob.glob(os.path.join(subfolder_path, '*.npy')))

    for dest_path in dest_subfolder_paths.values():
        os.makedirs(dest_path, exist_ok=True)

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    process_func = partial(
        process_single_file,
        dest_subfolder_paths=dest_subfolder_paths,
        modality_name=modality_name,
        is_reference=is_reference,
        used_dates_dict=used_dates_dict,
        ref_dims=ref_dims,
        tsave_list=tsave_list,
        fill_zeros=fill_zeros,
        interp_method=interp_method
    )

    if len(files) > 0:
        print(f"Processing {len(files)} files using {num_workers} workers...")
        with Pool(num_workers) as pool:
            results = pool.map(process_func, files)
        processed = sum(1 for r in results if r is not None)
        print(f"Successfully processed {processed}/{len(files)} files")

def parse_used_dates_log(log_path):
    used_dates_dict = {}
    current_file = None
    current_modality = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.rstrip()

            if line.startswith('==='):
                continue

            if line and not line.startswith(' '):
                if '(' in line:
                    current_file = line.split('(')[0].strip()
                    used_dates_dict[current_file] = {}

            elif line.startswith('  ') and ':' in line and 'dates' in line:
                modality_line = line.strip()
                if ':' in modality_line:
                    current_modality = modality_line.split(':')[0].strip()

            elif line.startswith('    ') and current_file and current_modality:
                dates_str = line.strip()
                dates = [d.strip() for d in dates_str.split(',')
                         if d.strip() and d.strip().count('-') == 2]
                if dates:
                    used_dates_dict[current_file][current_modality] = dates
                current_modality = None

    return used_dates_dict

def process_root_folder(root_path, dest_root_path, used_dates_dict=None, tsave=24, all=True, num_workers=None, fill_zeros=False, interp_method='polynomial'):
    if all == True:
        all_iters = list(range(1, 25))
        print(f"Processing all 24 weeks once and saving versions: {all_iters}")

        dest_root_paths = {}
        for i in all_iters:
            dest_root_paths[i] = f'processed_data_weekly_{i}'
            os.makedirs(dest_root_paths[i], exist_ok=True)

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
            if os.path.isdir(subfolder_path):
                dest_subfolder_paths = {i: os.path.join(dest_root_paths[i], subfolder) for i in all_iters}

                if subfolder == 'yield_geotiffs':
                    print(f"Processing reference folder: {subfolder}")
                    process_subfolder(
                        subfolder_path,
                        dest_subfolder_paths,
                        is_reference=True,
                        used_dates_dict=used_dates_dict,
                        tsave_list=all_iters,
                        num_workers=num_workers,
                        fill_zeros=fill_zeros,
                        interp_method=interp_method
                    )
                else:
                    print(f"Processing folder: {subfolder}")
                    process_subfolder(
                        subfolder_path,
                        dest_subfolder_paths,
                        ref_dims=ref_dims,
                        used_dates_dict=used_dates_dict,
                        tsave_list=all_iters,
                        num_workers=num_workers,
                        fill_zeros=fill_zeros,
                        interp_method=interp_method
                    )
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

        dest_subfolder_paths = {tsave: None}
        for subfolder in sorted(os.listdir(root_path)):
            subfolder_path = os.path.join(root_path, subfolder)
            dest_subfolder_path = os.path.join(dest_root_path, subfolder)
            dest_subfolder_paths[tsave] = dest_subfolder_path

            if os.path.isdir(subfolder_path):
                if subfolder == 'yield_geotiffs':
                    print(f"Processing reference folder: {subfolder}")
                    process_subfolder(subfolder_path, dest_subfolder_paths, is_reference=True, used_dates_dict=used_dates_dict, tsave_list=[tsave], num_workers=num_workers, fill_zeros=fill_zeros, interp_method=interp_method)
                else:
                    print(f"Processing folder: {subfolder}")
                    process_subfolder(subfolder_path, dest_subfolder_paths, ref_dims=ref_dims, used_dates_dict=used_dates_dict, tsave_list=[tsave], num_workers=num_workers, fill_zeros=fill_zeros, interp_method=interp_method)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ts', type=int, default=24, help='Number of timepoints to save (default: 24)')
    parser.add_argument('--all', action='store_true', default=True, help='Process all timepoints (1-24)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: cpu_count-1)')
    parser.add_argument('--fill-zeros', action='store_true', default=False, help='Apply curve fitting to fill zero values in the data')
    parser.add_argument('--interp-method', type=str, default='polynomial', choices=['polynomial', 'spline', 'linear'],
                        help='Interpolation method for filling zeros (default: polynomial)')
    args = parser.parse_args()

    log_path = 'chloe_dataset/dates_log.txt'
    
    root_path = 'chloe_dataset'

    used_dates_dict = parse_used_dates_log(log_path)
    dst_folder = f'processed_data_weekly_{args.ts}'
    os.makedirs(dst_folder, exist_ok=True)

    if args.fill_zeros:
        print(f"Curve fitting enabled using {args.interp_method} interpolation")

    process_root_folder(
        root_path,
        dst_folder,
        used_dates_dict=used_dates_dict,
        tsave=args.ts,
        all=args.all,
        num_workers=args.workers,
        fill_zeros=args.fill_zeros,
        interp_method=args.interp_method
    )
