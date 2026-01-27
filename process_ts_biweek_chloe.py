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

def crop_to_largest_square(data, is_reference=False):
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


def force_static_T(data, target_T=12):
    """
    Accepts (T,C,H,W) or (C,H,W) and returns (target_T,C,H,W).
    For static modalities like DEM, SOIL, CDL.
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


def get_biweekly_indices(file, used_dates_dict, modality_name):
    """
    For a given file, return a list of lists, each containing indices of dates for each biweek.
    April 1 to September 30 = 6 months × 2 = 12 biweeks
    Biweek 0: Apr 1-15, Biweek 1: Apr 16-30, ..., Biweek 11: Sep 16-30
    """
    base = os.path.basename(file)
    base = base.split('.')[0] + '.tif'
    
    # Get modality-specific dates
    file_dates = used_dates_dict.get(base, {})
    dates = file_dates.get(modality_name.upper(), [])
    
    # Define 12 biweekly bins
    biweekly_indices = [[] for _ in range(12)]
    
    for idx, d in enumerate(dates):
        m = int(d[5:7])
        day = int(d[8:10])
        
        if 4 <= m <= 9:
            # biweek 0: Apr 1-15, 1: Apr 16-30, ..., 10: Sep 1-15, 11: Sep 16-30
            biweek = (m - 4) * 2
            if day > 15:
                biweek += 1
            biweekly_indices[biweek].append(idx)
    
    return biweekly_indices, dates


def aggregate_biweekly(data, biweekly_indices):
    """
    Given data of shape (T, C, H, W) and biweekly_indices (list of 12 lists of indices),
    return (12, C, H, W) where each timepoint is the average of that biweek's available data.
    If no data for a biweek, fill with interpolation from nearest biweeks.
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
    
    # Fill missing biweeks with interpolation
    for i in range(len(biweekly_indices)):
        if valid_means[i] is not None:
            biweekly_data.append(valid_means[i][None])
        else:
            # Find nearest previous valid biweek
            prev = None
            for j in range(i-1, -1, -1):
                if valid_means[j] is not None:
                    prev = valid_means[j]
                    break
            
            # Find nearest next valid biweek
            next_ = None
            for j in range(i+1, len(biweekly_indices)):
                if valid_means[j] is not None:
                    next_ = valid_means[j]
                    break
            
            # Interpolate
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


def fit_and_fill_zeros(data, method='polynomial', order=3):
    """
    Fit a curve through the temporal dimension and fill zeros in the data.
    """
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
    """
    Process a single file (aggregating to 12 biweeks) and save multiple versions based on tsave_list.
    """
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

            # ✅ Force static modalities to T=12
            if modality_name.upper() in static_modalities:
                data = force_static_T(data, target_T=12)

            # ✅ Aggregate only dynamic modalities
            if used_dates_dict is not None and modality_name.upper() not in static_modalities:
                biweekly_indices, dates = get_biweekly_indices(file, used_dates_dict, modality_name)
                data = aggregate_biweekly(data, biweekly_indices)
                if data is None:
                    print(f"Warning: {file} could not be aggregated. Skipping.")
                    ignore_list.append(file)
                    return None

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

        # ✅ Process non-reference data
        if not is_reference:
            if modality_name.lower() == 's2l2a':
                data[data < 0] = 0
                data = data * 10000  # ✅ Scale for TerraMind
            elif modality_name.lower() == 'soil':
                data[data < 0] = 0

            if modality_name.upper() not in static_modalities:
                if fill_zeros:
                    data = fit_and_fill_zeros(data, method=interp_method)

        # Save multiple versions based on tsave_list
        for tsave in tsave_list:
            dest_subfolder_path = dest_subfolder_paths[tsave]

            if not is_reference:
                data_to_save = data[:tsave]
            else:
                data_to_save = data

            if is_reference:
                save_path = os.path.join(dest_subfolder_path, os.path.splitext(os.path.basename(file))[0] + '.npy')
                
                # ✅ Crop-specific min-max normalization
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


def process_subfolder(subfolder_path, dest_subfolder_paths, ref_dims=None, is_reference=False, used_dates_dict=None, tsave_list=[12], num_workers=None, fill_zeros=False, interp_method='polynomial'):
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
    """
    Parse dates_log.txt and return a nested dict: {filename: {modality: [date1, date2, ...]}}
    """
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


def process_root_folder(root_path, dest_root_path, used_dates_dict=None, tsave=12, all=True, num_workers=None, fill_zeros=False, interp_method='polynomial'):
    if all == True:
        # Process all 12 biweeks once and save all versions
        all_iters = list(range(1, 13))
        print(f"Processing all 12 biweeks once and saving versions: {all_iters}")

        dest_root_paths = {}
        for i in all_iters:
            dest_root_paths[i] = f'processed_data_biweekly_{i}'
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
    parser = argparse.ArgumentParser(description='Process temporal data with biweekly aggregation')
    parser.add_argument('--ts', type=int, default=12, help='Number of timepoints to save (default: 12)')
    parser.add_argument('--all', action='store_true', default=True, help='Process all timepoints (1-12)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: cpu_count-1)')
    parser.add_argument('--fill-zeros', action='store_true', default=False, help='Apply curve fitting to fill zero values')
    parser.add_argument('--interp-method', type=str, default='polynomial', choices=['polynomial', 'spline', 'linear'],
                        help='Interpolation method for filling zeros (default: polynomial)')
    args = parser.parse_args()

    # ✅ Modify paths as needed
    log_path = '/work/mech-ai-scratch/bgekim/project/ISA_Yield_Anirudha/ISA_Yield/chloe_dataset/dates_log.txt'
    root_path = '/work/mech-ai-scratch/bgekim/project/ISA_Yield_Anirudha/ISA_Yield/chloe_dataset'

    used_dates_dict = parse_used_dates_log(log_path)
    dst_folder = f'processed_data/biweekly_12/processed_data_biweekly_{args.ts}'
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