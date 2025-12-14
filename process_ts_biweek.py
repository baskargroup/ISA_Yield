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

def polynomial_curve(x, a, b, c, d):
    """3rd order polynomial for curve fitting"""
    return a * x**3 + b * x**2 + c * x + d

def fill_remaining_zeros_with_nearest(data):
    """
    Fill any remaining zeros in the data with the nearest non-zero value in the temporal dimension.
    
    Args:
        data: numpy array of shape (T, C, H, W) or (C, H, W)
    
    Returns:
        data_filled: numpy array with remaining zeros filled by nearest non-zero values
    """
    # Handle both (T, C, H, W) and (C, H, W) shapes
    if data.ndim == 4:
        T, C, H, W = data.shape
        has_time = True
    elif data.ndim == 3:
        C, H, W = data.shape
        has_time = False
    else:
        # Unsupported shape, return as-is
        return data
    
    data_filled = data.copy()
    
    if has_time:
        # Process each time step and channel separately, using spatial nearest neighbors
        for t in range(T):
            for c in range(C):
                channel_data = data_filled[t, c, :, :]
                zero_mask = channel_data == 0
                
                if np.any(zero_mask):
                    non_zero_mask = ~zero_mask
                    
                    if np.any(non_zero_mask):
                        # Get coordinates of zero and non-zero positions
                        zero_coords = np.argwhere(zero_mask)
                        non_zero_coords = np.argwhere(non_zero_mask)
                        
                        # For each zero position, find nearest non-zero position spatially
                        for zh, zw in zero_coords:
                            # Calculate Euclidean distances to all non-zero positions
                            distances = np.sqrt(np.sum((non_zero_coords - np.array([zh, zw]))**2, axis=1))
                            nearest_idx = np.argmin(distances)
                            nh, nw = non_zero_coords[nearest_idx]
                            data_filled[t, c, zh, zw] = data_filled[t, c, nh, nw]
    else:
        # For (C, H, W) shape, fill zeros with spatial nearest neighbor
        for c in range(C):
            channel_data = data_filled[c, :, :]
            zero_mask = channel_data == 0
            
            if np.any(zero_mask):
                non_zero_mask = ~zero_mask
                
                if np.any(non_zero_mask):
                    # Get coordinates of zero and non-zero positions
                    zero_coords = np.argwhere(zero_mask)
                    non_zero_coords = np.argwhere(non_zero_mask)
                    
                    # For each zero position, find nearest non-zero position
                    for zh, zw in zero_coords:
                        # Calculate distances to all non-zero positions
                        distances = np.sqrt(np.sum((non_zero_coords - np.array([zh, zw]))**2, axis=1))
                        nearest_idx = np.argmin(distances)
                        nh, nw = non_zero_coords[nearest_idx]
                        data_filled[c, zh, zw] = data_filled[c, nh, nw]
    
    return data_filled

def fit_and_fill_zeros(data, method='polynomial', order=3):
    """
    Fit a curve through the temporal dimension and fill zeros in the data.
    
    Args:
        data: numpy array of shape (T, C, H, W) where T=12 biweeks
        method: 'polynomial', 'spline', or 'linear' interpolation
        order: order of polynomial (default 3) or spline
    
    Returns:
        data_filled: numpy array of same shape with zeros filled by curve fitting
    """
    T, C, H, W = data.shape
    data_filled = data.copy()
    
    # Time indices for the 12 biweeks
    time_indices = np.arange(T)
    
    # Process each channel and spatial location
    for c in range(C):
        for h in range(H):
            for w in range(W):
                # Extract temporal profile for this pixel
                temporal_profile = data[:, c, h, w]
                
                # Find non-zero indices
                non_zero_mask = temporal_profile != 0
                non_zero_indices = time_indices[non_zero_mask]
                non_zero_values = temporal_profile[non_zero_mask]
                
                # Only interpolate if we have at least 2 non-zero points
                if len(non_zero_indices) >= 2:
                    zero_indices = time_indices[~non_zero_mask]
                    
                    if len(zero_indices) > 0:
                        try:
                            if method == 'polynomial':
                                # Use polynomial fitting (robust to outliers)
                                poly_order = min(order, len(non_zero_indices) - 1)
                                coeffs = np.polyfit(non_zero_indices, non_zero_values, poly_order)
                                poly = np.poly1d(coeffs)
                                filled_values = poly(zero_indices)
                                
                            elif method == 'spline':
                                # Use spline interpolation with extrapolation
                                if len(non_zero_indices) > order:
                                    interp_func = interp1d(non_zero_indices, non_zero_values, 
                                                          kind=order, fill_value='extrapolate')
                                else:
                                    interp_func = interp1d(non_zero_indices, non_zero_values, 
                                                          kind='linear', fill_value='extrapolate')
                                filled_values = interp_func(zero_indices)
                                
                            elif method == 'linear':
                                # Linear interpolation with extrapolation
                                interp_func = interp1d(non_zero_indices, non_zero_values, 
                                                      kind='linear', fill_value='extrapolate')
                                filled_values = interp_func(zero_indices)
                            
                            # Clip negative values to 0 (physical constraint)
                            filled_values = np.maximum(filled_values, 0)
                            
                            # Fill the zero positions
                            data_filled[zero_indices, c, h, w] = filled_values
                            
                        except Exception as e:
                            # If fitting fails, use linear interpolation as fallback
                            try:
                                interp_func = interp1d(non_zero_indices, non_zero_values, 
                                                      kind='linear', fill_value='extrapolate')
                                filled_values = interp_func(zero_indices)
                                filled_values = np.maximum(filled_values, 0)
                                data_filled[zero_indices, c, h, w] = filled_values
                            except:
                                # If all else fails, keep zeros
                                pass    
    return data_filled

def process_single_file(file, dest_subfolder_paths, modality_name, is_reference, used_dates_dict, ref_dims, tsave_list, fill_zeros=False, interp_method='polynomial'):
    """
    Process a single file once (aggregating all 12 biweeks) and save multiple versions based on tsave_list.
    This function is designed to be called in parallel.
    
    Args:
        dest_subfolder_paths: Dict mapping tsave -> dest_subfolder_path
        tsave_list: List of timepoint counts to save (e.g., [1, 2, 3, ..., 12])
        fill_zeros: If True, apply curve fitting to fill zero values in the data
        interp_method: Method for curve fitting ('polynomial', 'spline', or 'linear')
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
            if 's1rtc' in modality_name.lower():
                # Define a small positive epsilon value to avoid log(0) or log(-)
                EPSILON = 1e-5
                # Clip the array to ensure all values are at least EPSILON, then apply log10
                data = 10 * np.log10(np.clip(data, a_min=EPSILON, a_max=None))
            # Only aggregate if not static modality - always aggregate all 12 biweeks
            if used_dates_dict is not None and modality_name not in static_modalities:
                biweekly_indices, dates = get_biweekly_indices(file, used_dates_dict)
                data = aggregate_biweekly(data, biweekly_indices)  # Process all 12 biweeks
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
        
        # Process non-reference data (apply cleaning and interpolation once on full 12 biweeks)
        if not is_reference:
            if modality_name.lower() == 's2l2a':
                data[data < 0] = 0
            elif modality_name.lower() == 'soil':
                data[data < 0] = 0
            if modality_name.lower() not in static_modalities:
                # Apply curve fitting to fill zeros if requested (on full 12 biweeks)
                if fill_zeros:
                    data = fit_and_fill_zeros(data, method=interp_method)
            # # Fill any remaining zeros with nearest non-zero values
            # data = fill_remaining_zeros_with_nearest(data)
        
        # Save multiple versions based on tsave_list
        for tsave in tsave_list:
            dest_subfolder_path = dest_subfolder_paths[tsave]
            
            # Slice data to desired timepoints
            if not is_reference:
                data_to_save = data[:tsave]
            else:
                data_to_save = data
            
            # Save with .npy extension in destination subfolder
            if is_reference:
                save_path = os.path.join(dest_subfolder_path, os.path.splitext(os.path.basename(file))[0] + '.npy')
                # Min-max normalization
                data_min = 50.0 if 'corn' in file.lower() else 30.0
                data_max = norm_max_corn if 'corn' in file.lower() else norm_max_soybean
                normalized_data = (data_to_save - data_min) / (data_max - data_min)
                normalized_data[normalized_data < 0] = -1
                # Ensure float32 before saving
                normalized_data = normalized_data.astype(np.float32)
                np.save(save_path, normalized_data)
            else:
                save_path = os.path.join(dest_subfolder_path, os.path.splitext(os.path.splitext(os.path.basename(file))[0])[0] + '.npy')
                # Ensure float32 before saving
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
    
    # Create destination subfolders if they don't exist
    for dest_path in dest_subfolder_paths.values():
        os.makedirs(dest_path, exist_ok=True)
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    # Process all images with parallel processing
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

def process_root_folder(root_path, dest_root_path, used_dates_dict=None, tsave=12, all=True, num_workers=None, fill_zeros=False, interp_method='polynomial'):
    if all == True:
        # Process all 12 biweeks once and save all versions (1-12)
        all_iters = list(range(1, 13))
        print(f"Processing all 12 biweeks once and saving versions: {all_iters}")
        
        # Create destination paths for all iterations
        dest_root_paths = {}
        for i in all_iters:
            dest_root_paths[i] = f'processed_data_biweekly_{i}'
            os.makedirs(dest_root_paths[i], exist_ok=True)
        
        # Get reference dimensions
        yield_folder = os.path.join(root_path, 'yield_geotiffs')
        if not os.path.exists(yield_folder):
            print("yield_geotiffs subfolder not found!")
            return
        ref_files = sorted(glob.glob(os.path.join(yield_folder, '*.tif')) + glob.glob(os.path.join(yield_folder, '*.tiff')))
        ref_dims = {}
        for ref_file in ref_files:
            with rasterio.open(ref_file) as ds:
                ref_dims[os.path.splitext(os.path.basename(ref_file))[0] + '.npy'] = (ds.height, ds.width)
        
        # Process each subfolder once, saving all assigned iterations
        for subfolder in sorted(os.listdir(root_path)):
            subfolder_path = os.path.join(root_path, subfolder)
            if os.path.isdir(subfolder_path):
                # Create destination subfolder paths for all iterations
                dest_subfolder_paths = {i: os.path.join(dest_root_paths[i], subfolder) for i in all_iters}
                
                if subfolder == 'yield_geotiffs':
                    print(f"Processing reference folder: {subfolder}")
                    process_subfolder(subfolder_path, 
                                      dest_subfolder_paths, 
                                      is_reference=True, 
                                      used_dates_dict=used_dates_dict, 
                                      tsave_list=all_iters, 
                                      num_workers=num_workers,
                                      fill_zeros=fill_zeros,
                                      interp_method=interp_method)
                else:
                    print(f"Processing folder: {subfolder}")
                    # if 's1rtc' in subfolder.lower():
                    #     print(f"Processing s1rtc data in folder: {subfolder}")
                    #     print("Note: s1rtc data will be converted from linear to dB scale during processing.")
                    process_subfolder(subfolder_path, 
                                    dest_subfolder_paths, 
                                    ref_dims=ref_dims, 
                                    used_dates_dict=used_dates_dict, 
                                    tsave_list=all_iters, 
                                    num_workers=num_workers,
                                    fill_zeros=fill_zeros,
                                    interp_method=interp_method)
    else:
        # Single tsave mode
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
    parser.add_argument('--ts', type=int, default=12, help='Number of timepoints to save (default: 12)')
    parser.add_argument('--all', action='store_true', default=True, help='Process all timepoints (1-12)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: cpu_count-1)')
    parser.add_argument('--fill-zeros', action='store_true', default=False, help='Apply curve fitting to fill zero values in the data')
    parser.add_argument('--interp-method', type=str, default='polynomial', choices=['polynomial', 'spline', 'linear'], 
                        help='Interpolation method for filling zeros (default: polynomial)')
    args = parser.parse_args()

    log_path = './unprocessed_data/used_dates_log.txt'
    used_dates_dict = parse_used_dates_log(log_path)
    dst_folder = f'processed_data_biweekly_{args.ts}'
    os.makedirs(dst_folder, exist_ok=True)
    
    if args.fill_zeros:
        print(f"Curve fitting enabled using {args.interp_method} interpolation")
    
    process_root_folder('./unprocessed_data', dst_folder, used_dates_dict=used_dates_dict, 
                       tsave=args.ts, all=args.all, num_workers=args.workers,
                       fill_zeros=args.fill_zeros, interp_method=args.interp_method)