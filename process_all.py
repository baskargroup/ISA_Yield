import os
import glob
import numpy as np
import rasterio
import pdb
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

def process_subfolder(subfolder_path, dest_subfolder_path, ref_dims=None, is_reference=False):
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
                data = np.expand_dims(data, 0)  # Make (1, C, H, W)
                T, C, H, W = data.shape
        else:
            data = np.load(file)
            T, C, H, W = data.shape
            if T < 8:
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
            T, C, H, W = data.shape
            if T > 8:
                indices = np.round(np.linspace(0, T - 1, 8)).astype(int)
                data = data[indices]
        if not is_reference and ref_dims:
            ref_h, ref_w = ref_dims.get(os.path.basename(file).replace('.tif.npy', '.npy'), (H, W))
            data = rescale_data(data, ref_h, ref_w)
            H, W = ref_h, ref_w
       # Concatenate the same image
        if is_reference:
            if W > H:  # Wide, concat vertically
                new_h = 2 * H
                new_w = W
                new_data = np.empty((C, new_h, new_w), dtype=data.dtype)
                new_data[:, 0:H, :] = data
                new_data[:, H:2*H, :] = data
            else:  # Tall, concat horizontally
                new_h = H
                new_w = 2 * W
                new_data = np.empty((C, new_h, new_w), dtype=data.dtype)
                new_data[:, :, 0:W] = data
                new_data[:, :, W:2*W] = data
        else:
            if W > H:  # Wide, concat vertically
                new_h = 2 * H
                new_w = W
                new_data = np.empty((data.shape[0], data.shape[1], new_h, new_w), dtype=data.dtype)
                new_data[:, :, 0:H, :] = data
                new_data[:, :, H:2*H, :] = data
            else:  # Tall, concat horizontally
                new_h = H
                new_w = 2 * W
                new_data = np.empty((data.shape[0], data.shape[1], new_h, new_w), dtype=data.dtype)
                new_data[:, :, :, 0:W] = data
                new_data[:, :, :, W:2*W] = data

        data = new_data
        # Crop to largest square
        data = crop_to_largest_square(data, is_reference)
        # Rescale to 224x224
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
        else:
            data = np.load(file)
            T, C, H, W = data.shape
            if T > 8:
                indices = np.round(np.linspace(0, T - 1, 8)).astype(int)
                data = data[indices]
        if not is_reference and ref_dims:
            ref_h, ref_w = ref_dims.get(os.path.basename(file).replace('.tif.npy', '.npy'), (H, W))
            data = rescale_data(data, ref_h, ref_w)
            H, W = ref_h, ref_w
        # Crop to largest square
        data = crop_to_largest_square(data, is_reference)
        # Rescale to 32x32
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
        else:
            data = np.load(file)
            T, C, H, W = data.shape
            if T > 8:
                indices = np.round(np.linspace(0, T - 1, 8)).astype(int)
                data = data[indices]
        if not is_reference and ref_dims:
            ref_h, ref_w = ref_dims.get(os.path.basename(file).replace('.tif.npy', '.npy'), (H, W))
            data = rescale_data(data, ref_h, ref_w)
            H, W = ref_h, ref_w
        # Crop to largest square (noop if perfect square)
        data = crop_to_largest_square(data, is_reference)
        # Rescale to 32x32
        data = rescale_to_224x224(data, is_reference)
        # Save with .npy extension in destination subfolder
        if is_reference:
            save_path = os.path.join(dest_subfolder_path, os.path.splitext(os.path.basename(file))[0] + '.npy')
        else:
            save_path = os.path.join(dest_subfolder_path, os.path.splitext(os.path.splitext(os.path.basename(file))[0])[0] + '.npy')
        np.save(save_path, data)

def process_root_folder(root_path, dest_root_path):
    yield_folder = os.path.join(root_path, 'yield_geotiffs')
    dest_yield_folder = os.path.join(dest_root_path, 'yield_geotiffs')
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
                process_subfolder(subfolder_path, dest_subfolder_path, is_reference=True)
            # else:
            #     print(f"Processing folder: {subfolder}")
            #     process_subfolder(subfolder_path, dest_subfolder_path, ref_dims=ref_dims)

# Usage
process_root_folder('./unprocessed_data', './processed_data')