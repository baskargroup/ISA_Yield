import os
import glob
import rasterio
import numpy as np
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling

def get_aspect_ratio(ds):
    width = ds.width
    height = ds.height
    return max(width, height) / min(width, height)

def is_rectangular(ds, threshold=1.2):
    return get_aspect_ratio(ds) > threshold

def is_almost_square(ds, lower_threshold=1.0, upper_threshold=1.2):
    aspect = get_aspect_ratio(ds)
    return lower_threshold < aspect <= upper_threshold and ds.width != ds.height

def rescale_to_32x32(data, profile, crs):
    src_height, src_width = data.shape[1], data.shape[2]
    dst_height, dst_width = 32, 32
    dtype = data.dtype
    
    out_data = np.empty((data.shape[0], dst_height, dst_width), dtype=dtype)
    
    # Calculate new transform for 32x32
    scale_x = src_width / dst_width
    scale_y = src_height / dst_height
    new_transform = Affine(profile['transform'].a * scale_x, profile['transform'].b, profile['transform'].c,
                           profile['transform'].d, profile['transform'].e * scale_y, profile['transform'].f)
    
    reproject(
        source=data,
        destination=out_data,
        src_transform=profile['transform'],
        src_crs=crs,
        dst_transform=new_transform,
        dst_crs=crs,
        resampling=Resampling.nearest
    )
    
    profile.update({
        'width': dst_width,
        'height': dst_height,
        'transform': new_transform
    })
    
    return out_data, profile

def rescale_image(src, ref_width, ref_height, ref_transform, ref_crs):
    data = src.read()
    src_transform = src.transform
    src_crs = src.crs
    
    if len(set(src.dtypes)) > 1:
        raise ValueError(f"Image {src.name} has bands with different dtypes: {src.dtypes}")
    dtype = src.dtypes[0]
    
    out_shape = (src.count, ref_height, ref_width)
    out_data = np.empty(out_shape, dtype=dtype)
    
    scale_x = src.width / ref_width
    scale_y = src.height / ref_height
    new_transform = Affine(ref_transform.a * scale_x, ref_transform.b, ref_transform.c,
                           ref_transform.d, ref_transform.e * scale_y, ref_transform.f)
    
    reproject(
        source=data,
        destination=out_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=new_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest
    )
    
    profile = src.profile.copy()
    profile.update({
        'width': ref_width,
        'height': ref_height,
        'transform': new_transform,
        'crs': ref_crs
    })
    
    return out_data, profile

def crop_to_largest_square(ds, data, profile):
    height = ds.height
    width = ds.width
    side = min(height, width)
    
    if height == side and width == side:
        return data, profile
    
    transform = profile['transform']
    
    if height > width:
        start_row = (height - side) // 2
        data = data[:, start_row:start_row + side, :]
        new_transform = Affine(transform.a, transform.b, transform.c,
                               transform.d, transform.e, transform.f + start_row * transform.e)
        profile['height'] = side
        profile['transform'] = new_transform
    else:
        start_col = (width - side) // 2
        data = data[:, :, start_col:start_col + side]
        new_transform = Affine(transform.a, transform.b, transform.c + start_col * transform.a,
                               transform.d, transform.e, transform.f)
        profile['width'] = side
        profile['transform'] = new_transform
    
    return data, profile

def process_subfolder(subfolder_path, ref_dims=None, ref_crs=None, ref_transform=None, is_reference=False):
    tif_files = sorted(glob.glob(os.path.join(subfolder_path, '*.tif')) + glob.glob(os.path.join(subfolder_path, '*.tiff')))
    
    rectangular_files = []
    almost_square_files = []
    square_files = []
    for file in tif_files:
        with rasterio.open(file) as ds:
            if is_rectangular(ds, threshold=1.2):
                rectangular_files.append(file)
            elif is_almost_square(ds, lower_threshold=1.0, upper_threshold=1.2):
                almost_square_files.append(file)
            else:
                square_files.append(file)
    
    # Process rectangular images (concatenate with self)
    for file in rectangular_files:
        with rasterio.open(file) as ds:
            if not is_reference and ref_dims and ref_crs and ref_transform:
                ref_width, ref_height = ref_dims.get(os.path.basename(file), (ds.width, ds.height))
                data, profile = rescale_image(ds, ref_width, ref_height, ref_transform, ref_crs)
            else:
                data = ds.read()
                profile = ds.profile.copy()
            
            h = profile['height']
            w = profile['width']
            
            # Concatenate the same image
            if w > h:  # Wide, concat vertically
                new_h = 2 * h
                new_w = w
                new_data = np.empty((ds.count, new_h, new_w), dtype=ds.dtypes[0])
                new_data[:, 0:h, :] = data
                new_data[:, h:2*h, :] = data
                profile['height'] = new_h
            else:  # Tall, concat horizontally
                new_h = h
                new_w = 2 * w
                new_data = np.empty((ds.count, new_h, new_w), dtype=ds.dtypes[0])
                new_data[:, :, 0:w] = data
                new_data[:, :, w:2*w] = data
                profile['width'] = new_w
            
            # Crop to largest square
            cropped_data, cropped_profile = crop_to_largest_square(
                rasterio.io.MemoryFile().open(**profile), new_data, profile
            )
            
            # Rescale to 32x32
            final_data, final_profile = rescale_to_32x32(cropped_data, cropped_profile, ref_crs or ds.crs)
            
            # Save with same name
            with rasterio.open(file, 'w', **final_profile) as dst:
                dst.write(final_data)
    
    # Process almost square images (crop to perfect square)
    for file in almost_square_files:
        with rasterio.open(file) as ds:
            if not is_reference and ref_dims and ref_crs and ref_transform:
                ref_width, ref_height = ref_dims.get(os.path.basename(file), (ds.width, ds.height))
                data, profile = rescale_image(ds, ref_width, ref_height, ref_transform, ref_crs)
            else:
                data = ds.read()
                profile = ds.profile.copy()
            
            # Crop to largest square
            cropped_data, cropped_profile = crop_to_largest_square(
                rasterio.io.MemoryFile().open(**profile), data, profile
            )
            # Rescale to 32x32
            final_data, final_profile = rescale_to_32x32(cropped_data, cropped_profile, ref_crs or ds.crs)

            # Save with same name
            with rasterio.open(file, 'w', **final_profile) as dst:
                dst.write(final_data)
    
        # Process almost square images (crop to perfect square)
    for file in square_files:
        with rasterio.open(file) as ds:
            if not is_reference and ref_dims and ref_crs and ref_transform:
                ref_width, ref_height = ref_dims.get(os.path.basename(file), (ds.width, ds.height))
                data, profile = rescale_image(ds, ref_width, ref_height, ref_transform, ref_crs)
            else:
                data = ds.read()
                profile = ds.profile.copy()
            
            # Crop to largest square
            cropped_data, cropped_profile = crop_to_largest_square(
                rasterio.io.MemoryFile().open(**profile), data, profile
            )
            # Rescale to 32x32
            final_data, final_profile = rescale_to_32x32(cropped_data, cropped_profile, ref_crs or ds.crs)

            # Save with same name
            with rasterio.open(file, 'w', **final_profile) as dst:
                dst.write(final_data)

def process_root_folder(root_path):
    yield_folder = os.path.join(root_path, 'yield_geotiffs')
    if not os.path.exists(yield_folder):
        print("yield_geotiffs subfolder not found!")
        return
    
    ref_files = sorted(glob.glob(os.path.join(yield_folder, '*.tif')) + glob.glob(os.path.join(yield_folder, '*.tiff')))
    ref_dims = {}
    ref_crs = None
    ref_transform = None
    for ref_file in ref_files:
        with rasterio.open(ref_file) as ds:
            ref_dims[os.path.basename(ref_file)] = (ds.width, ds.height)
            ref_crs = ds.crs
            ref_transform = ds.transform
    
    for subfolder in sorted(os.listdir(root_path)):
        subfolder_path = os.path.join(root_path, subfolder)
        if os.path.isdir(subfolder_path):
            if subfolder == 'yield_geotiffs':
                print(f"Processing reference folder: {subfolder}")
                process_subfolder(subfolder_path, is_reference=True)
            else:
                print(f"Processing folder: {subfolder}")
                process_subfolder(subfolder_path, ref_dims, ref_crs, ref_transform)

# Usage
process_root_folder('./modalities2023')