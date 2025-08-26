import rasterio
from rasterio.warp import calculate_default_transform
import matplotlib.pyplot as plt

def resample_to_new_resolution(input_tif, output_tif, target_resolution=125):
    with rasterio.open(input_tif) as src:
        # Get the original CRS, transform, and bounds
        original_crs = src.crs
        original_transform = src.transform
        original_width = src.width
        original_height = src.height
        bounds = src.bounds

        print(f"Original resolution: {src.res}")
        print(f"Original width: {original_width}, height: {original_height}")

        # Calculate the new transform based on the target resolution
        new_transform, new_width, new_height = calculate_default_transform(
            original_crs, original_crs, original_width, original_height, *bounds, resolution=target_resolution
        )
        
        print(f"New transform: {new_transform}")
        print(f"New width: {new_width}, new height: {new_height}")

        # Update metadata to reflect the new transform, dimensions, and resolution
        out_meta = src.meta.copy()
        out_meta.update({
            "crs": original_crs,
            "transform": new_transform,
            "width": new_width,
            "height": new_height,
            "res": (target_resolution, target_resolution)  # New resolution
        })
        
        # Open the output file and resample the data to the new resolution
        with rasterio.open(output_tif, "w", **out_meta) as dst:
            for i in range(1, src.count + 1):  # Loop through all bands
                # Read the data for the current band
                band_data = src.read(i)
                
                # Resample the data to the new resolution
                resampled_data = band_data  # You may want to adjust the resampling here
                resampled_data = resampled_data  # Adjust based on the resampling method used (e.g., bilinear)

                # Write the resampled data to the new output file
                dst.write(resampled_data, i)

    print(f"Resampling complete. Output saved to {output_tif}")

def check_projection(input_tif, target_resolution=125):
    # Open original dataset
    with rasterio.open(input_tif) as src:
        # Print input bounds
        print(f"Input bounds for {input_tif}: {src.bounds}")

        # Define new CRS
        dst_crs = "EPSG:4326"  # WGS 84

        # Compute transform
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
        )

        # Print output transform and dimensions
        print(f"Output transform: {transform}")
        print(f"Output dimensions: width = {width}, height = {height}")

if __name__ == "__main__":
    # Specify the path to your input image (replace this with your actual file path)
    input_tif = "/work/mech-ai-scratch/rtali/gis-stac/IA_modis_NBAR/2019-04-07/Band1.tif"  # Example: "/path/to/your/image.tif"
    output_tif = "./resampled_image.tif"
    
    #Resample MODIS data from 250m to 125m resolution
    resample_to_new_resolution(input_tif, output_tif, target_resolution=125)
    # Check the projection for the single image
    check_projection(output_tif)
    
    #Plot the resampled image
    with rasterio.open(output_tif) as src:
        img = src.read(1)
        plt.imshow(img, cmap='jet')
        plt.title("Resampled Image")
        plt.savefig("./resampled_image.png")

