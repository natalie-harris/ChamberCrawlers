import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter  # For smoothing histogram

# 📥 Set your .tif file path
TIF_FILE = "../data/n25_e032_1arc_v3.tif"  # Update if needed

# 📂 Output folder
OUTPUT_DIR = "../plots"

def main():
    # Create output folder if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Open the GeoTIFF (no clipping)
    with rasterio.open(TIF_FILE) as dataset:
        elevation_array = dataset.read(1)  # Read full dataset
        transform = dataset.transform       # Full transform (no window)

        elevation_min = np.min(elevation_array)
        elevation_max = np.max(elevation_array)
        print(f"Elevation range: {elevation_min:.2f}m to {elevation_max:.2f}m")

        sample_lon, sample_lat = 32.610, 25.750
        try:
            row, col = dataset.index(sample_lon, sample_lat)
            elevation_value = elevation_array[row, col]
            print(f"Elevation at ({sample_lon}, {sample_lat}): {elevation_value:.2f} meters")
        except Exception:
            print(f"Sample coordinate ({sample_lon}, {sample_lat}) is outside the dataset bounds.")

        # === Plot Elevation Map with Real World Coordinates ===
        plt.figure(figsize=(12, 8))
        extent = [
            dataset.bounds.left, dataset.bounds.right,
            dataset.bounds.bottom, dataset.bounds.top
        ]
        img = plt.imshow(elevation_array, cmap='terrain', extent=extent, origin='upper')
        plt.colorbar(img, label="Elevation (meters)")
        plt.title("Elevation Map - Full Dataset")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(False)
        plt.savefig(os.path.join(OUTPUT_DIR, "elevation_map.png"))  # Save the plot
        plt.close()

        # === Plot Smoothed Elevation Histogram ===
        plt.figure(figsize=(10, 6))
        elevations = elevation_array.flatten()
        elevations = elevations[elevations > 0]  # Filter out invalid/no-data values
        hist, bins = np.histogram(elevations, bins=100)
        hist_smoothed = gaussian_filter(hist, sigma=2)  # Smooth

        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.plot(bin_centers, hist_smoothed, color="green")
        plt.title("Elevation Distribution")
        plt.xlabel("Elevation (meters)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "elevation_histogram.png"))  # Save the plot
        plt.close()

    print(f"Smoothed plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
