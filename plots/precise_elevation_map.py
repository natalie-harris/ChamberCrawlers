# Making the elevation_map precise to the Valley of the Kings

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter  # For smoothing histogram
from rasterio.windows import from_bounds  # ── ADDED

# 📥 Set your .tif file path
TIF_FILE = "../data/n25_e032_1arc_v3.tif"  # Update if needed

# 📂 Output folder
OUTPUT_DIR = "../plots/tmp"

def main():
    # Create output folder if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define clipping extent (lon_min, lon_max, lat_min, lat_max)
    clip_extent = [
        32.59891841248648, 32.60763,
        25.742440894018127, 25.73764942257686
    ]  # ── ADDED

    # Open the GeoTIFF
    with rasterio.open(TIF_FILE) as dataset:
        # ── ADDED: compute window for clipping
        win = from_bounds(
            clip_extent[0],  # left (min lon)
            clip_extent[3],  # bottom (min lat)
            clip_extent[1],  # right (max lon)
            clip_extent[2],  # top (max lat)
            dataset.transform
        )
        elevation_array = dataset.read(1, window=win)  # ── MODIFIED: read only window
        transform = dataset.window_transform(win)      # ── MODIFIED: update transform

        elevation_min = np.min(elevation_array)
        elevation_max = np.max(elevation_array)
        print(f"Elevation range: {elevation_min:.2f}m to {elevation_max:.2f}m")

        sample_lon, sample_lat = 32.610, 25.750
        try:
            row, col = dataset.index(sample_lon, sample_lat, op=win)  # ── MODIFIED: respect window
            elevation_value = elevation_array[row, col]
            print(f"Elevation at ({sample_lon}, {sample_lat}): {elevation_value:.2f} meters")
        except Exception:
            print(f"Sample coordinate ({sample_lon}, {sample_lat}) is outside the clipped bounds.")

        # === Plot Elevation Map with Real World Coordinates ===
        plt.figure(figsize=(12, 8))
        extent = clip_extent  # ── MODIFIED: use the clipped extent
        img = plt.imshow(elevation_array, cmap='terrain', extent=extent, origin='upper')
        plt.colorbar(img, label="Elevation (meters)")
        plt.title("Elevation Map - Valley of the Kings")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(False)
        plt.savefig(f"{OUTPUT_DIR}/elevation_map.png", bbox_inches='tight')
        plt.close()

        # === Plot Smoothed Elevation Histogram ===
        plt.figure(figsize=(10, 6))
        elevations = elevation_array.flatten()
        elevations = elevations[elevations > 0]  # Filter out invalid or no-data values
        hist, bins = np.histogram(elevations, bins=100)
        hist_smoothed = gaussian_filter(hist, sigma=2)  # Smooth

        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.plot(bin_centers, hist_smoothed, color="green")
        plt.title("Elevation Distribution")
        plt.xlabel("Elevation (meters)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(f"{OUTPUT_DIR}/elevation_histogram.png", bbox_inches='tight')
        plt.close()

    print(f"✅ Smoothed plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
