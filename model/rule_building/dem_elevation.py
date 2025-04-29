# Digital Elevation Model - elevation ranges
# Load DEM (n25_e032_1arc_v3.tif)
# Mask it to the Valley bounding box (32.590°E – 32.620°E, 25.735°N – 25.755°N)
# Bin all elevation values into the same ranges used for tombs (See elevation.py)
# Calculate the percentage of terrain that falls into each bin

import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the DEM
dem_path = "n25_e032_1arc_v3.tif"  # Adjust path if needed
raster = rasterio.open(dem_path)
elevation_array = raster.read(1)
transform = raster.transform

# Define Valley of the Kings bounding box (approximate)
lon_min, lon_max = 32.590, 32.620
lat_min, lat_max = 25.735, 25.755

# Mask: find rows/cols inside bounding box
rows, cols = raster.index([lon_min, lon_max], [lat_max, lat_min])
row_min, row_max = min(rows), max(rows)
col_min, col_max = min(cols), max(cols)

# Crop the elevation array
valley_elevation = elevation_array[row_min:row_max, col_min:col_max]

# Flatten and clean the data
valley_elevation = valley_elevation.flatten()
valley_elevation = valley_elevation[valley_elevation > 0]  # Remove no-data or negative values

# Define the same elevation bins as real tombs
bins = [0, 160, 165, 170, 175, 180, 190, 200, 220, 240, 260, 300]
labels = ["<160", "160–165", "165–170", "170–175", "175–180", "180–190", "190–200", "200–220", "220–240", "240–260", "260–300"]

# Bin the data
terrain_bins = pd.cut(valley_elevation, bins=bins, labels=labels, right=False)

# Calculate percentages
terrain_counts = terrain_bins.value_counts(normalize=True).sort_index() * 100

# Plot
plt.figure(figsize=(10, 6))
terrain_counts.plot(kind="bar", color="lightblue", edgecolor="black")
plt.title("Elevation Distribution of Valley Terrain")
plt.xlabel("Elevation Range (m)")
plt.ylabel("Percentage of Terrain Area")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

terrain_counts.round(2)
