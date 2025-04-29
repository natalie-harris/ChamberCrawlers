# ---------------------------------------------------------------------------- #
#                                    OUTPUT                                    #
# ---------------------------------------------------------------------------- #
# Tomb-to-Tomb Distance Analysis
# Minimum distance between tombs: 2.67 meters
# Average distance between tombs: 308.60 meters
# Median distance between tombs: 225.74 meters
# Maximum distance between tombs: 1173.68 meters

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# Load your real tomb data
df = pd.read_csv("../../data/tombs_data.csv")

# Replace with your actual column names
EASTING_COL = "Easting (m)"  
NORTHING_COL = "Northing (m)"  

# Convert Easting and Northing to numeric (force errors to NaN)
df[EASTING_COL] = pd.to_numeric(df[EASTING_COL], errors='coerce')
df[NORTHING_COL] = pd.to_numeric(df[NORTHING_COL], errors='coerce')

# Drop rows where Easting or Northing is invalid (-1 or missing)
df = df[(df[EASTING_COL] > 0) & (df[NORTHING_COL] > 0)]


# Calculate pairwise distances
distances = []
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        east1, north1 = df.iloc[i][EASTING_COL], df.iloc[i][NORTHING_COL]
        east2, north2 = df.iloc[j][EASTING_COL], df.iloc[j][NORTHING_COL]
        distance = math.sqrt((east1 - east2)**2 + (north1 - north2)**2)
        distances.append(distance)

distances = np.array(distances)

# Print stats
print(f"Minimum distance between tombs: {np.min(distances):.2f} meters")
print(f"Average distance between tombs: {np.mean(distances):.2f} meters")
print(f"Median distance between tombs: {np.median(distances):.2f} meters")
print(f"Maximum distance between tombs: {np.max(distances):.2f} meters")

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(distances, bins=50, color="purple", edgecolor="black")
plt.title("Distribution of Tomb-to-Tomb Distances (meters)")
plt.xlabel("Distance (meters)")
plt.ylabel("Number of tomb pairs")
plt.grid(True)
plt.show()
