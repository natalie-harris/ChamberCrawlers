# ---------------------------------------------------------------------------- #
#                                    OUTPUT                                    #
# ---------------------------------------------------------------------------- #
# Tomb-to-Tomb Distance Analysis
# Minimum distance between tombs: 2.67 meters
# Average distance between tombs: 308.60 meters
# Median distance between tombs: 225.74 meters
# Maximum distance between tombs: 1173.68 meters

# Perecentage of tombs with x meters of each other
    # "0_25": 0.0186,
    # "25_75": 0.0904,
    # "75_150": 0.1977,
    # "150_250": 0.2580,
    # "250_400": 0.2048,
    # "400_600": 0.0895,
    # "600_800": 0.0399,
    # "800_1000": 0.0736,
    # "1000_1200": 0.0275

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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
total_pairs = len(distances)

# Print stats
print(f"Minimum distance between tombs: {np.min(distances):.2f} meters")
print(f"Average distance between tombs: {np.mean(distances):.2f} meters")
print(f"Median distance between tombs: {np.median(distances):.2f} meters")
print(f"Maximum distance between tombs: {np.max(distances):.2f} meters")

bins = np.linspace(0, 1200, 7)

counts, bin_edges, _ = plt.hist(
    distances,
    bins=bins,
    color='blue',
    edgecolor='black',
    weights=np.ones(len(distances)) / len(distances) * 100
)

for count, left, right in zip(counts, bin_edges[:-1], bin_edges[1:]):
    height = count
    center = (left + right) / 2
    if height > 0:
        plt.text(center, height + 0.3, f"{height:.1f}%", ha='center', va='bottom', fontsize=8)

plt.xlabel("Distance (meters)")
plt.ylabel("Percentage of Tomb Pairs")
plt.title("Distribution of Tomb-to-Tomb Distances (Percentage)")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(2))
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()