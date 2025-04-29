# ---------------------------------------------------------------------------- #
#                                    OUTPUT                                    #
# ---------------------------------------------------------------------------- #
# Elevation Distribution of Real Tombs (% by bin):
# <160: 0.00%
# 160-165: 0.00%
# 165–170: 12.24%
# 170–175: 16.33%
# 175–180: 22.45%
# 180–190: 26.53%
# 190-200: 16.33%
# 200–220: 4.08%
# 220–240: 0.00%
# 240–260: 2.04%
# 260–300: 0.00%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load tomb data
csv_path = "../../data/tombs_data.csv"  # <- Adjust path as needed
df = pd.read_csv(csv_path)

# Clean elevation column
ELEVATION_COL = "Elevation_main (m)"
df[ELEVATION_COL] = pd.to_numeric(df[ELEVATION_COL], errors="coerce")
df = df[df[ELEVATION_COL] > 0]

# Define elevation bins and labels
bins = [0, 160, 165, 170, 175, 180, 190, 200, 220, 240, 260, 300]
labels = ["<160", "160-165", "165–170", "170–175", "175–180", "180–190", "190-200", "200–220", "220–240", "240–260", "260–300"]
df["elev_bin"] = pd.cut(df[ELEVATION_COL], bins=bins, labels=labels, right=False)

# Calculate percentages
bin_counts = df["elev_bin"].value_counts(normalize=True).sort_index() * 100

# Print rounded results
print("\n# Elevation Distribution of Real Tombs (% by bin):\n")
for label, pct in zip(bin_counts.index, bin_counts.round(2)):
    print(f"{label}: {pct:.2f}%")

# Plot
plt.figure(figsize=(10, 6))
bin_counts.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Elevation Distribution of Real Tombs")
plt.xlabel("Elevation Range (m)")
plt.ylabel("Percentage of Tombs")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()
plt.show()
