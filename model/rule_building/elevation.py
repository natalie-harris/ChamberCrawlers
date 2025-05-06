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
# ---------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load tomb data
csv_path = "../../data/tombs_data.csv" 
df = pd.read_csv(csv_path)

# Clean elevation column
ELEVATION_COL = "Elevation_main (m)"
df[ELEVATION_COL] = pd.to_numeric(df[ELEVATION_COL], errors="coerce")
df = df[df[ELEVATION_COL] > 0]

# Define elevation bins and labels
bins = [0, 165, 175, 185, 195, 205, np.inf]
labels = ["<165", "165-174", "175-184", "185-194", "195-204", "205+"]
df["elev_bin"] = pd.cut(df[ELEVATION_COL], bins=bins, labels=labels, right=False)

# Calculate percentages
bin_counts = df["elev_bin"].value_counts(normalize=True).sort_index() * 100

# Print rounded results
print("\n# Elevation Distribution of Real Tombs (% by bin):\n")
for label, pct in zip(bin_counts.index, bin_counts.round(2)):
    print(f"{label}: {pct:.2f}%")

# Plot
plt.figure(figsize=(12, 7))  # Adjusted figure size for better readability
bars = plt.bar(bin_counts.index, bin_counts, color="blue", edgecolor="black")
plt.title("Elevation Distribution of Real Tombs")
plt.xlabel("Elevation Range (m)")
plt.ylabel("Percentage of Tombs")
plt.xticks(rotation=45, ha="right")  # Rotated x-axis labels for better fit
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Add percentage labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.2f}%",
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
