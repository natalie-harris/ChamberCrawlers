import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

# Define bins and bin labels
bins = ["0-25 m", "25-75 m", "75-150 m", "150-250 m", "250-400 m", "400-600 m", "600-800 m", "800-1000 m", "1000-1200 m"]
x = np.arange(len(bins))

# Define percentages from message.txt
data = {
    "Real Tombs":        [0.0186, 0.0904, 0.1977, 0.2580, 0.2048, 0.0895, 0.0399, 0.0736, 0.0275],
    "w=0.0":             [0.0028, 0.0861, 0.2361, 0.4528, 0.2111, 0.0111, 0.0, 0.0, 0.0],
    "w=0.25":            [0.0028, 0.0861, 0.2361, 0.4528, 0.2111, 0.0111, 0.0, 0.0, 0.0],
    "w=0.5":             [0.0028, 0.1,    0.2278, 0.4472, 0.2083, 0.0139, 0.0, 0.0, 0.0],
    "w=0.75":            [0.0028, 0.1,    0.2444, 0.4306, 0.2111, 0.0111, 0.0, 0.0, 0.0],
    "w=1.0":             [0.0028, 0.1,    0.2444, 0.4306, 0.2111, 0.0111, 0.0, 0.0, 0.0],
}

# Plotting setup
bar_width = 0.11
offsets = np.linspace(-0.33, 0.33, len(data))  # To center bars at each bin group
colors = ['black', '#92253f', '#e4663a', 'orange', '#ffe35e', '#fbff7c', '#9467bd']

fig, ax = plt.subplots(figsize=(16, 6))

for i, (label, values) in enumerate(data.items()):
    ax.bar(x + offsets[i], np.array(values) * 100, width=bar_width, label=label, color=colors[i])

# Labels and formatting
ax.set_xticks(x)
ax.set_xticklabels(bins, rotation=45)
ax.set_ylabel("Percentage of Tomb Pairs")
ax.set_title("Tomb-to-Tomb Distance Distributions: Real vs ABM Simulations")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()
