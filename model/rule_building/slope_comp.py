import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

# Define bins and bin labels
bins = ["0_2", "2_6", "6_10", ">10"]
x = np.arange(len(bins))

# Define percentages from message.txt
data = {
    "Real Tombs":        [0.33, 0.58, 0.084, 0.0],
    "w=0.0":             [0.1, 0.3, 0.6, 0.0],
    "w=0.25":            [0.0, 0.3, 0.7, 0.0],
    "w=0.5":             [0.0, 0.1, 0.8, 0.1],
    "w=0.75":            [0.0, 0.2, 0.8, 0.0],
    "w=1.0":             [0.0, 0.1, 0.9, 0.0],
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
ax.set_ylabel("Percentage of Tomb Slopes")
ax.set_xlabel("Slope, in Meters per 30 Meters Across")
ax.set_title("Tomb Distance Distributions: Real vs ABM Simulations")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()


plt.savefig('slope_comp.png')