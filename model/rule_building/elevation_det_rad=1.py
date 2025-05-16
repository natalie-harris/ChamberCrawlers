import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

bins = ["<165 m", "165-169 m", "170-174 m", "175-179 m", "180-189 m", "190-199 m", "200-219 m", "220m+"]
x = np.arange(len(bins)) * 2

data = {
    "Real Tombs": [0.0, 0.1224, 0.1633, 0.2245, 0.2653, 0.1633, 0.0408, 0.0204],
    "w=0.0": [0.0, 0.1, 0.1, 0.1, 0.3, 0.2, 0.2, 0.0 ],
    "w=0.25": [0.0, 0.0, 0.2, 0.0, 0.4, 0.2, 0.2, 0.0],
    "w=0.5": [0.0, 0.0, 0.1, 0.0, 0.5, 0.0, 0.4, 0.0],
    "w=0.75": [0.0, 0.0, 0.0, 0.1, 0.2, 0.1, 0.6, 0.0],
    "w=1.0": [0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.7, 0.0],
}

bar_width = 0.18
offsets = np.linspace(-0.33, 0.33, len(data)) * 1.5  # To center bars at each bin group
colors = ['black', '#92253f', '#e4663a', 'orange', '#ffe35e', '#fbff7c']

fig, ax = plt.subplots(figsize=(20, 6))

for i, (label, values) in enumerate(data.items()):
    ax.bar(x + offsets[i], np.array(values) * 100, width=bar_width, label=label, color=colors[i])

ax.set_xticks(x)
ax.set_xticklabels(bins, rotation=45)
ax.set_ylabel("Percentage of Tombs")
ax.set_title("Elevation Distributions: Real vs ABM Simulations with Sight Radius of 30 Meters")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.savefig('elevation_det_radius=1_comp.png')