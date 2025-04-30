from tomb_model import TombModel
import matplotlib.pyplot as plt

# Load your DEM GeoTIFF
TIF_FILE = "../data/n25_e032_1arc_v3.tif"

# Set number of 'tomb builders' and simulation steps
NUM_AGENTS = 20
NUM_STEPS = 100

def main():
    # Create model
    model = TombModel(num_agents=NUM_AGENTS, tif_file=TIF_FILE)

    # Run steps
    for i in range(NUM_STEPS):
        print(f"Step {i+1}/{NUM_STEPS}")
        model.step()

    # After simulation: plot tombs!
    plot_tombs(model)

def plot_tombs(model):
    # Plot terrain background
    extent = [
        model.raster.bounds.left, model.raster.bounds.right,
        model.raster.bounds.bottom, model.raster.bounds.top
    ]

    plt.figure(figsize=(12, 8))
    plt.imshow(model.elevation_array, cmap='terrain', extent=extent, origin='upper')
    plt.colorbar(label="Elevation (meters)")

    # Plot built tombs
    lons = [t["lon"] for t in model.tombs]
    lats = [t["lat"] for t in model.tombs]

    plt.scatter(lons, lats, color="red", edgecolor="black", s=60, label="Simulated Tombs")
    plt.title("Simulated Tomb Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    main()
