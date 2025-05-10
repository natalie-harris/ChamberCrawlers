# CREATES ELEVATION HEATMAP using the elevation matrix (KVElevation.json)
import json
import numpy as np
import matplotlib.pyplot as plt

def display_elevation_heatmap(json_file="KVElevation.json", output_png="elevation_heatmap.png"):
    """
    Reads an elevation matrix from a JSON file and displays it as a heatmap
    using Matplotlib, saving the output to a PNG file.

    Args:
        json_file (str): The path to the JSON file containing the elevation matrix.
                         It is expected to have a key "elevation_matrix" whose value
                         is a list of lists representing the matrix.
        output_png (str): The filename to save the heatmap image to.
    """
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
            elevation_matrix = np.array(data.get("elevation_matrix"))
            if elevation_matrix is None:
                print("Error: 'elevation_matrix' key not found in JSON file.")
                return
            if elevation_matrix.ndim != 2:
                print("Error: 'elevation_matrix' does not contain a 2D array.")
                return
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file '{json_file}'.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(elevation_matrix, cmap='viridis', aspect='auto') # 'viridis' is a good default colormap
    plt.colorbar(label='Elevation (m)')
    plt.title('Elevation Heatmap of the Valley of the Kings')
    plt.xlabel('Longitude Index (approx. 30m per step)')
    plt.ylabel('Latitude Index (approx. 30m per step)')

    # Save the figure to a PNG file
    plt.savefig(output_png)
    print(f"Elevation heatmap saved to {output_png}")
    plt.close() # Close the figure to free up memory

if __name__ == "__main__":
    display_elevation_heatmap()