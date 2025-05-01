import pandas as pd
import numpy as np
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import subprocess  # To run ffmpeg
from pyproj import Transformer  # For coordinate transformations
import math

# Path to the elevation data JSON file
ELEVATION_DATA_PATH = "../../data/data_acquisition/elevation/KVElevation.json"
TOMB_DATA_PATH = "../../data/tombs_data.csv"
OUTPUT_DIR = "output"
OUTPUT_FRAMES_DIR = os.path.join(OUTPUT_DIR, "simulation_frames")
FRAMES_PER_SECOND = 5
TOTAL_STEPS = 100  # Number of simulation steps to run
CELL_SIZE = 30 # meters per cell
MIN_LAT_ELEVATION = 25.73753
MAX_LAT_ELEVATION = 25.74315
MIN_LON_ELEVATION = 32.59838
MAX_LON_ELEVATION = 32.6047

# KV1 Latitude/Longitude and Northing/Easting
KV1_LAT = 25.74246687974823
KV1_LON = 32.601921510860855
KV1_NORTHING = 99803.743
KV1_EASTING = 94006.256

# Rotation angle (clockwise)
rotation_degrees = 27 + (2/60) + (23/3600)
rotation_radians = math.radians(rotation_degrees)

def load_elevation_data(file_path):
    """Loads elevation data from a JSON file into a NumPy array."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            elevation_matrix = np.array(data.get("elevation_matrix"))
            if elevation_matrix is None:
                print(f"Error: 'elevation_matrix' key not found in '{file_path}'.")
                return None
            if elevation_matrix.ndim != 2:
                print(f"Error: 'elevation_matrix' in '{file_path}' is not a 2D array.")
                return None
            return elevation_matrix
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file '{file_path}'.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading elevation data: {e}")
        return None

def load_tomb_data(file_path):
    """Loads tomb data from a CSV file into a Pandas DataFrame."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Tomb data file '{file_path}' not found.")
        return None

def rotate_coords(easting, northing, angle_radians):
    """Rotates coordinates counter-clockwise by a given angle."""
    rotated_easting = easting * math.cos(angle_radians) - northing * math.sin(angle_radians)
    rotated_northing = easting * math.sin(angle_radians) + northing * math.cos(angle_radians)
    return rotated_easting, rotated_northing

def meters_per_degree(latitude):
    """Approximates meters per degree of latitude and longitude at a given latitude."""
    lat_rad = math.radians(latitude)
    lat_m_per_deg = 111132.954 - 559.822 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    lon_m_per_deg = (111320.747 - 372.936 * math.cos(2 * lat_rad) + 0.3975 * math.cos(4 * lat_rad)) * math.cos(lat_rad)
    return lat_m_per_deg, lon_m_per_deg

def latlon_to_grid_coords(latitude, longitude, min_lat, max_lat, min_lon, max_lon, grid_height, grid_width):
    """Converts latitude and longitude to grid coordinates (row, col)."""
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    if lat_range == 0 or lon_range == 0:
        return None, None

    row = int(((max_lat - latitude) / lat_range) * grid_height)
    col = int(((longitude - min_lon) / lon_range) * grid_width)

    # Ensure coordinates are within grid bounds
    if 0 <= row < grid_height and 0 <= col < grid_width:
        return row, col
    else:
        return None, None

"""
AGENT
"""
class WalkerAgent(Agent):
    def __init__(self, unique_id, model, start_location):
        super().__init__(model)
        self.pos = start_location
        self.steps_taken = 0
        self.max_steps = 50
        self.model.grid.place_agent(self, start_location)

    def step(self):
        if self.steps_taken < self.max_steps:
            x, y = self.pos
            curr_elev = self.model.elevation[y, x]
            neighbors = self.model.grid.get_neighborhood((x, y), moore=True, include_center=False)

            moves = [
                (nx, ny)
                for (nx, ny) in neighbors
                if 0 <= ny < self.model.grid.height
                and 0 <= nx < self.model.grid.width
                and self.model.elevation[ny, nx] > curr_elev
            ]

            if moves:
                new_location = self.random.choice(moves)
                self.model.grid.move_agent(self, new_location)

            self.steps_taken += 1

"""
MODEL
"""
class TerrainModel(Model):
    def __init__(self, width=None, height=None, num_agents=10, elevation_data_path=ELEVATION_DATA_PATH, tomb_data_path=TOMB_DATA_PATH):
        super().__init__()
        self.elevation_data = load_elevation_data(elevation_data_path)
        self.tomb_df = load_tomb_data(tomb_data_path)
        self.tomb_locations = set() # Store grid coordinates of tombs

        if self.elevation_data is None:
            if width is None or height is None:
                self.width = 8 # Example dimensions if no elevation data
                self.height = 6
                self.elevation = np.random.randint(0, 101, size=(self.height, self.width))
                print("Warning: Could not load elevation data. Using random elevation grid.")
            else:
                self.width = width
                self.height = height
                self.elevation = np.random.randint(0, 101, size=(self.height, self.width))
                print("Warning: Could not load elevation data. Using random elevation grid.")
        else:
            self.elevation = self.elevation_data
            self.height, self.width = self.elevation.shape
            if width is not None and height is not None and (self.width != width or self.height != height):
                print("Warning: Provided width and height do not match the dimensions of the loaded elevation data.")

        self.grid = MultiGrid(self.width, self.height, torus=False)
        self.schedule = RandomActivation(self)
        self.steps = 0

        # Convert tomb coordinates to grid locations
        if self.tomb_df is not None:
            kv1_easting = KV1_EASTING
            kv1_northing = KV1_NORTHING
            kv1_lat = KV1_LAT
            kv1_lon = KV1_LON

            lat_m_per_deg, lon_m_per_deg = meters_per_degree(kv1_lat)

            for index, row in self.tomb_df.iterrows():
                tomb_easting = row['Easting (m)']
                tomb_northing = row['Northing (m)']

                delta_easting_relative = tomb_easting - kv1_easting
                delta_northing_relative = tomb_northing - kv1_northing

                rotated_easting_na, rotated_northing_na = rotate_coords(delta_easting_relative, delta_northing_relative, -rotation_radians)

                delta_lat = rotated_northing_na / lat_m_per_deg
                delta_lon = rotated_easting_na / lon_m_per_deg

                estimated_lat = kv1_lat + delta_lat
                estimated_lon = kv1_lon + delta_lon

                grid_row, grid_col = latlon_to_grid_coords(
                    estimated_lat, estimated_lon, MIN_LAT_ELEVATION, MAX_LAT_ELEVATION, MIN_LON_ELEVATION, MAX_LON_ELEVATION, self.height, self.width
                )
                if grid_row is not None and grid_col is not None:
                    self.tomb_locations.add((grid_row, grid_col))

        for i in range(num_agents):
            # Ensure agents start within the grid bounds
            start_x = self.random.randrange(self.width)
            start_y = self.random.randrange(self.height)
            agent = WalkerAgent(f"walker_{i}", self, (start_x, start_y))
            self.schedule.add(agent)
            self.grid.place_agent(agent, (start_x, start_y))

        self.running = True

    def step(self):
        self.schedule.step()
        self.steps += 1

def generate_frame(model, step, output_dir, min_lat, max_lat, min_lon, max_lon, cell_size=30):
    """Generates a single frame of the simulation as a heatmap with tomb locations."""
    grid_width = model.grid.width
    grid_height = model.grid.height

    agent_positions = {agent.unique_id: agent.pos for agent in model.schedule.agents}

    elevation_data = np.array([[model.elevation[y, x] for x in range(grid_width)] for y in range(grid_height)])

    min_elev = np.min(elevation_data)
    max_elev = np.max(elevation_data)

    if min_elev != max_elev:
        normalized_elevation = (elevation_data - min_elev) / (max_elev - min_elev)
    else:
        normalized_elevation = np.full_like(elevation_data, 0.5)

    plt.figure(figsize=(8, 6))

    # Correct orientation: flip the y-axis and the elevation data
    extent = [0, grid_width * cell_size, 0, grid_height * cell_size]
    plt.imshow(np.flipud(model.elevation), cmap='viridis', origin='lower', extent=extent)
    plt.colorbar(label='Normalized Elevation', shrink=0.7)

    # Highlight tomb locations
    tomb_rows, tomb_cols = zip(*model.tomb_locations) if model.tomb_locations else ([], [])
    tomb_x = np.array(tomb_cols) * cell_size + cell_size / 2
    tomb_y = (grid_height - 1 - np.array(tomb_rows)) * cell_size + cell_size / 2
    plt.scatter(tomb_x, tomb_y, color='white', marker='^', s=100, edgecolors='black', linewidths=0.5, label='Tomb Entrance')

    # Plot agent positions (adjust y-coordinate for flipped axis)
    agent_x = [pos[0] * cell_size + cell_size / 2 for pos in agent_positions.values()]
    agent_y = [(grid_height - 1 - pos[1]) * cell_size + cell_size / 2 for pos in agent_positions.values()]
    plt.scatter(agent_x, agent_y, color='red', s=50, edgecolors='black', linewidths=0.5, label='Agents')

    # Set ticks based on meters from the bottom left
    x_ticks = np.arange(0, grid_width * cell_size, cell_size * 5)  # Every 5 cells (150 m)
    y_ticks = np.arange(0, grid_height * cell_size, cell_size * 5) # Every 5 cells (150 m)
    plt.xticks(x_ticks, [f'{int(t)}m' for t in x_ticks])
    plt.yticks(y_ticks, [f'{int(t)}m' for t in y_ticks])
    plt.xlabel('Meters (East)')
    plt.ylabel('Meters (North)')

    # Add lat/lon at the corners
    plt.text(extent[0], extent[2], f'Lat: {min_lat:.5f}\nLon: {min_lon:.5f}', ha='left', va='bottom', fontsize=8, bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3})
    plt.text(extent[1], extent[2], f'Lat: {min_lat:.5f}\nLon: {max_lon:.5f}', ha='right', va='bottom', fontsize=8, bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3})
    plt.text(extent[0], extent[3], f'Lat: {max_lat:.5f}\nLon: {min_lon:.5f}', ha='left', va='top', fontsize=8, bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3})
    plt.text(extent[1], extent[3], f'Lat: {max_lat:.5f}\nLon: {max_lon:.5f}', ha='right', va='top', fontsize=8, bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3})

    plt.title(f'Simulation Step: {step}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to make space for the legend

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"step_{step:04d}.png")
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Load elevation data to get grid dimensions
    elevation_data = load_elevation_data(ELEVATION_DATA_PATH)
    if elevation_data is not None:
        grid_height, grid_width = elevation_data.shape
        model = TerrainModel(width=grid_width, height=grid_height)
    else:
        model = TerrainModel() # Use default dimensions if no elevation data

    # Create output directory and frames directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

    # Run the simulation and generate frames
    for step in range(TOTAL_STEPS):
        print(f"Generating frame for step {step}")
        generate_frame(model, step, OUTPUT_FRAMES_DIR, MIN_LAT_ELEVATION, MAX_LAT_ELEVATION, MIN_LON_ELEVATION, MAX_LON_ELEVATION, CELL_SIZE)
        model.step()

    # Create the video using ffmpeg
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_file = os.path.join(OUTPUT_DIR, f"simulation_{timestamp}.mp4")
    frame_pattern = os.path.join(OUTPUT_FRAMES_DIR, "step_%04d.png")
    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', str(FRAMES_PER_SECOND),
        '-i', frame_pattern,
        '-c:v', 'mpeg4',  # Using mpeg4 for now, you might need libx264
        '-pix_fmt', 'yuv420p',
        '-y',  # Overwrite output file if it exists
        output_video_file
    ]

    try:
        print(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"Simulation video saved to {output_video_file}")
        # Clean up the frames directory (optional)
        # import shutil
        # shutil.rmtree(OUTPUT_FRAMES_DIR)

    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
    except FileNotFoundError:
        print("Error: ffmpeg command not found. Make sure it is installed and in your system's PATH.")