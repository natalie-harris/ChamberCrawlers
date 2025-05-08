import pandas as pd
import numpy as np
from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.time import RandomActivation
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import subprocess  # To run ffmpeg
from pyproj import Transformer  # For coordinate transformations
import math
import argparse
import random

# Path to the elevation data JSON file
ELEVATION_DATA_PATH = "../../data/data_acquisition/elevation/KVElevation.json"
TOMB_DATA_PATH = "../../data/tombs_data.csv"
OUTPUT_DIR = "output"
OUTPUT_FRAMES_DIR = os.path.join(OUTPUT_DIR, "simulation_frames")
FRAMES_PER_SECOND = 5
MAX_STEPS = 100  # Number of simulation steps to run
CELL_SIZE = 30  # meters per cell
MIN_LAT_ELEVATION = 25.73753
MAX_LAT_ELEVATION = 25.74315
MIN_LON_ELEVATION = 32.59838
MAX_LON_ELEVATION = 32.6047

# KV1 Latitude/Longitude and Northing/Easting
KV1_LAT = 25.74246687974823
KV1_LON = 32.601921510860855
KV1_NORTHING = 99803.743
KV1_EASTING = 94006.256

LAYERS = 2

# Rotation angle (clockwise)
rotation_degrees = 27 + (2 / 60) + (23 / 3600)
rotation_radians = math.radians(rotation_degrees)

# Target distribution of tomb distances
DIST_TARGET_DISTRIBUTION = {
    "0_25": 0.0186,
    "25_75": 0.0904,
    "75_150": 0.1977,
    "150_250": 0.2580,
    "250_400": 0.2048,
    "400_600": 0.0895,
    "600_800": 0.0399,
    "800_1000": 0.0736,
    "1000_1200": 0.0275
}

# Define agent starting location
AGENT_START_LOCATION = (13, 8)  # Example: row 20, column 20


def weighted_random_choice(weight_dict):
    total = sum(weight_dict.values())
    r = random.uniform(0, total)
    upto = 0
    for k, w in weight_dict.items():
        if upto + w >= r:
            return k
        upto += w


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
    def __init__(self, unique_id, model, start_location, weights, deterministic=True):
        super().__init__(model)
        self.pos = start_location
        self.steps_taken = 0
        self.max_steps = MAX_STEPS
        self.weights = weights
        self.unique_id = unique_id
        self.deterministic = deterministic

    def _get_elevation_preference(self, elevation):
        """Assigns a weight based on the provided elevation distribution."""
        if elevation < 160:
            return 0.00
        elif 160 <= elevation < 165:
            return 0.00
        elif 165 <= elevation < 170:
            return 0.1224
        elif 170 <= elevation < 175:
            return 0.1633
        elif 175 <= elevation < 180:
            return 0.2245
        elif 180 <= elevation < 190:
            return 0.2653
        elif 190 <= elevation < 200:
            return 0.1633
        elif 200 <= elevation < 220:
            return 0.408
        elif 220 <= elevation < 240:
            return 0.00
        elif 240 <= elevation < 260:
            return 0.204
        elif 260 <= elevation < 300:
            return 0.00
        else:
            return 0.00  # Default for elevations outside the given range

    def get_tomb_distance_bins(self, x, y):

        distances = []
        for tomb_x, tomb_y in self.model.tomb_locations:
            # Convert tile distance to meters
            distance = math.sqrt((x - tomb_x) ** 2 + (y - tomb_y) ** 2) * CELL_SIZE
            distances.append(distance)

        # Bin the distances (in meters)
        bins = {
            "0_25": 0,
            "25_75": 0,
            "75_150": 0,
            "150_250": 0,
            "250_400": 0,
            "400_600": 0,
            "600_800": 0,
            "800_1000": 0,
            "1000_1200": 0,
        }
        for distance in distances:
            if distance < 25:
                bins["0_25"] += 1
            elif 25 <= distance < 75:
                bins["25_75"] += 1
            elif 75 <= distance < 150:
                bins["75_150"] += 1
            elif 150 <= distance < 250:
                bins["150_250"] += 1
            elif 250 <= distance < 400:
                bins["250_400"] += 1
            elif 400 <= distance < 600:
                bins["400_600"] += 1
            elif 600 <= distance < 800:
                bins["600_800"] += 1
            elif 800 <= distance < 1000:
                bins["800_1000"] += 1
            elif 1000 <= distance < 1200:
                bins["1000_1200"] += 1

        # Convert counts to percentages
        total_tombs = len(distances)
        for key in bins:
            bins[key] /= total_tombs

        return bins


    def _get_tomb_distance_preference(self, x, y):
        """
        Calculates preference based on the percentage of surrounding tombs
        within certain distance bounds (in meters).

        Args:
            x (int): x-coordinate of the cell.
            y (int): y-coordinate of the cell.
            tomb_locations (list): List of (tomb_x, tomb_y) tuples.

        Returns:
            float: A value between 0 and 1 representing the preference.
        """
        bins = self.get_tomb_distance_bins(x, y)

        # Ideal percentages (from your provided data)
        ideal_bins = {
            "0_25": 0.0186,
            "25_75": 0.0904,
            "75_150": 0.1977,
            "150_250": 0.2580,
            "250_400": 0.2048,
            "400_600": 0.0895,
            "600_800": 0.0399,
            "800_1000": 0.0736,
            "1000_1200": 0.0275,
        }

        # Calculate errors
        errors = []
        for key in bins:
            errors.append(abs(bins[key] - ideal_bins[key]))

        # Sum of errors
        sum_of_errors = sum(errors)

        # Softmax function (to convert sum of errors to a value between 0 and 1)
        def softmax(x):
            e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
            return e_x / e_x.sum()

        # Return 1 - softmax(sum(errors)) as the preference
        preference = 1 - softmax(np.array([sum_of_errors]))[0] #convert to numpy array for softmax
        return preference
    
    def _get_agent_distance_preference(self, x, y):
        """
        Calculates preference based on the percentage of surrounding agents
        within certain distance bounds (in meters).
        Args:
            x (int): x-coordinate of the cell.
            y (int): y-coordinate of the cell.
        Returns:
            float: A value between 0 and 1 representing the preference.
        """
        distances = []
        for agent in self.model.schedule.agents:
            if agent != self:
                # Convert tile distance to meters using the global CELL_SIZE
                distance = math.sqrt((x - agent.pos[0]) ** 2 + (y - agent.pos[1]) ** 2) * CELL_SIZE
                distances.append(distance)

        # Bin the distances (in meters)
        bins = {
            "0_25": 0,
            "25_75": 0,
            "75_150": 0,
            "150_250": 0,
            "250_400": 0,
            "400_600": 0,
            "600_800": 0,
            "800_1000": 0,
            "1000_1200": 0,
        }
        for distance in distances:
            if distance < 25:
                bins["0_25"] += 1
            elif 25 <= distance < 75:
                bins["25_75"] += 1
            elif 75 <= distance < 150:
                bins["75_150"] += 1
            elif 150 <= distance < 250:
                bins["150_250"] += 1
            elif 250 <= distance < 400:
                bins["250_400"] += 1
            elif 400 <= distance < 600:
                bins["400_600"] += 1
            elif 600 <= distance < 800:
                bins["600_800"] += 1
            elif 800 <= distance < 1000:
                bins["800_1000"] += 1
            elif 1000 <= distance < 1200:
                bins["1000_1200"] += 1

        # Convert counts to percentages
        total_agents = len(distances)
        if total_agents == 0:
            return 0.5  # Return neutral preference if no other agents are around

        for key in bins:
            bins[key] /= total_agents

        # Ideal percentages
        ideal_bins = {
            "0_25": 0.0186,
            "25_75": 0.0904,
            "75_150": 0.1977,
            "150_250": 0.2580,
            "250_400": 0.2048,
            "400_600": 0.0895,
            "600_800": 0.0399,
            "800_1000": 0.0736,
            "1000_1200": 0.0275,
        }

        # Calculate errors
        errors = []
        for key in bins:
            errors.append(abs(bins[key] - ideal_bins[key]))

        # Sum of errors
        sum_of_errors = sum(errors)

        # Softmax function
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        # Return 1 - softmax(sum(errors)) as the preference
        preference = 1 - softmax(np.array([sum_of_errors]))[0]
        return preference
    
    def _get_slope_preference(self, x, y):
        """
        Calculates the slope of the terrain at the given coordinates using gradient descent.
        Prefers steeper slopes (magnitude of the gradient vector).
        """
        elevation = self.model.elevation
        width = self.model.grid.width
        height = self.model.grid.height

        # Get the elevation of the current cell
        center_elevation = elevation[y, x]

        # Calculate the partial derivatives (approximated using neighboring cells)
        dz_dx = 0
        dz_dy = 0
        num_neighbors = 0

        # Check neighbors and calculate partial derivatives
        if x > 0:
            dz_dx += center_elevation - elevation[y, x - 1]
            num_neighbors += 1
        if x < width - 1:
            dz_dx += elevation[y, x + 1] - center_elevation
            num_neighbors += 1
        if y > 0:
            dz_dy += center_elevation - elevation[y - 1, x]
            num_neighbors += 1
        if y < height - 1:
            dz_dy += elevation[y + 1, x] - center_elevation
            num_neighbors += 1

        if num_neighbors > 0:
            dz_dx /= num_neighbors
            dz_dy /= num_neighbors

        # Calculate the magnitude of the gradient vector
        slope_magnitude = math.sqrt(dz_dx ** 2 + dz_dy ** 2)

        # Normalize the slope magnitude to get a preference value between 0 and 1
        # You might need to adjust the normalization factor (e.g., 10, or the max slope)
        # depending on the range of your elevation data.
        max_expected_slope = 10  #  Adjust this based on your data's typical slope range
        preference = min(1.0, slope_magnitude / max_expected_slope)  # Clamp to 1
        return preference

    def _get_tile_preference(self, x, y):
        total_preference = 0

        # elevation preference
        tile_elevation = self.model.elevation[y, x]
        elevation_preference = self._get_elevation_preference(tile_elevation) * self.weights['elevation_weight']
        total_preference += elevation_preference

        # tomb distance preference
        tomb_distance_preference = self._get_tomb_distance_preference(x, y) * self.weights['tomb_distance_weight']
        total_preference += tomb_distance_preference

        # Bin the distances (in meters)
        bins = {
            "0_25": 0,
            "25_75": 0,
            "75_150": 0,
            "150_250": 0,
            "250_400": 0,
            "400_600": 0,
            "600_800": 0,
            "800_1000": 0,
            "1000_1200": 0,
        }

        # agent distance preference
        agent_distance_preference = self._get_agent_distance_preference(x, y) * self.weights['agent_distance_weight']
        total_preference += agent_distance_preference

        # slope preference
        slope_preference = self._get_slope_preference(x, y) * self.weights['slope_weight']
        total_preference += slope_preference

        return total_preference

    def step(self):
        if self.steps_taken < self.max_steps:
            x, y = self.pos
            best_move = None
            max_preference = -float('inf')
            best_adjacent_x = None
            best_adjacent_y = None

            for layer in range(1, LAYERS + 1):  # Iterate through layers
                # Get the squares in the current layer
                layer_squares = []
                for i in range(-layer, layer + 1):
                    for j in range(-layer, layer + 1):
                        if i == 0 and j == 0:
                            continue  # Skip the center cell
                        new_x, new_y = x + i, y + j
                        if 0 <= new_x < self.model.grid.width and 0 <= new_y < self.model.grid.height:
                            layer_squares.append((new_x, new_y))

                # Evaluate preferences for squares in the current layer
                for sx, sy in layer_squares:
                    if not self.model.grid.is_cell_empty((sx,sy)) and (sx,sy) != (x,y):
                        continue
                    
                    neighbor_preference = self._get_tile_preference(sx, sy)
                    if neighbor_preference > max_preference:
                        max_preference = neighbor_preference
                        # Determine the *adjacent* move towards the best square
                        dx = 0
                        if sx > x:
                            dx = 1
                        elif sx < x:
                            dx = -1
                        dy = 0
                        if sy > y:
                            dy = 1
                        elif sy < y:
                            dy = -1
                        
                        adjacent_x = x + dx
                        adjacent_y = y + dy
                        
                        if 0 <= adjacent_x < self.model.grid.width and 0 <= adjacent_y < self.model.grid.height:
                            best_adjacent_x = adjacent_x
                            best_adjacent_y = adjacent_y
                        else:
                            best_adjacent_x = None
                            best_adjacent_y = None

            if best_adjacent_x is not None and best_adjacent_y is not None:
                if self.model.grid.is_cell_empty((best_adjacent_x, best_adjacent_y)): # added this check
                    self.model.grid.remove_agent(self)
                    self.model.grid.move_agent(self, (best_adjacent_x, best_adjacent_y))

            self.steps_taken += 1

def create_distance_distribution(tombs):
    # Categorize all pairwise distances
    bins = {
        "0_25": 0,
        "25_75": 0,
        "75_150": 0,
        "150_250": 0,
        "250_400": 0,
        "400_600": 0,
        "600_800": 0,
        "800_1000": 0,
        "1000_1200": 0
    }
    total_pairs = 0

    for i in range(len(tombs)):
        for j in range(i + 1, len(tombs)):
            d = haversine(tombs[i]["lon"], tombs[i]["lat"], tombs[j]["lon"], tombs[j]["lat"])
            total_pairs += 1

            if d < 25:
                bins["0_25"] += 1
            elif d < 75:
                bins["25_75"] += 1
            elif d < 150:
                bins["75_150"] += 1
            elif d < 250:
                bins["150_250"] += 1
            elif d < 400:
                bins["250_400"] += 1
            elif d < 600:
                bins["400_600"] += 1
            elif d < 800:
                bins["600_800"] += 1
            elif d < 1000:
                bins["800_1000"] += 1
            elif d < 1200:
                bins["1000_1200"] += 1

    return bins, total_pairs

# Check if placement fits DISTANCE distribution
def fits_distribution(tombs):
    # Categorize all pairwise distances
    bins, total_pairs = create_distance_distribution(tombs)

    # Compare ratios to target
    if total_pairs == 0:
        return True  # No pairs yet, allow first tombs

    for key in bins:
        actual_ratio = bins[key] / total_pairs
        target_ratio = DIST_TARGET_DISTRIBUTION[key]

        # Allow some tolerance (+5%)
        if actual_ratio > (target_ratio + 0.05):
            return False

    return True

# Haversine Distance
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000  # radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


"""
MODEL
"""


class TerrainModel(Model):
    def __init__(self, width=None, height=None, num_agents=10, elevation_weight=0.5, tomb_distance_weight=0.5,
                 agent_distance_weight=0.5, slope_weight=0.5,  # Added slope_weight
                 elevation_data_path=ELEVATION_DATA_PATH, tomb_data_path=TOMB_DATA_PATH):
        super().__init__()
        self.elevation_data = load_elevation_data(elevation_data_path)
        self.tomb_df = load_tomb_data(tomb_data_path)
        self.tomb_locations = set()  # Store grid coordinates of tombs

        self.weights = {
            "elevation_weight": elevation_weight,
            "tomb_distance_weight": tomb_distance_weight,
            "agent_distance_weight": agent_distance_weight,
            "slope_weight": slope_weight,  # Include slope weight
        }

        if self.elevation_data is None:
            if width is None or height is None:
                self.width = 8  # Example dimensions if no elevation data
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
                print(
                    "Warning: Provided width and height do not match the dimensions of the loaded elevation data.")

        self.grid = SingleGrid(self.width, self.height, torus=False)
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

                rotated_easting_na, rotated_northing_na = rotate_coords(delta_easting_relative,
                                                                       delta_northing_relative, -rotation_radians)

                delta_lat = rotated_northing_na / lat_m_per_deg
                delta_lon = rotated_easting_na / lon_m_per_deg

                estimated_lat = kv1_lat + delta_lat
                estimated_lon = kv1_lon + delta_lon

                grid_row, grid_col = latlon_to_grid_coords(
                    estimated_lat, estimated_lon, MIN_LAT_ELEVATION, MAX_LAT_ELEVATION, MIN_LON_ELEVATION,
                    MAX_LON_ELEVATION, self.height, self.width
                )
                if grid_row is not None and grid_col is not None:
                    self.tomb_locations.add((grid_row, grid_col))

        for i in range(num_agents):
            # Use the constant starting location
            start_x, start_y = (AGENT_START_LOCATION[0] + i%2, AGENT_START_LOCATION[1] + i//2)
            # Ensure agents start within the grid bounds
            start_x = max(0, min(start_x, self.width - 1))
            start_y = max(0, min(start_y, self.height - 1))

            agent = WalkerAgent(unique_id=f"walker_{i}", model=self, start_location=(start_x, start_y),
                                weights=self.weights)
            self.schedule.add(agent)
            # self.grid.remove_agent(agent)
            self.grid.place_agent(agent, (start_x, start_y))

        self.running = True
        self.model_elevation = self.elevation

    def step(self):
        self.schedule.step()
        self.steps += 1
        self.check_for_agent_overlap()

    # Checks the grid for cells with more than one agent and prints a report.
    def check_for_agent_overlap(self):
        overlapping_cells = {}
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell_contents = self.grid.get_cell_list_contents((x, y))
                if len(cell_contents) > 1:
                    overlapping_cells[(x, y)] = [agent.unique_id for agent in cell_contents]

        if overlapping_cells:
            print(f"Step {self.steps}: Agent overlap detected in the following cells:")
            for coord, agent_ids in overlapping_cells.items():
                print(f"  Cell {coord}: Agents {agent_ids}")
        # else:
        #     print(f"Step {self.steps}: No agent overlap detected.") # Uncomment if you want a message at each step


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
    plt.scatter(tomb_x, tomb_y, color='white', marker='^', s=100, edgecolors='black', linewidths=0.5,
                label='Tomb Entrance')

    # Plot agent positions (adjust y-coordinate for flipped axis)
    agent_x = [pos[0] * cell_size + cell_size / 2 for pos in agent_positions.values()]
    agent_y = [(grid_height - 1 - pos[1]) * cell_size + cell_size / 2 for pos in agent_positions.values()]
    plt.scatter(agent_x, agent_y, color='red', s=50, edgecolors='black', linewidths=0.5, label='Builders')

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
    filename = os.path.join(output_dir, f"step_{step:03d}.png")
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":

    # still need to implement these vars in the program, too tired rn tho
    parser = argparse.ArgumentParser(description="Run a terrain analysis simulation.")
    parser.add_argument('--num_agents', type=int, default=10, help='Number of agents in the simulation.')
    parser.add_argument('--elevation_weight', type=float, default=0.5, help='Influence of elevation on agent movement.')
    parser.add_argument('--tomb_distance_weight', type=float, default=0.5, help='Influence of distance to tombs.')
    parser.add_argument('--agent_distance_weight', type=float, default=0.5, help='Influence of distance to other agents.')
    parser.add_argument('--slope_weight', type=float, default=0.5, help='Influence of slope of the tile relative to surrounding tiles.')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Where the simulation recording ends up.')

    args = parser.parse_args()  # Parse the arguments

    num_agents = args.num_agents
    elevation_weight = args.elevation_weight
    tomb_distance_weight = args.tomb_distance_weight
    agent_distance_weight = args.agent_distance_weight
    slope_weight = args.slope_weight
    output_path = args.output_dir

    # Load elevation data to get grid dimensions
    elevation_data = load_elevation_data(ELEVATION_DATA_PATH)
    if elevation_data is not None:
        grid_height, grid_width = elevation_data.shape
        model = TerrainModel(width=grid_width, height=grid_height, num_agents=num_agents, elevation_weight=elevation_weight, tomb_distance_weight=tomb_distance_weight, agent_distance_weight=agent_distance_weight, slope_weight=slope_weight)
    else:
        print("Elevation data not found!")
        exit()

    # Create output directory and frames directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

   # Run the simulation and generate frames
    last_agent_positions = {}
    for step in range(MAX_STEPS):
        print(f"Generating frame for step {step}")
        generate_frame(model, step, OUTPUT_FRAMES_DIR, MIN_LAT_ELEVATION, MAX_LAT_ELEVATION, MIN_LON_ELEVATION, MAX_LON_ELEVATION, CELL_SIZE)
        
        current_agent_positions = {agent.unique_id: agent.pos for agent in model.schedule.agents}
        if current_agent_positions == last_agent_positions:
            print(f"No agent moved for step {step}. Repeating frame.")
            for i in range(20):
                step += 1
                print(f"Generating repeated frame for step {step}")
                generate_frame(model, step, OUTPUT_FRAMES_DIR, MIN_LAT_ELEVATION, MAX_LAT_ELEVATION, MIN_LON_ELEVATION, MAX_LON_ELEVATION, CELL_SIZE)
            break  # Stop the simulation loop
        else:
            last_agent_positions = current_agent_positions
            model.step()

    # get elevation bins:
    elevation_bins = {
        "<160": 0.0,
        ">=160,<165": 0.0,
        ">=165,<170": 0.0,
        ">=170,<175": 0.0,
        ">=175,<180": 0.0,
        ">=180,<190": 0.0,
        ">=190,<200": 0.0,
        ">=200,<220": 0.0,
        ">=220,<240": 0.0,
        ">=240,<260": 0.0,
        ">=260,<300": 0.0,
        ">=300": 0.0,        
    }

    for agent in model.schedule.agents:
        x, y = agent.pos
        elevation = model.elevation[y, x]


        if elevation < 160:
            elevation_bins["<160"] += 1
        elif 160 <= elevation < 165:
            elevation_bins[">=160,<165"] += 1
        elif 165 <= elevation < 170:
            elevation_bins[">=165,<170"] += 1
        elif 170 <= elevation < 175:
            elevation_bins[">=170,<175"] += 1
        elif 175 <= elevation < 180:
            elevation_bins[">=175,<180"] += 1
        elif 180 <= elevation < 190:
            elevation_bins[">=180,<190"] += 1
        elif 190 <= elevation < 200:
            elevation_bins[">=190,<200"] += 1
        elif 200 <= elevation < 220:
            elevation_bins[">=200,<220"] += 1
        elif 220 <= elevation < 240:
            elevation_bins[">=220,<240"] += 1
        elif 240 <= elevation < 260:
            elevation_bins[">=240,<260"] += 1
        elif 260 <= elevation < 300:
            elevation_bins[">=260,<300"] += 1
        else:
            elevation_bins[">=300"] += 1

    # Convert counts to percentages
    total_elevations = len(model.schedule.agents)
    for key in elevation_bins:
        elevation_bins[key] /= total_elevations

    # print(f"Elevation weight: {elevation_weight}\n{elevation_bins}")

    # get tomb distance bins
    tomb_distance_bins = {
        "0_25": 0,
        "25_75": 0,
        "75_150": 0,
        "150_250": 0,
        "250_400": 0,
        "400_600": 0,
        "600_800": 0,
        "800_1000": 0,
        "1000_1200": 0,
    }

    for agent in model.schedule.agents:
        x, y = agent.pos
        distance_bins = agent.get_tomb_distance_bins(x, y)

        for key in distance_bins:
            tomb_distance_bins[key] += distance_bins[key]

    for key in tomb_distance_bins:
        tomb_distance_bins[key] /= len(model.schedule.agents)

    # print(f"Tomb distance weight: {tomb_distance_weight}\n{tomb_distance_bins}")
    

    # x, y = self.pos
    # tile_elevation = self.model.elevation[y, x]


    # Create the video using ffmpeg
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_file = os.path.join(output_path, f"simulation_{timestamp}.mp4")
    frame_pattern = os.path.join(OUTPUT_FRAMES_DIR, "step_%03d.png")
    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', str(FRAMES_PER_SECOND),
        '-i', frame_pattern,
        '-c:v', 'libx264',  # Using mpeg4 for now, you might need libx264
        '-pix_fmt', 'yuv420p',
        '-y',  # Overwrite output file if it exists
        output_video_file
    ]

    try:
        print(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"Simulation video saved to {output_video_file}")
        print(f"Video path: {os.path.abspath(output_video_file)}")
        # Clean up the frames directory (optional)
        import shutil
        shutil.rmtree(OUTPUT_FRAMES_DIR)
        print(f"Frames deleted from {OUTPUT_FRAMES_DIR}") #added print
    except subprocess.CalledProcessError as e:
        print(f"Error generating video: {e}")
        print(f"FFmpeg output: {e.stderr.decode()}")
    except FileNotFoundError:
        print("Error: ffmpeg command not found. Make sure it is installed and in your system's PATH.")