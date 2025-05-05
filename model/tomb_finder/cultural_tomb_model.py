#region SECTION 1: IMPORTS
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
import random
from collections import defaultdict
#endregion SECTION 1 END 
 
#region SECTION 2: CONSTANTS
KING_TOMBS = 29 # 24 kings + 4 unknown "probably royal"
QUEEN_TOMBS = 5 # 3 queens + 2 unknown "probably royal"
OTHER_TOMBS = 30 # 30 non-royal tombs

# Path to the elevation data JSON file
ELEVATION_DATA_PATH = "../../data/data_acquisition/elevation/KVElevation.json"
TOMB_DATA_PATH = "../../data/tombs_data.csv"
OUTPUT_DIR = "cultural_output"
OUTPUT_FRAMES_DIR = os.path.join(OUTPUT_DIR, "simulation_frames")
FRAMES_PER_SECOND = 5
TOTAL_STEPS = 50  # Number of simulation steps to run
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
#endregion SECTION 2 END 

#region SECTION 3: HELPER FUNCTIONS
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

# Check if placement fits DISTANCE distribution
def fits_distribution(tombs):
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
        for j in range(i+1, len(tombs)):
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

    import pandas as pd
from collections import defaultdict

def calculate_conditional_owner_probs(tomb_df):
    df = tomb_df.copy()

    # Step 1: Bin features
    elev_bins = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
    df["ElevationBin"] = pd.cut(df["Elevation_main (m)"], bins=elev_bins)
    df["EntranceBin"] = df["Entrance Location"]
    df["AreaBin"] = pd.qcut(df["Area (m²)"], q=3, labels=["Small", "Medium", "Large"])

    # Step 2: Drop missing
    df = df.dropna(subset=["ElevationBin", "EntranceBin", "AreaBin", "Owner Type"])

    # Step 3: Count + normalize
    grouped = df.groupby(["ElevationBin", "EntranceBin", "AreaBin", "Owner Type"]).size()
    cond_probs = grouped / grouped.groupby(level=[0, 1, 2]).sum()
    cond_probs = cond_probs.reset_index()
    cond_probs.columns = ["ElevationBin", "EntranceBin", "AreaBin", "OwnerType", "Probability"]

    # Step 4: Convert to nested dict
    lookup = defaultdict(dict)
    for _, row in cond_probs.iterrows():
        key = (str(row["ElevationBin"]), row["EntranceBin"], row["AreaBin"])
        lookup[key][row["OwnerType"]] = row["Probability"]

    return dict(lookup)


# Haversine Distance
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000  # radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi/2)**2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


#endregion SECTION 3 END

#region SECTION 4: AGENT CLASS
class WalkerAgent(Agent):
    def __init__(self, unique_id, model, start_location, target_tomb_type=None):
        super().__init__(model)
        self.pos = start_location
        self.steps_taken = 0
        self.max_steps = 50
        self.model = model
        self.target_tomb_type = target_tomb_type

    def _get_elevation_preference_weight(self, elevation):
        """Assigns a weight based on the provided elevation distribution."""
        if 165 <= elevation < 170:
            return 12.24
        elif 170 <= elevation < 175:
            return 16.33
        elif 175 <= elevation < 180:
            return 22.45
        elif 180 <= elevation < 190:
            return 26.53
        elif 190 <= elevation < 200:
            return 16.33
        elif 200 <= elevation < 220:
            return 4.08
        elif 220 <= elevation < 240:
            return 0.00
        elif 240 <= elevation < 260:
            return 2.04
        else:
            return 0.00  # Default for elevations outside the given range

    def step(self):
        if self.steps_taken < self.max_steps:
            x, y = self.pos
            neighbors = self.model.grid.get_neighborhood((x, y), moore=True, include_center=False)
            possible_moves = []
            weights = []

            for nx, ny in neighbors:
                if 0 <= ny < self.model.grid.height and 0 <= nx < self.model.grid.width:
                    neighbor_elevation = self.model.elevation[ny, nx]

                    # 🪜 Step 1: Bin the elevation
                    if neighbor_elevation < 170:
                        elev_bin = "(160, 170]"
                    elif neighbor_elevation < 180:
                        elev_bin = "(170, 180]"
                    elif neighbor_elevation < 190:
                        elev_bin = "(180, 190]"
                    elif neighbor_elevation < 200:
                        elev_bin = "(190, 200]"
                    elif neighbor_elevation < 210:
                        elev_bin = "(200, 210]"
                    elif neighbor_elevation < 220:
                        elev_bin = "(210, 220]"
                    elif neighbor_elevation < 230:
                        elev_bin = "(220, 230]"
                    elif neighbor_elevation < 240:
                        elev_bin = "(230, 240]"
                    else:
                        elev_bin = "(240, 250]"

                    # 🪜 Step 2: Make assumptions for entrance + area bin
                    entrance_bin = "end of spur"  # You can refine this later
                    area_bin = "Medium"           # Or estimate based on terrain, etc.

                    # 🧠 Step 3: Look up conditional probability
                    key = (elev_bin, entrance_bin, area_bin)
                    tomb_type_prob = self.model.conditional_owner_probs.get(key, {}).get(self.target_tomb_type, 0)

                    # 🧮 Step 4: Combine elevation preference and type probability
                    elev_weight = self._get_elevation_preference_weight(neighbor_elevation)
                    total_weight = elev_weight * (1 + 3 * tomb_type_prob)

                    possible_moves.append((nx, ny))
                    weights.append(total_weight)


            # Normalize weights to create a probability distribution
            total_weight = sum(weights)
            if total_weight > 0:
                probabilities = [w / total_weight for w in weights]
                new_location = self.random.choices(possible_moves, weights=probabilities, k=1)[0]
                self.model.grid.move_agent(self, new_location)

            self.steps_taken += 1

    def _get_tomb_type_preference_weight(self, grid_coords, local_probabilities):
        """Assigns a weight based on the probability of the target tomb type in the area."""
        if self.target_tomb_type is None:
            return 1.0  # No preference if no target type is set

        if grid_coords in self.model.tomb_locations_attributes:
            attributes = self.model.tomb_locations_attributes[grid_coords]
            owner_type = attributes.get('Owner Type')
            if owner_type == self.target_tomb_type:
                return 2.0  # Higher weight if the tomb in the cell matches the target
            else:
                return 1.0
        else:
            # Weight based on the local probability of the target tomb type in the neighboring cell's bin
            neighbor_bin = self.model.grid_bins.get(grid_coords)
            if neighbor_bin and self.model.tomb_type_bin_probabilities.get(neighbor_bin):
                probability_of_target = self.model.tomb_type_bin_probabilities[neighbor_bin].get(self.target_tomb_type, 0)
                # Weight can be proportional to the probability
                return 1.0 + probability_of_target * 3 # Adjust the multiplier as needed
            else:
                return 1.0

#endregion SECTION 4 END

#region SECTION 5: MODEL CLASS
"""
MODEL
"""
class TerrainModel(Model):
    def __init__(self, width=None, height=None, num_king_seekers=KING_TOMBS, num_queen_seekers=QUEEN_TOMBS, num_other_seekers=OTHER_TOMBS, elevation_data_path=ELEVATION_DATA_PATH, tomb_data_path=TOMB_DATA_PATH):
        super().__init__()
        self.elevation_data = load_elevation_data(elevation_data_path)
        self.tomb_df = load_tomb_data(tomb_data_path)
        if self.tomb_df is not None:
            self.conditional_owner_probs = calculate_conditional_owner_probs(self.tomb_df)
        self.tomb_locations = set()
        self.tomb_locations_attributes = {}
        self.tomb_type_frequencies = {}
        self.tomb_type_bin_probabilities = {}
        self.grid_bins = {}
        self.steps = 0

        if self.elevation_data is None:
            self.width = width if width is not None else 8
            self.height = height if height is not None else 6
            self.elevation = np.random.randint(0, 101, size=(self.height, self.width))
            print("Warning: Could not load elevation data. Using random elevation grid.")
        else:
            self.elevation = self.elevation_data
            self.height, self.width = self.elevation.shape
            if width is not None and height is not None and (self.width != width or self.height != height):
                print("Warning: Provided width and height do not match the dimensions of the loaded elevation data.")

        self.grid = MultiGrid(self.width, self.height, torus=False)
        self.schedule = RandomActivation(self)

        # Analyze tomb data and calculate probabilities
        if self.tomb_df is not None:
            self._analyze_tomb_data()

        # Create specialized agents
        for i in range(num_king_seekers):
            start_x = self.random.randrange(self.width)
            start_y = self.random.randrange(self.height)
            agent = WalkerAgent(f"king_seeker_{i}", self, (start_x, start_y), target_tomb_type='King')
            self.schedule.add(agent)
            if agent.pos is not None:
                x, y = agent.pos
                if agent in self.grid[x][y]:
                    self.grid.remove_agent(agent)
            self.grid.place_agent(agent, (start_x, start_y))

        for i in range(num_queen_seekers):
            start_x = self.random.randrange(self.width)
            start_y = self.random.randrange(self.height)
            agent = WalkerAgent(f"queen_seeker_{i}", self, (start_x, start_y), target_tomb_type='Queen')
            self.schedule.add(agent)
            # Only remove the agent if they're already placed AND in that cell
            if agent.pos is not None:
                x, y = agent.pos
                if agent in self.grid[x][y]:
                    self.grid.remove_agent(agent)
            self.grid.place_agent(agent, (start_x, start_y))

        # Add a few default agents (optional)
        for i in range(num_other_seekers):
            start_x = self.random.randrange(self.width)
            start_y = self.random.randrange(self.height)
            agent = WalkerAgent(f"walker_{i}", self, (start_x, start_y))
            self.schedule.add(agent)
            if agent.pos is not None:
                x, y = agent.pos
                if agent in self.grid[x][y]:
                    self.grid.remove_agent(agent)
            self.grid.place_agent(agent, (start_x, start_y))

        # Initialize the grid with tomb locations
        self.running = True

    def _analyze_tomb_data(self):
        """Analyzes the tomb data to calculate frequencies and bin probabilities."""
        self.tomb_type_frequencies = self.tomb_df['Owner Type'].value_counts(normalize=True).to_dict()
        kv1_easting = KV1_EASTING
        kv1_northing = KV1_NORTHING
        kv1_lat = KV1_LAT
        kv1_lon = KV1_LON
        lat_m_per_deg, lon_m_per_deg = meters_per_degree(kv1_lat)

        for index, row in self.tomb_df.iterrows():
            tomb_easting = row['Easting (m)']
            tomb_northing = row['Northing (m)']
            owner_type = row['Owner Type']

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
                self.tomb_locations.add((grid_col, grid_row))
                self.tomb_locations_attributes[(grid_col, grid_row)] = {'Owner Type': owner_type}

        bin_width = 10
        bin_height = 10
        num_bins_x = self.width // bin_width
        num_bins_y = self.height // bin_height
        for x in range(self.width):
            for y in range(self.height):
                bin_x = x // bin_width
                bin_y = y // bin_height
                bin_id = (bin_x, bin_y)
                self.grid_bins[(x, y)] = bin_id
        self.tomb_type_bin_probabilities = self._calculate_tomb_type_bin_probabilities()

    def _calculate_tomb_type_bin_probabilities(self):
        """Calculates the probability distribution of tomb types within each bin."""
        bin_tomb_counts = {}
        for (x, y), bin_id in self.grid_bins.items():
            if (x, y) in self.tomb_locations_attributes:
                owner_type = self.tomb_locations_attributes[(x, y)]['Owner Type']
                if bin_id not in bin_tomb_counts:
                    bin_tomb_counts[bin_id] = {}
                if owner_type not in bin_tomb_counts[bin_id]:
                    bin_tomb_counts[bin_id][owner_type] = 0
                bin_tomb_counts[bin_id][owner_type] += 1

        bin_probabilities = {}
        for bin_id, type_counts in bin_tomb_counts.items():
            total_tombs_in_bin = sum(type_counts.values())
            bin_probabilities[bin_id] = {
                owner_type: count / total_tombs_in_bin for owner_type, count in type_counts.items()
            }
        return bin_probabilities

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

#endregion SECTION 5 END

#region SECTION 6: FRAME GENERATION
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
#endregion SECTION 6 END

#region SECTION 7: MAIN FUNCTION
if __name__ == "__main__":
    elevation_data = load_elevation_data(ELEVATION_DATA_PATH)
    if elevation_data is not None:
        grid_height, grid_width = elevation_data.shape

        # Adjust the grid size based on the elevation data 
        model = TerrainModel(width=grid_width, height=grid_height, num_king_seekers=KING_TOMBS, num_queen_seekers=QUEEN_TOMBS) 
    else:
        model = TerrainModel(num_king_seekers=KING_TOMBS, num_queen_seekers=QUEEN_TOMBS)

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
#endregion SECTION 7 END