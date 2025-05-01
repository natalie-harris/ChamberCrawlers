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

# Path to the elevation data JSON file
ELEVATION_DATA_PATH = "../../data/data_acquisition/elevation/KVElevation.json"
OUTPUT_DIR = "output"
OUTPUT_FRAMES_DIR = os.path.join(OUTPUT_DIR, "simulation_frames")
FRAMES_PER_SECOND = 5
TOTAL_STEPS = 100  # Number of simulation steps to run

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
    def __init__(self, width=None, height=None, num_agents=10, elevation_data_path=ELEVATION_DATA_PATH):
        super().__init__()
        self.elevation_data = load_elevation_data(elevation_data_path)

        if self.elevation_data is None:
            if width is None or height is None:
                self.width = 20
                self.height = 25
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

        origin = (0, 0)
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
    """Generates a single frame of the simulation as a heatmap with correct orientation and labels."""
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
    plt.imshow(np.flipud(normalized_elevation), cmap='viridis', origin='lower', extent=[0, grid_width * cell_size, 0, grid_height * cell_size])
    plt.colorbar(label='Normalized Elevation')

    # Plot agent positions (adjust y-coordinate for flipped axis)
    agent_x = [pos[0] * cell_size + cell_size / 2 for pos in agent_positions.values()]
    agent_y = [(grid_height - 1 - pos[1]) * cell_size + cell_size / 2 for pos in agent_positions.values()]
    plt.scatter(agent_x, agent_y, color='blue', s=50, label='Agents')

    # Set ticks based on meters from the bottom left
    x_ticks = np.arange(0, grid_width * cell_size, cell_size * 5)  # Every 5 cells (150 m)
    y_ticks = np.arange(0, grid_height * cell_size, cell_size * 5) # Every 5 cells (150 m)
    plt.xticks(x_ticks, [f'{int(t)}m' for t in x_ticks])
    plt.yticks(y_ticks, [f'{int(t)}m' for t in y_ticks])
    plt.xlabel('Meters (East)')
    plt.ylabel('Meters (North)')

    # Add lat/lon at the corners
    plt.text(0, 0, f'Lat: {min_lat:.5f}\nLon: {min_lon:.5f}', ha='left', va='bottom', fontsize=8, bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3})
    plt.text(grid_width * cell_size, 0, f'Lat: {min_lat:.5f}\nLon: {max_lon:.5f}', ha='right', va='bottom', fontsize=8, bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3})
    plt.text(0, grid_height * cell_size, f'Lat: {max_lat:.5f}\nLon: {min_lon:.5f}', ha='left', va='top', fontsize=8, bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3})
    plt.text(grid_width * cell_size, grid_height * cell_size, f'Lat: {max_lat:.5f}\nLon: {max_lon:.5f}', ha='right', va='top', fontsize=8, bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3})

    plt.title(f'Simulation Step: {step}')
    plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"step_{step:04d}.png")
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    model = TerrainModel(width=8, height=6) # Adjust width and height based on your data dimensions
    min_latitude = 25.73753
    max_latitude = 25.74315
    min_longitude = 32.59838
    max_longitude = 32.6047
    cell_size = 30 # meters per cell

    # Create output directory and frames directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

    # Run the simulation and generate frames
    for step in range(TOTAL_STEPS):
        print(f"Generating frame for step {step}")
        generate_frame(model, step, OUTPUT_FRAMES_DIR, min_latitude, max_latitude, min_longitude, max_longitude, cell_size)
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