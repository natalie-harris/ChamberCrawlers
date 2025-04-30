from mesa import Model
from mesa.time import RandomActivation
from tomb_builder import TombBuilder  # tomb_builder.py
import rasterio
import numpy as np

class TombModel(Model):
    def __init__(self, num_agents, tif_file):
        super().__init__()
        self.num_agents = num_agents
        self.schedule = RandomActivation(self)
        self.tombs = []  # Stores all placed tombs

        # Load elevation data
        self.raster = rasterio.open(tif_file)
        self.elevation_array = self.raster.read(1)
        self.transform = self.raster.transform

        # Define valley boundaries (restrict agent movement)
        # self.lon_min = self.raster.bounds.left
        # self.lon_max = self.raster.bounds.right
        # self.lat_min = self.raster.bounds.bottom
        # self.lat_max = self.raster.bounds.top

        # Use manually set Valley of the Kings bounds
        self.lon_min = 32.590
        self.lon_max = 32.620
        self.lat_min = 25.735
        self.lat_max = 25.755

        # Spawn agents
        for i in range(self.num_agents):
            # Start agents at random positions in the valley
            start_lon = self.random.uniform(self.lon_min, self.lon_max)
            start_lat = self.random.uniform(self.lat_min, self.lat_max)
            agent = TombBuilder(i, self, start_lon, start_lat)
            self.schedule.add(agent)

    def step(self):
        self.schedule.step()

    def get_elevation(self, lon, lat):
        try:
            row, col = self.raster.index(lon, lat)
            return self.elevation_array[row, col]
        except IndexError:
            return None  # Out of bounds
