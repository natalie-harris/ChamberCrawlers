# tomb_model.py
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
from mesa import Agent, Model
from shapely.geometry import Point

# Load tomb data
df = pd.read_csv("All Theban Tombs.csv")
df = df.rename(columns={"x": "Latitude", "y": "Longitude"})
# Do not include data that lacks latitude and longitude in training
df = df.dropna(subset=["Latitude", "Longitude"]) 

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.y, df.x))
# EPSG:4326 represents the WGS 84 geographic coordinate system (latitude/longitude)
gdf.set_crs(epsg=4326, inplace=True) 
# Convert to Web Mercator (EPSG:3857) for better visualization and calculations (it's in meters!)
gdf = gdf.to_crs(epsg=3857)  

# Get bounds - these are the limits for the random predictions
minx, miny, maxx, maxy = gdf.total_bounds

class TombSeekerAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.predicted_location = None

    def step(self):
        # Simple random prediction within bounds
        x = self.random.uniform(self.model.minx, self.model.maxx)
        y = self.random.uniform(self.model.miny, self.model.maxy)
        self.predicted_location = Point(x, y)

class TombModel(Model):
    def __init__(self, n_agents=10):
        self.agents = []
        self.minx, self.miny, self.maxx, self.maxy = minx, miny, maxx, maxy

        for i in range(n_agents):
            agent = TombSeekerAgent(i, self)
            self.agents.append(agent)

        self.predictions = []

    def step(self):
        for agent in self.agents:
            agent.step()
        self.predictions = [a.predicted_location for a in self.agents]
