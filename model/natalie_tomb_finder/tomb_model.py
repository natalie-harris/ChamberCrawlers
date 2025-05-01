import pandas as pd
import numpy as np
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.visualization import SolaraViz
import solara
import altair as alt
# Function to generate a random elevation grid
def generate_elevation_grid(width, height):
    return np.random.randint(0, 101, size=(height, width))

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
    def __init__(self, width=50, height=50, num_agents=10):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.elevation = generate_elevation_grid(width, height)
        self.steps = 0
        self.datachanged = solara.reactive(False)  # Trigger re-render

        origin = (0, 0)
        for i in range(num_agents):
            agent = WalkerAgent(f"walker_{i}", self, origin)
            self.schedule.add(agent)
            self.grid.place_agent(agent, origin)

        self.running = True

    def step(self):
        self.schedule.step()
        self.steps += 1
        self.datachanged.set(not self.datachanged.value)  # Force re-render

"""
APP
"""

@solara.component
def TerrainGrid(model):
    grid_width = model.grid.width
    grid_height = model.grid.height
    scale = 20  # pixels per cell

    # Agent data: Update this on each step to reflect agent positions
    agents_df = pd.DataFrame([
        {"x": a.pos[0], "y": a.pos[1]}
        for a in model.schedule.agents
    ])

    # Background grid colored by elevation
    tiles_df = pd.DataFrame([
        {"x": x, "y": y, "elevation": model.elevation[y, x]}
        for x in range(grid_width)
        for y in range(grid_height)
    ])

    # Color scale for elevation (low to high)
    color_scale = alt.Scale(domain=[tiles_df["elevation"].min(), tiles_df["elevation"].max()],
                            range=["lightblue", "darkgreen"])

    background = (
        alt.Chart(tiles_df)
        .mark_rect(stroke="lightgray", fillOpacity=0.8, strokeWidth=0.5)
        .encode(
            x=alt.X("x:O", title="X"),
            y=alt.Y("y:O", title="Y"),
            fill=alt.Color("elevation:Q", scale=color_scale, legend=None)  # Color by elevation
        )
    )

    # Agent layer (blue circles)
    agents_layer = (
        alt.Chart(agents_df)
        .mark_circle(size=100, color="blue")
        .encode(
            x=alt.X("x:O"),
            y=alt.Y("y:O")
        )
    )

    # Combine background and agent layers
    chart = (
        (background + agents_layer)
        .properties(width=grid_width * scale, height=grid_height * scale)
    )

    return solara.AltairChart(chart)

# @solara.component
# def StepCounter(model):
#     return solara.Text(f"Step: {model.steps}")

# Model parameters
model_params = {"width": 20, "height": 25, "num_agents": 10}
initial_model = TerrainModel(**model_params)

# Make the model reactive for the Solara app
model = solara.reactive(initial_model)
page = SolaraViz(
    model=initial_model,
    components=[TerrainGrid],
    model_params=model_params,
    name="Terrain Following Agents",
)
page  # noqa
