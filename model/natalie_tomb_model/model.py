# model.py
from mesa import Model                                          # Mesa Model base class :contentReference[oaicite:0]{index=0}
from mesa.space import SingleGrid                                # 2D grid space :contentReference[oaicite:1]{index=1}
from mesa.time import RandomActivation                           # Scheduler for random activation :contentReference[oaicite:2]{index=2}

class GridModel(Model):
    """
    A Mesa model with a SingleGrid spanning logical coordinates -10 to +10.
    """
    def __init__(self, width=21, height=21, torus=False):
        super().__init__()                                       # Initializes Model and RNG :contentReference[oaicite:3]{index=3}
        # Create the grid space
        self.grid = SingleGrid(width, height, torus)             # SingleGrid has no register_agent :contentReference[oaicite:4]{index=4}
        # Offset to map logical coords (-10…+10) → grid indices (0…20)
        self.coord_offset = width // 2

        # Scheduler to activate agents in random order
        self.schedule = RandomActivation(self)                   # Model holds agents and their schedule :contentReference[oaicite:5]{index=5}

    def step(self):
        """
        Advance the model by one step.
        """
        self.schedule.step()                                      # Steps through each agent’s step() :contentReference[oaicite:6]{index=6}
