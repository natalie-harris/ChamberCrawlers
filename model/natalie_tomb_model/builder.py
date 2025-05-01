from mesa.experimental.cell_space import Grid2DMovingAgent
from mesa import Model

class Builder(Grid2DMovingAgent):
    """
    A Builder agent that can move on a 2D grid.
    """
    def __init__(self, unique_id: int, model: Model, initial_cell: tuple[int, int] = None):
        """
        Initialize a Builder agent.

        Args:
            unique_id (int): A unique identifier for this agent.
            model (Model): The model instance.
            initial_cell (tuple[int, int], optional): The initial position of the agent on the grid.
        """
        super().__init__(model)
        self.initial_cell = initial_cell
        if initial_cell:
            self.location = initial_cell
            self.model.grid.place_agent(self, initial_cell)
        self.unique_id = unique_id
        self.model = model
        # Example state variable
        self.built = False

    def step(self):
        """
        Defines the Builder's behavior each tick.
        Currently a placeholder to be filled based on requirements.
        """
        pass