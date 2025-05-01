from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import animation
from model import GridModel                                      # Your custom Model
from builder import Builder                                      # Your Builder agent

def run_simulation(steps=50):
    # 1. Initialize the model
    model = GridModel(width=21, height=21, torus=False)

    # 2. Create and place Builder agent
    x_logical, y_logical = 0, 0
    x = x_logical + model.coord_offset
    y = y_logical + model.coord_offset

    builder_agent = Builder(unique_id=1, model=model, initial_cell=(x, y))
    model.schedule.add(builder_agent)                            # Add to scheduler

    # 3. Set up Matplotlib figure for visualization
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, model.grid.width)
    ax.set_ylim(0, model.grid.height)
    ax.set_aspect('equal')

    def update(frame_number):
        ax.clear()
        model.step()                                             # Advance the simulation
        # Draw each agent
        for agent in model.schedule.agents:
            ax.scatter(agent.location[0], agent.location[1], s=100, c="blue") # Use agent.location
        ax.set_title(f"Step {frame_number}")

    # 4. Animate and save to a timestamped video file
    anim = animation.FuncAnimation(fig, update, frames=steps, interval=200)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/media/agm_simulation_{timestamp}.mp4"
    writer = animation.FFMpegWriter(fps=5)
    anim.save(filename, writer=writer)

    print(f"Saved simulation video as {filename}")

if __name__ == "__main__":
    run_simulation(steps=50)