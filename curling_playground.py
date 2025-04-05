from envs.curling_env import CurlingEnv
import numpy as np
import yaml
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FFMpegWriter
from utils.factory import create_agent
import os
import time
from utils.trainer import seed_all

CONTROLLER = "agent"  # "keyboard", "agent"
config_path = "config/curling/ppo.yml"  # Path to the configuration file
checkpoint_path = "checkpoints/curling/ppo/checkpoint_800.pth"  # Path to the checkpoint file

# Load configuration from YAML file
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Initialize environment
seed_all(1)
# config["env"]["num_steps_per_step"] = 10  # Set number of steps per step
config["env"]["max_steps"] = 1500  # Set maximum steps
env = CurlingEnv.from_config(config["env"])

# Initialize agent
agent = create_agent(config["agent"])
agent.load_checkpoint(checkpoint_path)
agent.eval()

# Create global variables to store current action
current_action_x = 0
current_action_y = 0
state = env.reset()
total_reward = 0
action = (0, 0)

# Variables to track distance and episodes
distances = []
episode_boundaries = [0]  # Mark the start of the first episode

# Set up the figure for animation
fig, ax = plt.subplots(figsize=(8, 8))


# Add keyboard event handler
def on_key_press(event):
    global current_action_x, current_action_y
    force = 5
    if event.key == "up":
        current_action_y = force
    elif event.key == "down":
        current_action_y = -force
    elif event.key == "left":
        current_action_x = -force
    elif event.key == "right":
        current_action_x = force


# Add keyboard event release handler
def on_key_release(event):
    global current_action_x, current_action_y
    if event.key in ["up", "down"]:
        current_action_y = 0
    elif event.key in ["left", "right"]:
        current_action_x = 0


# Function to update the animation
def animate(render_interval, i):
    global current_action_x, current_action_y, state, total_reward, action, distances, episode_boundaries
    reward = 0

    for _ in range(render_interval):
        if not env.done:

            if CONTROLLER == "keyboard":
                # Use keyboard input for action
                action = (current_action_x, current_action_y)
            elif CONTROLLER == "agent":
                # Use DQN agent for action
                action = agent.act(state)

            # Execute action and get new state
            state, reward, done, info = env.step(action)
            total_reward += reward

            # Calculate and record distance to target
            current_pos = np.array(info["current_position"])
            target_pos = np.array(info["target_position"])
            distance = np.linalg.norm(current_pos - target_pos)
            distances.append(distance)

        else:
            # Reset the environment if done
            state = env.reset()
            total_reward = 0
            # Mark the boundary of a new episode
            episode_boundaries.append(len(distances))

    # Render the environment
    env.render(ax, action, reward)

    # Display instructions and current state
    ax.text(
        0.5,
        0.02,
        f"Use arrow keys to control. Current force: ({current_action_x:.1f}, {current_action_y:.1f})",
        transform=ax.transAxes,
        fontsize=10,
        ha="center",
    )
    ax.text(
        0.5,
        0.06,
        f"Total Reward: {total_reward:.2f}, Step: {env.current_step}/{env.max_steps}"
        f" Time: {env.current_step * env.dt:.2f}s",
        transform=ax.transAxes,
        fontsize=10,
        ha="center",
    )

    # Display position information
    ax.text(
        0.5,
        0.1,
        f"Position: ({state[0]:.2f}, {state[1]:.2f}), Speed: ({state[2]:.2f}, {state[3]:.2f})",
        transform=ax.transAxes,
        fontsize=10,
        ha="center",
    )

    return ax


# Connect keyboard events
fig.canvas.mpl_connect("key_press_event", on_key_press)
fig.canvas.mpl_connect("key_release_event", on_key_release)

# Create animation
render_interval = 1
num_envs = 5  # Number of environments to run in total
ani = FuncAnimation(
    fig,
    partial(animate, render_interval),
    frames=env.max_steps // (render_interval * env.num_steps_per_step) * num_envs,
    interval=int(env.dt * 1000) * render_interval * env.num_steps_per_step,
    repeat=False,
)

# Add usage instruction text
instruction_text = "Controls: Up/Down/Left/Right Arrows to apply force"
plt.figtext(
    0.5,
    0.95,
    instruction_text,
    ha="center",
    fontsize=12,
)

# Save animation to mp4 file
writer = FFMpegWriter(
    fps=1000 / (env.dt * 1000 * render_interval * env.num_steps_per_step), metadata=dict(artist="Me"), bitrate=1800
)

# Save the animation
# output_file = os.path.join("./visualization", "curling_ppo_animation.mp4")
# os.makedirs(os.path.dirname(output_file), exist_ok=True)
# ani.save(output_file, writer=writer)
# print(f"Animation saved to {output_file}")
# plt.close(fig)

# Display the animation
plt.show()

# After the animation is complete, plot the distance to target over time
if distances:
    plt.figure(figsize=(10, 5))

    # Plot each episode as a separate line segment
    for i in range(len(episode_boundaries)):
        start_idx = episode_boundaries[i]
        # Determine end index (either next boundary or end of data)
        end_idx = episode_boundaries[i + 1] if i + 1 < len(episode_boundaries) else len(distances)

        # Plot this episode segment
        episode_data = distances[start_idx:end_idx]
        x_values = range(start_idx, end_idx)
        plt.plot(x_values, episode_data, color="blue", alpha=0.8)

        # Add episode marker for all boundary
        if start_idx < len(distances):
            plt.axvline(x=start_idx, color="red", linestyle="--", alpha=0.5)
            plt.text(
                start_idx + 5,
                max(distances) * 0.5,
                f"Episode {i}",
                fontsize=8,
                color="red",
                alpha=0.7,
                verticalalignment="top",
                rotation=90,
                ha="left",
            )

    plt.xlabel("Time Step")
    plt.ylabel("Distance to Target")
    plt.xlim(0, len(distances))
    plt.title("Distance to Target Over Time")
    plt.text(0.02, 0.95, f"Total Episodes: {len(episode_boundaries)}", transform=plt.gca().transAxes)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
