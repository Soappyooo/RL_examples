from envs.pendulum_env import PendulumEnv
import numpy as np
import yaml
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from functools import partial
from utils.factory import create_agent
from matplotlib.animation import FFMpegWriter
import os
import torch
from utils.trainer import seed_all

CONTROLLER = "agent"  # "keyboard", "agent" or "pid"
perturbation = True  # Enable perturbation at specific steps
config_path = "config/pendulum/dqn.yml"  # Path to the configuration file
checkpoint_path = "checkpoints/pendulum/dqn/checkpoint_200.pth"  # Path to the checkpoint file


# Load configuration from YAML file
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Initialize environment
seed_all(1)
config["env"]["max_steps"] = 2000  # Set maximum steps
env = PendulumEnv.from_config(config["env"])

# Initialize agent
agent = create_agent(config["agent"])
agent.load_checkpoint(checkpoint_path)
agent.eval()

# Create a global variable to store current action
current_action = 0
state = env.reset()
total_reward = 0
action = 0

alphas = []
dot_alphas = []


# PID Controller class
class PIDController:
    def __init__(self, kp=10.0, ki=0.1, kd=1.0, setpoint=0.0, output_limits=(-3, 3)):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.setpoint = setpoint  # Target value (angle at which we want the pendulum, usually 0, meaning vertical up)
        self.output_limits = output_limits  # Output limits
        self.integral_limit = 1  # Integral limit (to prevent integral saturation)
        # Internal state
        self.prev_error = 0
        self.integral = 0
        self.last_time = None

    def reset(self):
        """Reset the PID controller state"""
        self.prev_error = 0
        self.integral = 0
        self.last_time = None

    def update(self, state, dt=0.02):
        """
        Update the PID controller

        Args:
            state: Current state, including angle and angular velocity
            dt: Time step

        Returns:
            float: Control output
        """
        # Extract angle and angular velocity from state
        angle, angular_velocity = state

        # Calculate error (target angle - current angle)
        # Note: We may need to handle angle periodicity, ensuring error is in [-pi, pi] range
        error = self.setpoint - angle
        error = ((error + np.pi) % (2 * np.pi)) - np.pi  # Ensure error is in [-pi, pi] range

        # Calculate integral term
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)  # Limit integral term

        # Calculate derivative term (using angular velocity as derivative can reduce noise)
        derivative = (
            -angular_velocity
        )  # Negative sign because angular velocity is opposite to error derivative direction

        # Calculate PID output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Limit output range
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        # Update previous error
        self.prev_error = error

        return output


# Create PID controller instance
kp = 6
ki = 0
kd = 0.5
pid_controller = PIDController(kp=kp, ki=ki, kd=kd, setpoint=0, output_limits=(-3, 3))

# Set up the figure for animation
fig, ax = plt.subplots(figsize=(5, 5))


# Add keyboard event handler
def on_key_press(event):
    global current_action
    if event.key == "left":
        current_action = 3
    elif event.key == "right":
        current_action = -3
    else:
        current_action = 0


# Add keyboard event release handler
def on_key_release(event):
    global current_action
    current_action = 0


# Function to update the animation
def animate(render_interval, i):
    global current_action, state, total_reward, action, alphas, dot_alphas
    reward = 0
    info = {}
    for _ in range(render_interval):
        if not env.done:
            # Default action is 0 for first 100 steps
            action = 0
            mixed_action = 0
            current_step = info.get("current_step", 0)

            # After 100 steps, enable controllers
            if current_step >= 100:

                if perturbation:
                    if 700 <= current_step <= 760:  # Change action for specific steps
                        current_action = -3  # Set action to -3 for perturbation

                    elif 1400 <= current_step <= 1460:
                        current_action = 3

                    else:
                        current_action = 0

                # Determine action based on controller type
                if CONTROLLER == "keyboard":
                    action = current_action
                    mixed_action = action
                elif CONTROLLER == "agent":
                    if current_step % config["env"]["final_act_interval"] == 0:
                        with torch.no_grad():
                            action = agent.act(state)
                        mixed_action = action + current_action  # Add keyboard control action
                elif CONTROLLER == "pid":
                    # PID controller with swing-up phase
                    if current_step < 150:  # First 50 steps after activation (100-150)
                        action = 3  # Swing right
                    elif current_step < 200:  # Next 50 steps (150-200)
                        action = -3  # Swing left
                    elif current_step % config["env"]["final_act_interval"] == 0:
                        action = pid_controller.update(state, dt=config["env"]["Ts"])

                    mixed_action = action + current_action  # Add keyboard control action

            # Execute action and get new state
            state, reward, done, info = env.step(mixed_action)
            alphas.append(state[0])
            dot_alphas.append(state[1])
            total_reward += reward

    # Status text for display
    status = "WAITING"
    if info.get("current_step", 0) >= 100:
        if CONTROLLER == "pid" and info.get("current_step", 0) < 200:
            status = "SWINGING UP"
        else:
            status = "ACTIVE"

    env.render(ax, action, reward, total_reward, current_action)

    # Display current controller type and status in chart title
    ax.set_title(f"Controller: {CONTROLLER.upper()} ({status}), Action: {action:.2f}")

    return ax


# Connect keyboard events
fig.canvas.mpl_connect("key_press_event", on_key_press)
fig.canvas.mpl_connect("key_release_event", on_key_release)

# Create animation
render_interval = 10
ani = FuncAnimation(
    fig,
    partial(animate, render_interval),
    frames=config["env"]["max_steps"] // render_interval,
    interval=int(config["env"]["Ts"] * 1000) * render_interval,
    repeat=False,
)

# Add usage instruction text
if CONTROLLER == "keyboard":
    instruction_text = "Press Left Arrow: Voltage +3, Press Right Arrow: Voltage -3, Other Keys: Voltage 0"
elif CONTROLLER == "pid":
    instruction_text = f"PID Controller: Kp={kp}, Ki={ki}, Kd={kd}"
else:
    instruction_text = "Agent Control"

plt.figtext(
    0.5,
    0.01,
    instruction_text,
    ha="center",
    fontsize=10,
)

# Save animation to mp4 file
# Setup the writer with desired parameters
writer = FFMpegWriter(
    fps=1000 / (config["env"]["Ts"] * 1000 * render_interval), metadata=dict(artist="Me"), bitrate=1800
)

# Save the animation
# output_file = os.path.join("./visualization", f"pendulum_{CONTROLLER}_animation.mp4")
# os.makedirs(os.path.dirname(output_file), exist_ok=True)
# ani.save(output_file, writer=writer)
# print(f"Animation saved to {output_file}")
# plt.close(fig)

# Display the animation
plt.show()

# plot alpha and dot_alpha in one figure with different y-axis
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
ax1.plot(alphas, label="Alpha (Angle)", color="blue", alpha=0.8)
ax2.plot(dot_alphas, label="Dot Alpha (Angular Velocity)", color="orange", alpha=0.8)
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Alpha (Angle)")
ax2.set_ylabel("Dot Alpha (Angular Velocity)")
# ax1.tick_params(axis="y", labelcolor="blue")
# ax2.tick_params(axis="y", labelcolor="orange")
ax1.set_title("Pendulum State Over Time")
# add perturbation indicator
ax2.axvspan(700, 760, color="red", alpha=0.3, label="Perturbation")
ax2.axvspan(1400, 1460, color="red", alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# make axes center at zero
ax1.set_ylim(-np.pi, np.pi)
max_dot_alpha = max(np.abs(dot_alphas)) if dot_alphas else 0
ax2.set_ylim(-max_dot_alpha, max_dot_alpha)
ax1.set_xlim(0, len(alphas))
ax1.grid()

plt.show()
