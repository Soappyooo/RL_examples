from envs.base_env import Env
import numpy as np
import matplotlib.pyplot as plt
from utils.registry import register_env


@register_env("Pendulum")
class PendulumEnv(Env):
    def __init__(self, m, g, l, J, b, K, R, max_voltage, max_speed, Ts, max_steps):
        super().__init__()
        self.m = m
        self.g = g
        self.l = l
        self.J = J
        self.b = b
        self.K = K
        self.R = R
        self.max_voltage = max_voltage
        self.max_speed = max_speed
        self.Ts = Ts
        self.max_steps = max_steps
        self.current_step = 0
        self.state = None
        self.done = False
        self.alpha = 0
        self.dot_alpha = 0
        self.ddot_alpha = 0
        self.constants = self._calculate_constants()
        self.reset()

    def _calculate_constants(self) -> tuple:
        """
        Calculate constants for calculation based on the parameters of the pendulum.
        """
        return (
            self.m * self.g * self.l / self.J,
            (-self.b - np.power(self.K, 2) / self.R) / self.J,
            self.K / (self.J * self.R),
        )

    def reset(self):
        self.current_step = 0
        self.alpha = -np.pi
        self.dot_alpha = 0
        self.ddot_alpha = 0
        self.state = (self.alpha, self.dot_alpha)
        self.done = False
        return self.state

    def step(self, action: float) -> tuple:
        """
        Take a step in the environment with the given action.
        Returns the new state, reward, done flag, and additional info.
        """
        if self.done:
            raise Exception("Environment is done. Please reset it.")

        # Limit action to max voltage
        # action = np.clip(action, -self.max_voltage, self.max_voltage)

        # Calculate the new state based on the action
        self.ddot_alpha = (
            self.constants[0] * np.sin(self.alpha) + self.constants[1] * self.dot_alpha + self.constants[2] * action
        )
        self.dot_alpha += self.ddot_alpha * self.Ts
        self.alpha += self.dot_alpha * self.Ts
        # alpha is limited to [-pi, pi)
        self.alpha = ((self.alpha + np.pi) % (2 * np.pi)) - np.pi
        # dot_alpha is limited to [-max_speed, max_speed]
        self.dot_alpha = np.clip(self.dot_alpha, -self.max_speed, self.max_speed)
        self.state = (self.alpha, self.dot_alpha)

        # Calculate the reward

        # original reward
        # reward = -5 * np.power(self.alpha, 2) - 0.1 * np.power(self.dot_alpha, 2) - np.power(action, 2)
        # reward *= 1e-2

        # improved reward
        upright_angle = -np.abs(self.alpha) + np.pi / 2
        reward = 0.05 * (
            upright_angle + 0.001 * np.exp(upright_angle * 5)
        )  # higher value like 0.2 or lower value like 0.001 might be hard for DQN to learn

        if np.abs(np.degrees(self.alpha)) < 20:
            velocity_penalty = 0.05 * (np.tanh(np.abs(self.dot_alpha) * 0.1))
            action_penalty = 0.05 * (np.tanh(np.abs(action) * 0.1))
            reward += 0.1 - velocity_penalty - action_penalty  # Extra reward approaches 0 as velocity increases

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        additional_info = {
            "current_step": self.current_step,
        }

        return self.state, reward, self.done, additional_info

    def render(
        self,
        ax: plt.Axes,
        action: float = None,
        reward: float = None,
        total_reward: float = None,
        perturbation: float = None,
    ) -> None:
        """
        Render the environment with matplotlib.

        Args:
            ax: The matplotlib Axes object to render on.
            action: The action taken in the environment.
            reward: The reward received from the environment.
            total_reward: The total reward received so far.
            perturbation: The perturbation action applied to the environment.
        """

        ax.clear()
        ax.set_xlim(-self.l * 2, self.l * 2)
        ax.set_ylim(-self.l * 2, self.l * 2)
        ax.set_title("Pendulum")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Draw the pendulum
        # alpha=0 is top
        x = self.l * np.sin(self.alpha)
        y = self.l * np.cos(self.alpha)
        ax.plot([0, x], [0, y], "r-", lw=2)
        ax.plot(x, y, "bo", markersize=10)

        # Display the current state at top left corner
        fontsize = 10
        ax.text(
            0,
            1,
            f"alpha: {np.degrees(self.alpha):.2f}°",
            fontsize=fontsize,
            ha="left",
            transform=ax.transAxes,
        )
        ax.text(
            0,
            0.9,
            f"dot_alpha: {self.dot_alpha:.2f}",
            fontsize=fontsize,
            ha="left",
            transform=ax.transAxes,
        )
        if action is not None:
            ax.text(
                0,
                0.8,
                f"action: {action:.2f}",
                fontsize=fontsize,
                ha="left",
                transform=ax.transAxes,
            )
        if perturbation is not None:
            ax.text(
                0,
                0.7,
                f"perturbation: {perturbation:.2f}",
                fontsize=fontsize,
                ha="left",
                transform=ax.transAxes,
            )
            # Draw perturbation arrow when perturbation is not zero
            if perturbation != 0:
                # Calculate pendulum endpoint
                x = self.l * np.sin(self.alpha)
                y = self.l * np.cos(self.alpha)

                # Determine arrow length based on perturbation sign
                arrow_length = 0.3 * self.l * np.sign(perturbation)

                # Arrow is perpendicular to pendulum but in opposite direction
                dx = arrow_length * np.cos(self.alpha)  # Reversed direction
                dy = -arrow_length * np.sin(self.alpha)  # Reversed direction

                # Draw arrow at pendulum endpoint
                ax.arrow(x, y, dx, dy, head_width=0.01, head_length=0.01, fc="green", ec="green", width=0.002)
        if total_reward is not None:
            ax.text(
                0,
                0.6,
                f"total_reward: {total_reward:.2f}",
                fontsize=fontsize,
                ha="left",
                transform=ax.transAxes,
            )
        if reward is not None:
            ax.text(
                0,
                0.5,
                f"reward: {reward:.2f}",
                fontsize=fontsize,
                ha="left",
                transform=ax.transAxes,
            )
        ax.text(
            0,
            0.4,
            f"step: {self.current_step}/{self.max_steps}",
            fontsize=fontsize,
            ha="left",
            transform=ax.transAxes,
        )
        ax.text(
            0,
            0.3,
            f"time: {self.current_step * self.Ts:.2f}s/{self.max_steps * self.Ts:.2f}s",
            fontsize=fontsize,
            ha="left",
            transform=ax.transAxes,
        )
        ax.axis("off")

    @classmethod
    def from_config(cls, config: dict) -> "PendulumEnv":
        """
        Create a PendulumEnv instance from a configuration dictionary.
        """
        return cls(
            m=config["m"],
            g=config["g"],
            l=config["l"],
            J=config["J"],
            b=config["b"],
            K=config["K"],
            R=config["R"],
            max_voltage=config["max_voltage"],
            max_speed=config["max_speed"],
            Ts=config["Ts"],
            max_steps=config["max_steps"],
        )
