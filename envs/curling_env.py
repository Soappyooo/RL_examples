from envs.base_env import Env
import numpy as np
import matplotlib.pyplot as plt
from utils.registry import register_env


@register_env("Curling")
class CurlingEnv(Env):
    def __init__(
        self, r, m, h, w, rebound_coefficient, initial_speed_range, dt, max_steps, num_steps_per_step, normalize=False
    ):
        super().__init__()
        self.r = r
        self.m = m
        self.h = h
        self.w = w
        self.rebound_coefficient = rebound_coefficient
        self.initial_speed_range = initial_speed_range
        self.max_steps = max_steps
        self.num_steps_per_step = num_steps_per_step  # take this many steps in one step of the environment
        self.dt = dt
        self.current_step = 0
        self.state = None
        self.current_position = None
        self.target_position = None
        self.done = False
        self.current_speed = None
        self.normalize = normalize  # Normalize the state space and reward
        self.reset()

    def reset(self):
        self.current_step = 0
        self.current_position = self._sample_position()
        self.target_position = self._sample_position()
        self.done = False
        self.current_speed = np.random.uniform(self.initial_speed_range[0], self.initial_speed_range[1], (2,))
        self.current_speed = tuple(self.current_speed)
        self.state = self._get_state()
        return self.state

    def _get_state(self) -> tuple[float, float, float, float, float, float]:
        """
        Get the current state of the environment.
        """
        if not self.normalize:
            return (
                self.current_position[0],
                self.current_position[1],
                self.current_speed[0],
                self.current_speed[1],
                self.target_position[0],
                self.target_position[1],
            )
        else:
            return (
                self.current_position[0] / self.w,
                self.current_position[1] / self.h,
                self.current_speed[0] / self.initial_speed_range[1],
                self.current_speed[1] / self.initial_speed_range[1],
                self.target_position[0] / self.w,
                self.target_position[1] / self.h,
            )

    def _sample_position(self) -> tuple[float, float]:
        """
        Sample a random position within the curling area.
        """
        x = np.random.uniform(0, self.w)
        y = np.random.uniform(0, self.h)
        return x, y

    def _calculate_reward(self) -> float:
        """
        Calculate the reward based on the distance to the target.
        """
        distance = np.linalg.norm(np.array(self.current_position) - np.array(self.target_position))
        reward = -distance
        if self.normalize:
            reward /= np.linalg.norm(np.array([self.w, self.h]))  # Normalize reward
        return reward

    def _update_speed(self, action: tuple[float, float]) -> None:
        """
        Update the speed of the stone based on the action taken.
        """
        # acceleration = action / self.m
        acceleration = np.array(action) / self.m
        # Apply acceleration & air resistance
        self.current_speed = tuple(
            np.array(self.current_speed)
            + acceleration * self.dt
            - 0.005 * np.linalg.norm(self.current_speed) ** 2 * np.sign(self.current_speed) * self.dt
        )

    def _update_position(self) -> None:
        """
        Update the position of the stone based on the current speed.
        """
        new_x = self.current_position[0] + self.current_speed[0] * self.dt
        new_y = self.current_position[1] + self.current_speed[1] * self.dt

        # Check for rebound with walls, consider radius of the curling stone
        if new_x - self.r < 0 or new_x + self.r > self.w:
            new_x = max(self.r, min(new_x, self.w - self.r))
            self.current_speed = (
                self.rebound_coefficient * -self.current_speed[0],
                self.rebound_coefficient * self.current_speed[1],
            )
        if new_y - self.r < 0 or new_y + self.r > self.h:
            new_y = max(self.r, min(new_y, self.h - self.r))
            self.current_speed = (
                self.rebound_coefficient * self.current_speed[0],
                self.rebound_coefficient * -self.current_speed[1],
            )

        self.current_position = (new_x, new_y)

    def step(self, action: tuple[float, float]) -> tuple[tuple[float, float], float, bool, dict]:
        """
        Take a step in the environment with the given action.
        Returns the new state, reward, done flag, and additional info.
        """
        if self.done:
            raise Exception("Environment is done. Please reset it.")

        for _ in range(self.num_steps_per_step):
            # Update speed and position
            self._update_speed(action)
            self._update_position()

            # Check if the episode is done
            self.current_step += 1
            if self.current_step >= self.max_steps:
                self.done = True
                break

        # Update state
        self.state = self._get_state()

        # Calculate reward
        reward = self._calculate_reward()

        additional_info = {
            "current_step": self.current_step,
            "current_position": self.current_position,
            "current_speed": self.current_speed,
            "target_position": self.target_position,
        }

        return self.state, reward, self.done, additional_info

    def render(
        self,
        ax: plt.Axes,
        action: float = None,
        reward: float = None,
    ) -> None:
        """
        Render the environment.
        """
        ax.clear()
        ax.set_xlim(0, self.w)
        ax.set_ylim(0, self.h)
        ax.set_title("Curling Environment")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

        # Draw the curling stone
        # circle = plt.Circle(self.current_position, self.r, color="blue")
        # ax.add_artist(circle)
        ax.plot(self.current_position[0], self.current_position[1], "bo", markersize=10)

        # Draw the target position as square
        # target_square = plt.Rectangle(self.target_position, self.r, self.r, color="red")
        # ax.add_artist(target_square)
        ax.plot(self.target_position[0], self.target_position[1], "ro", markersize=10)

        # Draw the action and reward if provided
        if action is not None:
            ax.text(0.5, 0.9, f"Action: {action}", transform=ax.transAxes, fontsize=12, ha="center")
        if reward is not None:
            ax.text(0.5, 0.85, f"Reward: {reward:.2f}", transform=ax.transAxes, fontsize=12, ha="center")

        # Draw force vector if action is provided
        if action is not None:
            force_vector = np.array(action) / self.m
            ax.quiver(
                self.current_position[0],
                self.current_position[1],
                force_vector[0],
                force_vector[1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color="green",
            )

        # Draw speed vector
        speed_vector = np.array(self.current_speed)
        ax.quiver(
            self.current_position[0],
            self.current_position[1],
            speed_vector[0],
            speed_vector[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="orange",
        )

    @classmethod
    def from_config(cls, config) -> "CurlingEnv":
        """
        Create an environment instance from the provided configuration.
        """
        return cls(
            r=config["r"],
            m=config["m"],
            h=config["h"],
            w=config["w"],
            rebound_coefficient=config["rebound_coefficient"],
            initial_speed_range=config["initial_speed_range"],
            dt=config["dt"],
            max_steps=config["max_steps"],
            num_steps_per_step=config["num_steps_per_step"],
            normalize=config["normalize"],
        )
