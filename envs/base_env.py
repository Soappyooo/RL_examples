class Env:
    """
    Base class for all environments.
    """

    def __init__(self) -> None:
        """
        Initialize the environment.
        """
        pass

    def reset(self) -> tuple:
        """
        Reset the environment to its initial state.

        Returns:
            tuple: Initial state of the environment
        """
        raise NotImplementedError("Reset method not implemented.")

    def step(self, action: float | tuple) -> tuple:
        """
        Take a step in the environment using the given action.

        Args:
            action (float | tuple): Action to take in the environment

        Returns:
            tuple: A tuple containing the next state, reward, done flag, and additional info.
        """
        raise NotImplementedError("Step method not implemented.")

    def render(self) -> None:
        """
        Render the environment.
        """
        raise NotImplementedError("Render method not implemented.")

    @classmethod
    def from_config(cls, config: dict) -> "Env":
        """
        Create an environment instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing environment parameters

        Returns:
            Env: Instance of the environment class
        """
        raise NotImplementedError("from_config method not implemented.")
