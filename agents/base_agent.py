class Agent:
    def __init__(self) -> None:
        pass

    def buffer_tuple_to_dict(self, buffer: tuple) -> dict:
        """
        Convert a buffer tuple to a dictionary.

        Args:
            buffer (tuple): Tuple containing experiences.
        """
        raise NotImplementedError("Buffer tuple to dict method not implemented.")

    def update(self, buffer: dict) -> dict[str, float]:
        """
        Update the agent using the provided buffer.

        Args:
            buffer (dict): Buffer containing experiences.

        Returns:
            dict[str, float]: Loss value from the update
        """
        raise NotImplementedError("Update method not implemented.")

    def act(self, state: tuple) -> float | tuple:
        """
        Select an action based on the current state using epsilon-greedy policy.

        Args:
            state (tuple): Current state of the environment

        Returns:
            float | tuple: Selected action
        """
        raise NotImplementedError("Act method not implemented.")

    def train(self) -> None:
        """
        Set the agent to training mode.
        """
        pass

    def eval(self) -> None:
        """
        Set the agent to evaluation mode.
        """
        pass

    def save_checkpoint(self, filename: str) -> None:
        """
        Save the agent's checkpoint.
        """
        raise NotImplementedError("Save checkpoint method not implemented.")

    def load_checkpoint(self, filename: str) -> None:
        """
        Load the agent's checkpoint.
        """
        raise NotImplementedError("Load checkpoint method not implemented.")

    def end_episode(self) -> None:
        """
        Perform any necessary cleanup or finalization at the end of an episode.
        """
        pass

    def begin_episode(self) -> None:
        """
        Perform any necessary initialization at the beginning of an episode.
        """
        pass

    @classmethod
    def from_config(cls, config: dict) -> "Agent":
        """
        Create an agent instance from a configuration dictionary.
        """
        raise NotImplementedError("From config method not implemented.")
