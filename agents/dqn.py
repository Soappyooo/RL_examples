from agents.base_agent import Agent
from models.mlp import MLP
from models.input_encoder import InputEncoder
import torch
from utils.registry import register_agent


@register_agent("DQN")
class DQNAgent(Agent):
    def __init__(
        self,
        gamma: float,
        epsilon: float,
        epsilon_decay: float,
        epsilon_decay_steps: int,
        epsilon_min: float,
        learning_rate: float,
        learning_rate_decay: float,
        learning_rate_min: float,
        learning_rate_decay_steps: int,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple,
        actions: list[float | tuple],
        target_update_interval: int = 100,
        device: str = "cpu",
        input_encoding: bool = False,
        input_encoding_dim: int = 2,
    ) -> None:
        """
        Initialize the DQN agent.

        Args:
            gamma (float): Discount factor for future rewards
            epsilon (float): Exploration rate for epsilon-greedy policy
            epsilon_decay (float): Decay rate for epsilon
            epsilon_decay_steps (int): Steps for epsilon decay
            epsilon_min (float): Minimum value for epsilon
            learning_rate (float): Learning rate for the optimizer
            learning_rate_decay (float): Decay rate for learning rate
            learning_rate_min (float): Minimum value for learning rate
            learning_rate_decay_steps (int): Steps for learning rate decay
            input_dim (int): Dimension of the input features
            output_dim (int): Dimension of the output
            hidden_dims (tuple): Dimensions of the hidden layers
            actions (list): List of possible actions
            target_update_interval (int): Frequency of updating the target network
            device (str): Device to run the model on ('cpu' or 'cuda')
            input_encoding (bool): Whether to encode input into Fourier features
            input_encoding_dim (int): Dimension of the Fourier features
        """
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_min = learning_rate_min
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.device = device
        self.actions = actions
        self.input_encoding = input_encoding
        self.input_encoding_dim = input_encoding_dim
        # Convert any nested lists within actions to tuples
        if all(isinstance(a, list) for a in actions):
            self.actions = [tuple(action) for action in actions]
        else:
            self.actions = actions
        self.target_update_interval = target_update_interval
        self._update_counter = 0
        self._stored_epsilon = epsilon
        if self.input_encoding:
            self.q_network = torch.nn.Sequential(
                InputEncoder(input_dim, input_encoding_dim + input_dim, append_initial_input=True),
                MLP(input_encoding_dim + input_dim, output_dim, hidden_dims),
            ).to(device)
            self.target_network = torch.nn.Sequential(
                InputEncoder(input_dim, input_encoding_dim + input_dim, append_initial_input=True),
                MLP(input_encoding_dim + input_dim, output_dim, hidden_dims),
            ).to(device)
        else:
            self.q_network = MLP(input_dim, output_dim, hidden_dims).to(device)
            self.target_network = MLP(input_dim, output_dim, hidden_dims).to(device)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay
        )
        self._episode_counter = 0

    def eval(self) -> None:
        """
        Set the agent to evaluation mode.
        """
        self.q_network.eval()
        self._stored_epsilon = self.epsilon if self.epsilon > 0 else self._stored_epsilon
        self.epsilon = 0.0  # Disable exploration during evaluation

    def train(self) -> None:
        """
        Set the agent to training mode.
        """
        self.q_network.train()
        self.epsilon = self._stored_epsilon if self.epsilon == 0 else self.epsilon  # Restore exploration rate

    def buffer_tuple_to_dict(self, buffer: tuple) -> dict:
        """
        Convert a buffer tuple to a dictionary.

        Args:
            buffer (tuple): Tuple containing states, actions, rewards, next states, and done flags
        """
        states, actions, rewards, next_states, dones = buffer
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }

    def update(self, buffer: dict) -> dict[str, float]:
        """
        Update the Q-network using the replay buffer.

        Args:
            buffer (dict): Replay buffer containing states, actions, rewards, next states, and done flags

        Returns:
            dict[str, float]: Loss value from the update
        """
        states = torch.tensor(buffer["states"], dtype=torch.float32).to(self.device)  # Batch size x input_dim

        # Convert actions to indices
        action_ids = (
            torch.tensor(
                [self.action_to_index(action) for action in buffer["actions"]],
                dtype=torch.long,
            )
            .to(self.device)
            .view(-1, 1)
        )  # Batch size x 1
        rewards = torch.tensor(buffer["rewards"], dtype=torch.float32).to(self.device).view(-1, 1)  # Batch size x 1
        next_states = torch.tensor(buffer["next_states"], dtype=torch.float32).to(self.device)  # Batch size x input_dim
        dones = torch.tensor(buffer["dones"], dtype=torch.float32).to(self.device).view(-1, 1)  # Batch size x 1

        # Compute Q-values for current states
        q_values = self.q_network(states).gather(1, action_ids)  # Batch size x 1

        # Compute target Q-values for next states
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].view(-1, 1)  # Batch size x 1

        # Compute target values
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.loss_fn(q_values, targets)

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self._update_counter += 1
        if self._update_counter % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()

        return {"loss": loss.item()}

    def act(self, state: tuple) -> float | tuple:
        """
        Select an action based on the current state using epsilon-greedy policy.

        Args:
            state (tuple): Current state of the environment

        Returns:
            float | tuple: Selected action
        """

        with torch.no_grad():
            q_values = self.q_network(torch.tensor(state, dtype=torch.float32).to(self.device).view(-1))  # output_dim
        action_id = q_values.argmax().item()  # index of the action with max Q-value
        if torch.rand(1).item() < self.epsilon:
            action_id = torch.randint(0, len(self.actions), (1,)).item()  # Select action based on epsilon-greedy policy
        action = self.index_to_action(action_id)  # convert index to action
        return action  # return the selected action

    def action_to_index(self, action: float | tuple) -> int:
        """
        Convert action to index.

        Args:
            action (float | tuple): Action(s) to convert

        Returns:
            int: Index of the action
        """
        return self.actions.index(action)

    def index_to_action(self, index: int) -> float | tuple:
        """
        Convert index to action.

        Args:
            index (int): Index to convert

        Returns:
            float | tuple: Action(s) corresponding to the index
        """
        return self.actions[index]

    def end_episode(self):
        """
        End the current episode and update epsilon and learning rate.
        """

        if self._episode_counter % self.epsilon_decay_steps == 0:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        if self.learning_rate > self.learning_rate_min:
            self.lr_scheduler.step()
            self.learning_rate = self.lr_scheduler.get_last_lr()[0]

        self._episode_counter += 1

    @classmethod
    def from_config(cls, config: dict) -> "DQNAgent":
        """
        Create a DQNAgent instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing agent parameters

        Returns:
            DQNAgent: Instance of the DQNAgent class
        """
        # Define default values for configuration parameters
        default_config = {
            "gamma": 1,
            "epsilon": 0.1,
            "epsilon_decay": 1,
            "epsilon_decay_steps": 1,
            "epsilon_min": 0.01,
            "learning_rate": 0.001,
            "learning_rate_decay": 1,
            "learning_rate_min": 0.0001,
            "learning_rate_decay_steps": 100,
            "input_dim": 2,
            "output_dim": 1,
            "hidden_dims": [64, 64],
            "target_update_interval": 100,
            "device": "cpu",
            "actions": [0.0],
            "input_encoding": False,
            "input_encoding_dim": 2,
        }

        # Create a new configuration by combining the default config with the provided config
        agent_config = default_config.copy()
        for key, value in agent_config.items():
            try:
                agent_config[key] = config[key]
            except KeyError:
                print(f"\nKey {key} not found in config. Using default value: {value}")

        return cls(
            gamma=agent_config["gamma"],
            epsilon=agent_config["epsilon"],
            epsilon_decay=agent_config["epsilon_decay"],
            epsilon_decay_steps=agent_config["epsilon_decay_steps"],
            epsilon_min=agent_config["epsilon_min"],
            learning_rate=agent_config["learning_rate"],
            learning_rate_decay=agent_config["learning_rate_decay"],
            learning_rate_min=agent_config["learning_rate_min"],
            learning_rate_decay_steps=agent_config["learning_rate_decay_steps"],
            input_dim=agent_config["input_dim"],
            output_dim=agent_config["output_dim"],
            hidden_dims=tuple(agent_config["hidden_dims"]),
            target_update_interval=agent_config["target_update_interval"],
            device=agent_config["device"],
            actions=agent_config["actions"],
            input_encoding=agent_config["input_encoding"],
            input_encoding_dim=agent_config["input_encoding_dim"],
        )

    def save_checkpoint(self, filename: str) -> None:
        """
        Save the agent's checkpoint.

        Args:
            filename (str): Filename to save the checkpoint
        """
        torch.save(self.q_network.state_dict(), filename)

    def load_checkpoint(self, filename: str) -> None:
        """
        Load the agent's checkpoint.

        Args:
            filename (str): Filename to load the checkpoint from
        """
        self.q_network.load_state_dict(torch.load(filename, weights_only=True))
        self.target_network.load_state_dict(torch.load(filename, weights_only=True))
        self.q_network.to(self.device)
        self.target_network.to(self.device)
