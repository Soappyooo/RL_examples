from agents.base_agent import Agent
from models.mlp import MLP
from models.input_encoder import InputEncoder
import torch
import torch.nn.functional as F
import numpy as np
from utils.registry import register_agent


@register_agent("PPO")
class PPOAgent(Agent):
    def __init__(
        self,
        gamma: float,
        clip_ratio: float,
        policy_lr: float,
        value_lr: float,
        learning_rate_decay: float,
        learning_rate_min: float,
        learning_rate_decay_steps: int,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple,
        actions: list[float | tuple],
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        device: str = "cpu",
        input_encoding: bool = False,
        input_encoding_dim: int = 2,
    ) -> None:
        """
        Initialize the PPO agent.

        Args:
            gamma (float): Discount factor for future rewards
            clip_ratio (float): Clipping parameter for PPO
            policy_lr (float): Learning rate for the policy network
            value_lr (float): Learning rate for the value network
            learning_rate_decay (float): Decay rate for learning rate
            learning_rate_min (float): Minimum value for learning rate
            learning_rate_decay_steps (int): Steps for learning rate decay
            input_dim (int): Dimension of the input features
            output_dim (int): Dimension of the output (number of actions)
            hidden_dims (tuple): Dimensions of the hidden layers
            actions (list): List of possible actions
            gae_lambda (float): Lambda parameter for GAE
            entropy_coef (float): Coefficient for entropy bonus
            value_coef (float): Coefficient for value function loss
            ppo_epochs (int): Number of epochs to update policy for each batch
            mini_batch_size (int): Mini-batch size for updates
            device (str): Device to run the model on ('cpu' or 'cuda')
            input_encoding (bool): Whether to encode input into Fourier features
            input_encoding_dim (int): Dimension of the Fourier features
        """
        super().__init__()
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_min = learning_rate_min
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.device = device
        self.actions = actions
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.input_encoding = input_encoding
        self.input_encoding_dim = input_encoding_dim
        self.use_max_action = False

        # Convert any nested lists within actions to tuples
        if all(isinstance(a, list) for a in actions):
            self.actions = [tuple(action) for action in actions]
        else:
            self.actions = actions

        # Create policy and value networks
        if self.input_encoding:
            feature_extractor = InputEncoder(input_dim, input_encoding_dim + input_dim, append_initial_input=True)
            feature_dim = input_encoding_dim + input_dim

            self.policy_network = torch.nn.Sequential(
                feature_extractor,
                MLP(feature_dim, output_dim, hidden_dims),
            ).to(device)

            self.value_network = torch.nn.Sequential(
                InputEncoder(input_dim, input_encoding_dim + input_dim, append_initial_input=True),
                MLP(feature_dim, 1, hidden_dims),
            ).to(device)
        else:
            self.policy_network = MLP(input_dim, output_dim, hidden_dims).to(device)
            self.value_network = MLP(input_dim, 1, hidden_dims).to(device)

        # Setup optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=value_lr)

        # Setup learning rate schedulers
        self.policy_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.policy_optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay
        )
        self.value_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.value_optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay
        )

        self._episode_counter = 0

    def eval(self) -> None:
        """
        Set the agent to evaluation mode.
        """
        self.policy_network.eval()
        self.value_network.eval()
        self.use_max_action = True

    def train(self) -> None:
        """
        Set the agent to training mode.
        """
        self.policy_network.train()
        self.value_network.train()
        self.use_max_action = False

    def buffer_tuple_to_dict(self, buffer: tuple) -> dict:
        """
        Convert a buffer tuple to a dictionary.

        Args:
            buffer (tuple): Tuple containing states, actions, rewards, next states, dones, and log probs
        """
        states, actions, rewards, next_states, dones, log_probs = buffer
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "log_probs": log_probs,
        }

    def compute_advantages(self, rewards, values, next_values, dones):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            rewards: Tensor of rewards [batch_size]
            values: Tensor of state values [batch_size]
            next_values: Tensor of next state values [batch_size]
            dones: Tensor of episode termination flags [batch_size]

        Returns:
            advantages: Tensor of advantages [batch_size]
            returns: Tensor of returns [batch_size]
        """
        # Calculate deltas: r + gamma * V(s') - V(s)
        deltas = rewards + self.gamma * next_values * (1 - dones) - values

        # Calculate advantages using GAE
        advantages = torch.zeros_like(deltas)
        gae = 0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        # Calculate returns
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, buffer: dict) -> dict[str, float]:
        """
        Update the policy and value networks using PPO algorithm.

        Args:
            buffer (dict): Buffer containing states, actions, rewards, next states, dones, and log probs

        Returns:
            dict[str, float]: Average loss value from the update
        """
        states = torch.tensor(buffer["states"], dtype=torch.float32).to(self.device)
        actions = torch.tensor([self.action_to_index(action) for action in buffer["actions"]], dtype=torch.long).to(
            self.device
        )
        rewards = torch.tensor(buffer["rewards"], dtype=torch.float32).to(self.device).view(-1, 1)
        next_states = torch.tensor(buffer["next_states"], dtype=torch.float32).to(self.device)
        dones = torch.tensor(buffer["dones"], dtype=torch.float32).to(self.device).view(-1, 1)
        old_log_probs = torch.tensor(buffer["log_probs"], dtype=torch.float32).to(self.device).view(-1, 1)

        # Compute current values
        with torch.no_grad():
            values = self.value_network(states)
            next_values = self.value_network(next_states)

        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, values, next_values, dones)

        # PPO update loop
        total_actor_loss = 0
        total_critic_loss = 0

        for _ in range(self.ppo_epochs):
            # Generate random indices for mini-batches
            indices = torch.randperm(states.size(0))

            # Process mini-batches
            for start_idx in range(0, states.size(0), self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, states.size(0))
                idx = indices[start_idx:end_idx]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                # Get current policy distribution
                logits = self.policy_network(batch_states)
                dist = torch.distributions.Categorical(logits=logits)

                # Get current log probs
                batch_new_log_probs = dist.log_prob(batch_actions).unsqueeze(1)

                # Calculate entropy
                entropy = dist.entropy().mean()

                # Calculate ratio between new and old policies
                ratio = torch.exp(batch_new_log_probs - batch_old_log_probs)

                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Calculate value loss
                value_pred = self.value_network(batch_states)
                value_loss = F.mse_loss(value_pred, batch_returns)

                # Calculate total loss
                # loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                actor_loss = policy_loss - self.entropy_coef * entropy
                critic_loss = self.value_coef * value_loss

                # Update policy network
                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                self.policy_optimizer.step()

                # Update value network
                self.value_optimizer.zero_grad()
                critic_loss.backward()
                self.value_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        # Calculate average loss
        avg_actor_loss = total_actor_loss / (self.ppo_epochs * (states.size(0) // self.mini_batch_size + 1))
        avg_critic_loss = total_critic_loss / (self.ppo_epochs * (states.size(0) // self.mini_batch_size + 1))

        return {"actor_loss": avg_actor_loss, "critic_loss": avg_critic_loss}

    def act(self, state: tuple) -> float | tuple:
        """
        Select an action based on the current state.

        Args:
            state (tuple): Current state of the environment

        Returns:
            float | tuple: Selected action
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).view(1, -1)
            logits = self.policy_network(state_tensor).squeeze(0)
            if self.use_max_action:
                action_id = torch.argmax(logits).item()
                log_prob = logits[action_id].item()
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action_id = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action_id, device=self.device)).item()

        action = self.index_to_action(action_id)

        # Store log probability for PPO update
        self._last_log_prob = log_prob

        return action

    def get_log_prob(self) -> float:
        """
        Return the log probability of the last action.

        Returns:
            float: Log probability of the last action
        """
        return self._last_log_prob

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
        End the current episode and update learning rates.
        """
        # Update learning rates
        if self.policy_lr > self.learning_rate_min:
            self.policy_lr_scheduler.step()
            self.policy_lr = self.policy_lr_scheduler.get_last_lr()[0]

        if self.value_lr > self.learning_rate_min:
            self.value_lr_scheduler.step()
            self.value_lr = self.value_lr_scheduler.get_last_lr()[0]

        self._episode_counter += 1

    @classmethod
    def from_config(cls, config: dict) -> "PPOAgent":
        """
        Create a PPOAgent instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing agent parameters

        Returns:
            PPOAgent: Instance of the PPOAgent class
        """
        # Define default values for configuration parameters
        default_config = {
            "gamma": 0.99,
            "clip_ratio": 0.2,
            "policy_lr": 0.0003,
            "value_lr": 0.001,
            "learning_rate_decay": 1.0,
            "learning_rate_min": 0.0001,
            "learning_rate_decay_steps": 100,
            "input_dim": 2,
            "output_dim": 1,
            "hidden_dims": [64, 64],
            "actions": [0.0],
            "gae_lambda": 0.95,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "ppo_epochs": 4,
            "mini_batch_size": 64,
            "device": "cpu",
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
            clip_ratio=agent_config["clip_ratio"],
            policy_lr=agent_config["policy_lr"],
            value_lr=agent_config["value_lr"],
            learning_rate_decay=agent_config["learning_rate_decay"],
            learning_rate_min=agent_config["learning_rate_min"],
            learning_rate_decay_steps=agent_config["learning_rate_decay_steps"],
            input_dim=agent_config["input_dim"],
            output_dim=agent_config["output_dim"],
            hidden_dims=tuple(agent_config["hidden_dims"]),
            actions=agent_config["actions"],
            gae_lambda=agent_config["gae_lambda"],
            entropy_coef=agent_config["entropy_coef"],
            value_coef=agent_config["value_coef"],
            ppo_epochs=agent_config["ppo_epochs"],
            mini_batch_size=agent_config["mini_batch_size"],
            device=agent_config["device"],
            input_encoding=agent_config["input_encoding"],
            input_encoding_dim=agent_config["input_encoding_dim"],
        )

    def save_checkpoint(self, filename: str) -> None:
        """
        Save the agent's checkpoint.

        Args:
            filename (str): Filename to save the checkpoint
        """
        checkpoint = {
            "policy_state_dict": self.policy_network.state_dict(),
            "value_state_dict": self.value_network.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename: str) -> None:
        """
        Load the agent's checkpoint.

        Args:
            filename (str): Filename to load the checkpoint from
        """
        checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
        self.policy_network.load_state_dict(checkpoint["policy_state_dict"])
        self.value_network.load_state_dict(checkpoint["value_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self.policy_network.to(self.device)
        self.value_network.to(self.device)
