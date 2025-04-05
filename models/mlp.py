import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) network implementation.

    This MLP can be used for various tasks including reinforcement learning
    policies, value functions, or general function approximation.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=(64, 64),
        activation_fn=F.relu,
        output_activation=None,
    ):
        """
        Initialize the MLP network.

        Args:
            input_dim (int): Dimension of the input features
            output_dim (int): Dimension of the output
            hidden_dims (tuple): Dimensions of the hidden layers
            activation_fn (callable): Activation function for hidden layers
            output_activation (callable, optional): Activation function for output layer
        """
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation_fn = activation_fn
        self.output_activation = output_activation

        # Build the network architecture
        self.layers = nn.ModuleList()

        # Input layer
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Ensure input is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Forward through hidden layers
        for layer in self.layers:
            x = self.activation_fn(layer(x))

        # Output layer
        x = self.output_layer(x)

        # Apply output activation if specified
        if self.output_activation is not None:
            x = self.output_activation(x)

        return x

    def save(self, path):
        """Save the model parameters to disk."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model parameters from disk."""
        self.load_state_dict(torch.load(path))
