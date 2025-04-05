import torch
import numpy as np
import torch.nn as nn


class InputEncoder(nn.Module):
    """
    A network that encodes N-dimensional input into positional feature vectors using
    sine and cosine functions at different frequencies.
    """

    def __init__(self, input_dim, output_dim, initial_freq_log2=0, append_initial_input=False):
        """
        Initialize the PositionalEncoder.

        Args:
            input_dim (int): Dimension of the input features
            output_dim (int, optional): Desired output dimension, should be multiple of 2 * input_dim
                if append_initial_input is False, otherwise k * (2 * input_dim) + input_dim.
            initial_freq_log2 (int, optional): Initial frequency log2 value for the sine/cosine functions.
                Default is 0.
            append_initial_input (bool, optional): If True, appends the original input to the output.
        """
        super(InputEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initial_freq_log2 = initial_freq_log2
        self.append_initial_input = append_initial_input

        if append_initial_input:
            assert (output_dim - input_dim) % (
                2 * input_dim
            ) == 0, "Output dimension must be a multiple of 2 * input_dim plus input_dim."
        else:
            assert output_dim % (2 * input_dim) == 0, "Output dimension must be a multiple of 2 * input_dim."

        # Calculate max_freq_log2 based on output_dim
        # Since each input dimension generates 2*(max_freq_log2+1) output dimensions
        # We solve for max_freq_log2 from the equation: output_dim = 2*input_dim*(max_freq_log2+1)
        self.max_freq_log2 = (self.output_dim // (2 * input_dim)) - 1

        # Pre-compute frequency bands
        self.register_buffer(
            "freq_bands", 2.0 ** (torch.arange(0, self.max_freq_log2 + 1) + self.initial_freq_log2) * np.pi
        )

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Encoded features of shape (batch_size, output_dim)
        """
        # Ensure x is 2D with shape (batch_size, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        assert x.shape[1] == self.input_dim, f"Input tensor must have shape (batch_size, {self.input_dim})."

        # Reshape for broadcasting:
        # x: (batch_size, input_dim) -> (batch_size, input_dim, 1)
        # freq_bands: (max_freq_log2+1) -> (1, 1, max_freq_log2+1)
        x_expanded = x.unsqueeze(-1)
        freq_bands_expanded = self.freq_bands.reshape(1, 1, -1)

        # Project input to different frequency bands
        # Result shape: (batch_size, input_dim, max_freq_log2+1)
        projection = x_expanded * freq_bands_expanded

        # Compute sin and cos features
        sin_features = torch.sin(projection)
        cos_features = torch.cos(projection)

        # Concatenate sin and cos features along the last dimension
        # Shape: (batch_size, input_dim, 2*(max_freq_log2+1))
        encoded = torch.cat([sin_features, cos_features], dim=-1)

        # Reshape to (batch_size, input_dim * 2 * (max_freq_log2+1))
        encoded = encoded.reshape(batch_size, -1)

        if self.append_initial_input:
            # Append the original input to the encoded features
            # Shape: (batch_size, input_dim * 2 * (max_freq_log2+1) + input_dim)
            encoded = torch.cat([encoded, x], dim=-1)

        return encoded
