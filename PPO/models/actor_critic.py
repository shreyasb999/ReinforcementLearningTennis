# Actor-Critic network module
# Defines the shared architecture for both actor and critic networks

import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, output_gate=torch.tanh):
        """
        Initializes the ActorCritic network.
        
        Args:
            in_dim (int): Input dimension (state size)
            hidden_dim (int): Hidden layer dimension
            out_dim (int): Output dimension (action size for actor or 1 for critic)
            output_gate (callable, optional): Activation function for the output layer
        """
        super().__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer
        self.head = nn.Linear(hidden_dim, out_dim)
        # Output activation function (e.g., tanh for actor, None for critic)
        self.output_gate = output_gate

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input state tensor
        
        Returns:
            Tensor: Output tensor after passing through network layers
        """
        x = F.relu(self.fc1(x))   # Apply ReLU after first layer
        x = F.relu(self.fc2(x))   # Apply ReLU after second layer
        x = self.head(x)          # Final linear layer
        if self.output_gate is not None:
            x = self.output_gate(x)  # Apply output activation if provided
        return x
