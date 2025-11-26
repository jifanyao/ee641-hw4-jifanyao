"""
Neural network models for multi-agent DQN with communication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AgentDQN(nn.Module):
    """
    Deep Q-Network for agent with communication capability.

    Network processes observations (11-dim) and outputs both Q-values (5 actions) 
    and a communication signal (scalar in [0,1]).
    """

    def __init__(self, input_dim: int = 11, hidden_dim: int = 64, num_actions: int = 5):
        """
        Initialize DQN with dual outputs.
        """
        super(AgentDQN, self).__init__()

        # Shared Feature Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Shared Feature Layer 2 (used as h)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Action head: outputs Q-values for each action
        self.action_head = nn.Linear(hidden_dim, num_actions)

        # Communication head: outputs single scalar before sigmoid
        self.comm_head = nn.Linear(hidden_dim, 1)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.
        
        Returns:
            action_values: Q-values for each action [batch_size, num_actions]
            comm_signal: Communication signal in [0,1] [batch_size, 1]
        """
        # Pass input through shared feature layers
        h1 = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h1)) # Hidden representation 'h'

        # Compute Q-values (action_values)
        action_values = self.action_head(h)

        # Compute communication signal (Sigmoid activation for [0, 1] range)
        comm_signal = torch.sigmoid(self.comm_head(h))

        return action_values, comm_signal


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network for agent with communication capability.
    
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """

    def __init__(self, input_dim: int = 11, hidden_dim: int = 64, num_actions: int = 5):
        """
        Initialize Dueling DQN.
        """
        super(DuelingDQN, self).__init__()

        # Shared feature layer (h)
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream: V(s)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Advantage stream: A(s,a)
        self.advantage_head = nn.Linear(hidden_dim, num_actions)
        
        # Communication head: c_out
        self.comm_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dueling architecture.
        
        Returns:
            Q_values: Q-values for each action [batch_size, num_actions]
            comm_signal: Communication signal in [0,1] [batch_size, 1]
        """
        features = self.feature_layer(x) # Hidden representation 'h'

        # Compute state value V(s)
        value = self.value_head(features) # [batch_size, 1]

        # Compute advantages A(s,a)
        advantage = self.advantage_head(features) # [batch_size, num_actions]

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Compute communication signal
        comm_signal = torch.sigmoid(self.comm_head(features))

        return q_values, comm_signal