"""
Experience replay buffer for multi-agent DQN training.
"""

import numpy as np
import random
from typing import Tuple, List, Optional
from collections import deque
import torch


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    Stores joint experiences from both agents for coordinated learning.
    Transition tuple: (sA, sB, aA, aB, cA, cB, r, sA', sB', done)
    """

    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        """
        Initialize replay buffer.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
    def __len__(self) -> int:
        """
        Return the current size of the buffer.
        """
        return len(self.buffer)

    def push(self, state_A: np.ndarray, state_B: np.ndarray,
             action_A: int, action_B: int,
             comm_A: float, comm_B: float,
             reward: float,
             next_state_A: np.ndarray, next_state_B: np.ndarray,
             done: bool) -> None:
        """
        Store a transition in the buffer.
        """
        transition = (state_A, state_B, action_A, action_B, 
                      comm_A, comm_B, reward, 
                      next_state_A, next_state_B, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Sample a batch of transitions and convert to PyTorch tensors.

        Returns: 
            Tuple of 10 tensors (sA, sB, aA, aB, cA, cB, r, sA', sB', done)
        """
        if len(self.buffer) < batch_size:
            return None

        transitions = random.sample(self.buffer, batch_size)
        
        # Unzip the batch of transitions
        state_A, state_B, action_A, action_B, comm_A, comm_B, reward, \
            next_state_A, next_state_B, done = zip(*transitions)

        # Convert to numpy arrays, then to PyTorch tensors
        state_A = torch.from_numpy(np.stack(state_A)).float()
        state_B = torch.from_numpy(np.stack(state_B)).float()
        next_state_A = torch.from_numpy(np.stack(next_state_A)).float()
        next_state_B = torch.from_numpy(np.stack(next_state_B)).float()

        # Discrete Actions: long/int64
        action_A = torch.tensor(action_A, dtype=torch.long).unsqueeze(1)
        action_B = torch.tensor(action_B, dtype=torch.long).unsqueeze(1)

        # Communication, Reward, Done: float32
        comm_A = torch.tensor(comm_A, dtype=torch.float).unsqueeze(1)
        comm_B = torch.tensor(comm_B, dtype=torch.float).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float).unsqueeze(1) 

        return (state_A, state_B, action_A, action_B, comm_A, comm_B, reward, 
                next_state_A, next_state_B, done)

# Placeholder for PrioritizedReplayBuffer (for interface compliance if needed)
class PrioritizedReplayBuffer(ReplayBuffer):
    pass