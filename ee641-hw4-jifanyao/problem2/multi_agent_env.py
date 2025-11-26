"""
Multi-agent gridworld environment with partial observations and communication.
"""

import numpy as np
from typing import Tuple, Optional, List
import math


class MultiAgentEnv:
    """
    Two-agent cooperative gridworld with partial observations.

    Agents must coordinate to simultaneously reach a target cell.
    Each agent observes a 3x3 local patch and exchanges communication signals.
    """
    # Actions: 0:Up, 1:Down, 2:Left, 3:Right, 4:Stay
    action_map = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
        4: (0, 0)    # Stay
    }
    num_actions = 5

    def __init__(self, grid_size: Tuple[int, int] = (10, 10), obs_window: int = 3,
                 max_steps: int = 50, seed: Optional[int] = None):
        """
        Initialize multi-agent environment.
        """
        self.grid_size = grid_size
        self.obs_window = obs_window
        self.max_steps = max_steps
        self.H, self.W = grid_size
        self.target_pos = None

        if seed is not None:
            np.random.seed(seed)
            self.seed = seed

        # Calculate max L2 distance for normalization
        self.max_l2_dist = math.sqrt((self.H - 1)**2 + (self.W - 1)**2)

        # Initialize grid components
        self._initialize_grid()

        # Agent state
        # --- FIX START: Ensure positions are valid tuples upon initialization ---
        # Original: self.agent_positions = [None, None]
        # FIX: Use the default reset positions (2, 1) and (6, 4) for initialization
        self.agent_positions = [(2, 1), (6, 4)]
        # --- FIX END ---
        self.comm_signals = [0.0, 0.0]  # [comm_A_to_B, comm_B_to_A]
        self.step_count = 0
        self.reward = 0.0 

    def _initialize_grid(self) -> None:
        """
        Create grid with obstacles and target based on Problem 2 image.

        Grid values:
        0: Free cell
        1: Obstacle (X)
        2: Target (T)
        """
        self.grid = np.zeros(self.grid_size, dtype=np.int32)

        # Obstacle positions (Row, Col) from the PDF image
        self.obstacle_positions = [
            (2, 3), (2, 4),
            (4, 1), 
            (5, 6), 
            (7, 2), 
            (8, 7)
        ]
        # Target position (bottom-right corner)
        self.target_pos = (self.H - 1, self.W - 1) # (9, 9)

        # Place obstacles
        for r, c in self.obstacle_positions:
            if 0 <= r < self.H and 0 <= c < self.W:
                 self.grid[r, c] = 1
            
        # Place target
        self.grid[self.target_pos] = 2

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not an obstacle."""
        r, c = pos
        if not (0 <= r < self.H and 0 <= c < self.W):
            return False
        if self.grid[r, c] == 1: # 1 is Obstacle
            return False
        return True

    def _move_agent(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """ Apply a movement action and return the new position. """
        dr, dc = self.action_map.get(action, (0, 0))
        new_pos = (pos[0] + dr, pos[1] + dc)
        
        if self._is_valid_position(new_pos):
            return new_pos
        else:
            return pos # Stay put

    def _get_local_patch(self, pos: Tuple[int, int]) -> np.ndarray:
        """ 
        Extract the 3x3 local patch centered on the agent's position. 
        Pads with -1 (Off-grid) if out of bounds.
        """
        r, c = pos # <--- 错误发生在这里，因为 pos 之前是 None
        half_window = self.obs_window // 2
        H, W = self.H, self.W
        obs_patch = np.full((self.obs_window, self.obs_window), -1, dtype=np.float32)

        for i in range(self.obs_window):
            for j in range(self.obs_window):
                grid_r, grid_c = r - half_window + i, c - half_window + j
                
                if 0 <= grid_r < H and 0 <= grid_c < W:
                    obs_patch[i, j] = self.grid[grid_r, grid_c]
        
        # Cell values: 0 (free), 1 (obstacle), 2 (target), -1 (off-grid)
        return obs_patch.flatten()

    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """
        Construct the 11-dimensional observation vector.

        Elements: 0-8 (3x3 patch), 9 (partner comm in), 10 (normalized L2 dist).
        """
        pos_A = self.agent_positions[0]
        pos_B = self.agent_positions[1]
        
        if agent_idx == 0:
            my_pos = pos_A
            comm_in = self.comm_signals[1] # Comm from B
        else: # agent_idx == 1
            my_pos = pos_B
            comm_in = self.comm_signals[0] # Comm from A

        # 1. 3x3 Local Grid Patch (Elements 0-8)
        obs_patch = self._get_local_patch(my_pos)

        # 2. Normalized L2 Distance (Element 10)
        # Note: We can safely unpack here because self.agent_positions is guaranteed 
        # to be non-None tuples by the fix in __init__ and the line in reset
        r_A, c_A = pos_A
        r_B, c_B = pos_B
        l2_dist = math.sqrt((r_A - r_B)**2 + (c_A - c_B)**2)
        normalized_dist = l2_dist / self.max_l2_dist
        
        # Combine all components (11 dimensions)
        observation = np.concatenate([
            obs_patch, 
            np.array([comm_in]),       # Element 9: Comm from partner
            np.array([normalized_dist]) # Element 10: Normalized L2 distance
        ]).astype(np.float32)

        return observation

    def reset(self, start_A: Tuple[int, int]=(2, 1), start_B: Tuple[int, int]=(6, 4), target: Tuple[int, int]=(9, 9)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset environment to initial state, supporting custom start/target positions for generalization test.
        """
        self.step_count = 0
        self.reward = 0.0
        
        # Update target and re-initialize grid for generalization (if needed)
        if target != self.target_pos:
            self.target_pos = target
            self._initialize_grid()

        # Set initial positions (R, C)
        self.agent_positions = [start_A, start_B]
        self.comm_signals = [0.0, 0.0]
        
        obs_A = self._get_observation(0)
        obs_B = self._get_observation(1)
        
        return obs_A, obs_B

    def step(self, action_A: int, action_B: int, comm_A: float, comm_B: float) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool]:
        """
        Take a step in the environment.
        """
        # Note: reward and done should be checked before step_count increment if using max_steps for hard done
        # However, it's safer to check max_steps done at the beginning/end of the step

        self.step_count += 1

        # 1. Update positions
        next_pos_A = self._move_agent(self.agent_positions[0], action_A)
        next_pos_B = self._move_agent(self.agent_positions[1], action_B)
        self.agent_positions = [next_pos_A, next_pos_B]

        # 2. Update communication signals (outputs from this step become inputs for next observation)
        self.comm_signals = [
            np.clip(comm_A, 0.0, 1.0), # comm_A is used by B for next step
            np.clip(comm_B, 0.0, 1.0)  # comm_B is used by A for next step
        ]

        # 3. Compute Reward
        pos_A = self.agent_positions[0]
        pos_B = self.agent_positions[1]

        agent_A_at_target = (pos_A == self.target_pos)
        agent_B_at_target = (pos_B == self.target_pos)

        if agent_A_at_target and agent_B_at_target:
            reward = 10.0 # Both agents on target
        elif agent_A_at_target or agent_B_at_target:
            reward = 2.0 # One agent on target
        else:
            reward = -0.1 # Step penalty

        # 4. Check for Done
        done = (self.step_count >= self.max_steps) or (reward == 10.0)
        
        # 5. Get next observations
        obs_A_prime = self._get_observation(0)
        obs_B_prime = self._get_observation(1)
        
        # Store last step reward for render function
        self.reward = reward 

        return (obs_A_prime, obs_B_prime), reward, done

    def render(self) -> None:
        """
        Render current environment state.
        """
        grid_repr = np.full(self.grid_size, '.', dtype=str)
        
        # Grid content
        grid_repr[self.grid == 1] = 'X'  # Obstacle
        grid_repr[self.grid == 2] = 'T'  # Target

        # Agents 
        if self.agent_positions[0] is not None:
            grid_repr[self.agent_positions[0]] = 'A'
        if self.agent_positions[1] is not None:
            if self.agent_positions[0] == self.agent_positions[1]:
                grid_repr[self.agent_positions[1]] = 'B/A' # Both agents in the same cell
            else:
                grid_repr[self.agent_positions[1]] = 'B'

        print("-" * (self.W * 2 + 1))
        for row in grid_repr:
            print("|" + " ".join(row) + "|")
        print("-" * (self.W * 2 + 1))
        print(f"Step: {self.step_count}/{self.max_steps}, Reward: {self.reward:.1f}, Pos A: {self.agent_positions[0]}, Pos B: {self.agent_positions[1]}")
        print(f"Comm A->B: {self.comm_signals[0]:.4f}, Comm B->A: {self.comm_signals[1]:.4f}")