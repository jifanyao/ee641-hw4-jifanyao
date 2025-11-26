"""
Stochastic gridworld environment for reinforcement learning.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict


class GridWorldEnv:
    """
    5x5 Stochastic GridWorld Environment.

    The agent navigates a grid with stochastic transitions:
    - 0.8 probability of moving in the intended direction
    - 0.1 probability of drifting left (perpendicular)
    - 0.1 probability of drifting right (perpendicular)

    Grid layout:
    - Start: (0, 0)
    - Goal: (4, 4)
    - Obstacles: (1, 2), (2, 1)
    - Penalties: (3, 3), (3, 0)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize gridworld environment.

        Args:
            seed: Random seed for reproducibility
        """
        self.grid_size = 5
        self.max_steps = 50

        # Define special cells (Row, Col)
        # Note: The provided snippet had a conflict with the diagram (Image 1/Page 2).
        # The code snippet used: obstacles = [(1, 2), (2, 1)], penalties = [(3, 3), (3, 0)]
        # Based on the image and standard practice, I will use:
        # Obstacles (X): (1, 2), (2, 1)
        # Penalties (P): (3, 3), (3, 0)
        self.start_pos = (0, 0) # S
        self.goal_pos = (4, 4) # G
        self.obstacles = [(1, 2), (2, 1)] # X at (1,2) and (2,1)
        self.penalties = [(3, 3), (3, 0)] # P at (3,3) and (3,0)

        # Rewards
        self.goal_reward = 10.0 # +10 reward
        self.penalty_reward = -5.0 # -5 reward
        self.step_cost = -0.1 # -0.1 cost per step

        # Transition probabilities
        self.prob_intended = 0.8 # 0.8 probability: Move in intended direction
        self.prob_drift = 0.1 # 0.1 probability: Drift perpendicular left/right

        # Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_space = 4
        self.action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        
        # Action deltas: (row_change, col_change)
        self.action_deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        if seed is not None:
            np.random.seed(seed)

        self.agent_pos = None
        self.current_state = None
        self.n_steps = 0
        self.done = False
        
        self.reset()

    def reset(self) -> int:
        """
        Reset environment to initial state.

        Returns:
            state: Initial state index
        """
        # Initialize agent position to start_pos
        self.agent_pos = self.start_pos
        self.current_state = self._pos_to_state(self.agent_pos)
        # Reset step counter
        self.n_steps = 0
        # Set done flag to False
        self.done = False
        # Return state index (use _pos_to_state)
        return self.current_state

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Execute action in environment.

        Args:
            action: Action index (0-3)

        Returns:
            next_state: Next state index
            reward: Reward received
            done: Whether episode terminated
            info: Additional information
        """
        # Check if episode already done
        if self.done:
            return self.current_state, 0.0, True, {}

        # Get next position based on stochastic transitions
        possible_transitions = self._get_next_positions(self.agent_pos, action)
        
        # Sample next position based on probabilities
        
        # Consolidate and prepare for np.random.choice
        unique_next_pos = {}
        for pos, prob in possible_transitions:
            unique_next_pos[pos] = unique_next_pos.get(pos, 0.0) + prob
        
        positions = list(unique_next_pos.keys())
        probs = list(unique_next_pos.values())
        
        # Choose next position
        if not positions:
            next_pos = self.agent_pos # Should not happen, but safe fallback
        else:
            # Re-normalize just in case of slight floating point error, though it should sum to 1
            if not np.isclose(sum(probs), 1.0):
                 probs = np.array(probs) / sum(probs)
                 
            next_pos_idx = np.random.choice(len(positions), p=probs)
            next_pos = positions[next_pos_idx]


        next_state = self._pos_to_state(next_pos)
        
        # Calculate reward (R(s,a,s') logic based on next_state/pos)
        reward = self.get_reward(self.current_state, action, next_state)

        # Update position and step count
        self.agent_pos = next_pos
        self.current_state = next_state
        self.n_steps += 1
        
        # Check termination conditions
        # Terminates if goal reached, or step limit reached
        is_goal = next_pos == self.goal_pos
        is_max_steps = self.n_steps >= self.max_steps
        self.done = is_goal or is_max_steps
        
        # Return (next_state, reward, done, info)
        info = {"pos": next_pos, "steps": self.n_steps}
        return next_state, reward, self.done, info

    def get_transition_prob(self, state: int, action: int) -> Dict[int, float]:
        """
        Get transition probabilities P(s'|s,a).

        Args:
            state: Current state index
            action: Action index

        Returns:
            Dictionary mapping next_state -> probability
        """
        if self.is_terminal(state):
            # If in terminal state (goal), probability of staying is 1.
            return {state: 1.0}

        # Convert state to position
        pos = self._state_to_pos(state)
        
        # For given action, compute all possible next positions considering stochastic transitions
        possible_transitions = self._get_next_positions(pos, action)
        
        # Handle state conversion and merging probabilities for same state
        transition_probs = {}
        for next_pos, prob in possible_transitions:
            next_state = self._pos_to_state(next_pos)
            transition_probs[next_state] = transition_probs.get(next_state, 0.0) + prob

        # Return probability distribution over next states
        return transition_probs

    def get_reward(self, state: int, action: int, next_state: int) -> float:
        """
        Get reward for transition.

        Args:
            state: Current state index
            action: Action taken
            next_state: Resulting state

        Returns:
            Reward value
        """
        # Convert next_state to position
        next_pos = self._state_to_pos(next_state)
        
        # Check if goal reached (+10)
        if next_pos == self.goal_pos:
            return self.goal_reward # +10
        
        # Check if penalty cell (-5)
        if next_pos in self.penalties:
            return self.penalty_reward # -5
        
        # Otherwise return step cost (-0.1)
        return self.step_cost # -0.1

    def is_terminal(self, state: int) -> bool:
        """
        Check if state is terminal.

        Args:
            state: State index

        Returns:
            True if terminal state
        """
        # Convert state to position
        pos = self._state_to_pos(state)
        
        # Return True if position equals goal_pos
        return pos == self.goal_pos # Goal terminates episode

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """
        Convert grid position to state index.

        Args:
            pos: (row, col) position

        Returns:
            State index (0-24)
        """
        # Convert 2D position to 1D state index
        row, col = pos
        # State = row * grid_size + col
        return row * self.grid_size + col

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """
        Convert state index to grid position.

        Args:
            state: State index

        Returns:
            (row, col) position
        """
        # Convert 1D state index to 2D position
        # row = state // grid_size
        row = state // self.grid_size
        # col = state % grid_size
        col = state % self.grid_size
        return (row, col)

    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is valid (in bounds and not obstacle).

        Args:
            pos: (row, col) position

        Returns:
            True if valid position
        """
        row, col = pos
        # Check if position is within grid bounds
        in_bounds = (0 <= row < self.grid_size) and (0 <= col < self.grid_size)
        
        # Check if position is not an obstacle
        not_obstacle = pos not in self.obstacles # Obstacles are impassable

        return in_bounds and not_obstacle

    def _get_next_positions(self, pos: Tuple[int, int], action: int) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get possible next positions and probabilities for stochastic transition.

        Args:
            pos: Current position
            action: Action to take

        Returns:
            List of (next_position, probability) tuples
        """
        if self.is_terminal(self._pos_to_state(pos)):
            # If in terminal state (goal), stay there with probability 1
            return [(pos, 1.0)]

        # Define action effects (deltas for UP, RIGHT, DOWN, LEFT)
        deltas = self.action_deltas # [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # Get intended direction (0.8 prob)
        intended_delta = deltas[action]
        
        # Get perpendicular directions (0.1 prob each)
        # Left drift is (action - 1) % 4
        # Right drift is (action + 1) % 4
        drift_left_delta = deltas[(action - 1) % 4]
        drift_right_delta = deltas[(action + 1) % 4]

        # Outcomes: (delta, probability)
        outcomes = [
            (intended_delta, self.prob_intended), # 0.8 intended
            (drift_left_delta, self.prob_drift), # 0.1 drift left
            (drift_right_delta, self.prob_drift) # 0.1 drift right
        ]
        
        next_pos_probs = []
        
        # Map to actual next positions
        for (dr, dc), prob in outcomes:
            next_r, next_c = pos[0] + dr, pos[1] + dc
            next_pos = (next_r, next_c)
            
            # If invalid (wall or obstacle), agent stays in current position
            if not self._is_valid_pos(next_pos):
                next_pos = pos
            
            next_pos_probs.append((next_pos, prob))

        # Merge probabilities for same positions (in case of collisions)
        merged_probs = {}
        for next_pos, prob in next_pos_probs:
            merged_probs[next_pos] = merged_probs.get(next_pos, 0.0) + prob
            
        # Convert back to list of tuples
        result = list(merged_probs.items())
        
        return result

    def _calculate_reward(self, pos: Tuple[int, int]) -> float:
        """
        Calculate reward for entering a position. (Helper, not critical for DP)

        Args:
            pos: Position entered

        Returns:
            Reward value
        """
        # This function is redundant given get_reward(s,a,s') where s' corresponds to pos
        if pos == self.goal_pos:
            return self.goal_reward
        if pos in self.penalties:
            return self.penalty_reward
        return self.step_cost

    def render(self, value_function: Optional[np.ndarray] = None) -> None:
        """
        Render current state of environment.
        """
        # Not required for core assignment, leaving as not implemented.
        raise NotImplementedError