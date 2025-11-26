"""
Value Iteration algorithm for solving MDPs.
"""

import numpy as np
from typing import Tuple, Optional
from environment import GridWorldEnv


class ValueIteration:
    """
    Value Iteration solver for gridworld MDP.

    Computes optimal value function V* using dynamic programming.
    """

    def __init__(self, env: GridWorldEnv, gamma: float = 0.95, epsilon: float = 1e-4):
        """
        Initialize Value Iteration solver.

        Args:
            env: GridWorld environment
            gamma: Discount factor
            epsilon: Convergence threshold
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = env.grid_size ** 2
        self.n_actions = env.action_space
        self.error_history = [] # To store max Bellman error at each iteration

    def solve(self, max_iterations: int = 1000) -> Tuple[np.ndarray, int]:
        """
        Run value iteration until convergence.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            values: Converged value function V(s)
            n_iterations: Number of iterations until convergence
        """
        # Initialize value function to zeros
        values = np.zeros(self.n_states)
        n_iterations = 0
        self.error_history = [] # Reset history
        
        # Iterate until convergence
        for i in range(max_iterations):
            n_iterations = i + 1
            values_old = values.copy()
            values_new = np.zeros(self.n_states)
            
            # For each state:
            for s in range(self.n_states):
                # Compute V(s) = max_a BellmanBackup(s, a)
                values_new[s] = self.bellman_backup(s, values_old)
                
            # Update value function
            values = values_new
            
            # Check convergence: max|V_new - V_old| < epsilon
            max_error = np.max(np.abs(values - values_old))
            self.error_history.append(max_error)
            
            if max_error < self.epsilon:
                break

        # Return final values and iteration count
        return values, n_iterations

    def compute_q_values(self, state: int, values: np.ndarray) -> np.ndarray:
        """
        Compute Q-values for all actions in a state.

        Args:
            state: State index
            values: Current value function

        Returns:
            q_values: Array of Q(s,a) for each action
        """
        q_values = np.zeros(self.n_actions)
        
        # If terminal state, Q(s,a) is 0 for all actions
        if self.env.is_terminal(state):
            return q_values
            
        # For each action:
        for a in range(self.n_actions):
            expected_return = 0.0
            # Get transition probabilities P(s'|s,a)
            transitions = self.env.get_transition_prob(state, a)
            
            # Compute expected value: Q(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
            for next_state, prob in transitions.items():
                if prob > 0:
                    # Get reward R(s,a,s')
                    reward = self.env.get_reward(state, a, next_state)
                    
                    # Get V(s')
                    next_value = values[next_state]
                    
                    # Accumulate: prob * [R(s,a,s') + gamma * V(s')]
                    expected_return += prob * (reward + self.gamma * next_value)
            
            q_values[a] = expected_return
            
        # Return Q-values array
        return q_values

    def extract_policy(self, values: np.ndarray) -> np.ndarray:
        """
        Extract optimal policy from value function.

        Args:
            values: Optimal value function

        Returns:
            policy: Array of optimal actions for each state
        """
        policy = np.zeros(self.n_states, dtype=np.int32)
        
        # For each state:
        for s in range(self.n_states):
            # If terminal state, policy action doesn't matter (set to 0)
            if self.env.is_terminal(s):
                policy[s] = 0 
                continue
                
            # Compute Q-values for all actions
            q_values = self.compute_q_values(s, values)
            
            # Select action with maximum Q-value: argmax_a Q(s,a)
            # Use np.argmax to get the optimal action.
            optimal_action = np.argmax(q_values)
            policy[s] = optimal_action
            
        # Return policy array
        return policy

    def bellman_backup(self, state: int, values: np.ndarray) -> float:
        """
        Perform Bellman backup for a single state.

        Args:
            state: State index
            values: Current value function

        Returns:
            Updated value for state
        """
        # If terminal state, return 0
        if self.env.is_terminal(state):
            return 0.0
        
        # Compute Q-values for all actions
        q_values = self.compute_q_values(state, values)
        
        # Return maximum Q-value
        return np.max(q_values)

    def compute_bellman_error(self, values: np.ndarray) -> float:
        """
        Compute Bellman error for current value function.

        Bellman error = max_s |V(s) - max_a Q(s,a)|

        Args:
            values: Current value function

        Returns:
            Maximum Bellman error across all states
        """
        max_error = 0.0
        
        # For each state:
        for s in range(self.n_states):
            # Compute optimal value using Bellman backup: max_a Q(s,a)
            # Note: For terminal states, V(s)=0 and bellman_backup returns 0, so error is 0.
            optimal_value = self.bellman_backup(s, values)
            
            # Calculate absolute difference from current value: |V(s) - max_a Q(s,a)|
            current_value = values[s]
            error = np.abs(current_value - optimal_value)
            
            # Update maximum error
            max_error = max(max_error, error)
                
        # Return maximum error
        return max_error