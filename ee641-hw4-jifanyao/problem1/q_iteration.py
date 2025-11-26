"""
Q-Iteration algorithm for solving MDPs.
"""

import numpy as np
from typing import Tuple, Optional
from environment import GridWorldEnv


class QIteration:
    """
    Q-Iteration solver for gridworld MDP.

    Computes optimal action-value function Q* using dynamic programming.
    """

    def __init__(self, env: GridWorldEnv, gamma: float = 0.95, epsilon: float = 1e-4):
        """
        Initialize Q-Iteration solver.

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
        Run Q-iteration until convergence.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            q_values: Converged Q-function Q(s,a)
            n_iterations: Number of iterations until convergence
        """
        # Initialize Q-function to zeros (shape: [n_states, n_actions])
        q_values = np.zeros((self.n_states, self.n_actions))
        n_iterations = 0
        self.error_history = [] # Reset history

        # Iterate until convergence
        for i in range(max_iterations):
            n_iterations = i + 1
            q_values_old = q_values.copy()
            q_values_new = np.zeros((self.n_states, self.n_actions))
            
            # For each state-action pair:
            for s in range(self.n_states):
                # Terminal states are handled by bellman_update logic
                for a in range(self.n_actions):
                    # Compute updated Q-value using Bellman equation
                    q_values_new[s, a] = self.bellman_update(s, a, q_values_old)
            
            # Check convergence: max|Q_new - Q_old| < epsilon
            max_error = np.max(np.abs(q_values_new - q_values_old))
            self.error_history.append(max_error)
            
            # Update Q-function
            q_values = q_values_new

            if max_error < self.epsilon:
                break
                
        # Return final Q-values and iteration count
        return q_values, n_iterations

    def bellman_update(self, state: int, action: int, q_values: np.ndarray) -> float:
        """
        Compute updated Q-value for a state-action pair.

        Args:
            state: State index
            action: Action index
            q_values: Current Q-function

        Returns:
            Updated Q-value for (s,a)
        """
        # If terminal state, Q-value is 0 for all actions
        if self.env.is_terminal(state):
            return 0.0

        updated_q = 0.0
        
        # Get transition probabilities P(s'|s,a)
        transitions = self.env.get_transition_prob(state, action)
        
        # For each possible next state:
        for next_state, prob in transitions.items():
            if prob > 0:
                # Get reward R(s,a,s')
                reward = self.env.get_reward(state, action, next_state)
                
                # Get max Q-value for next state: max_a' Q(s',a')
                # If next_state is terminal, max_a' Q(s',a') = 0.0
                if self.env.is_terminal(next_state):
                    max_q_next = 0.0
                else:
                    max_q_next = np.max(q_values[next_state, :])
                    
                # Accumulate: prob * [reward + gamma * max_q_next]
                # Q(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * max_a' Q(s',a')]
                updated_q += prob * (reward + self.gamma * max_q_next)
                
        # Return updated Q-value
        return updated_q

    def extract_policy(self, q_values: np.ndarray) -> np.ndarray:
        """
        Extract optimal policy from Q-function.

        Args:
            q_values: Optimal Q-function

        Returns:
            policy: Array of optimal actions for each state
        """
        # For each state:
        # Select action with maximum Q-value: argmax_a Q(s,a)
        # np.argmax performs the argmax over the last axis (axis=1)
        policy = np.argmax(q_values, axis=1)
        
        # Return policy array
        return policy

    def extract_values(self, q_values: np.ndarray) -> np.ndarray:
        """
        Extract value function from Q-function.

        Args:
            q_values: Q-function

        Returns:
            values: State value function V(s) = max_a Q(s,a)
        """
        # For each state:
        # Compute V(s) = max_a Q(s,a)
        # np.max performs the max over the last axis (axis=1)
        values = np.max(q_values, axis=1)
        
        # Return value function
        return values

    def compute_bellman_error(self, q_values: np.ndarray) -> float:
        """
        Compute Bellman error for current Q-function.

        Args:
            q_values: Current Q-function

        Returns:
            Maximum Bellman error across all state-action pairs
        """
        max_error = 0.0
        
        # For each state-action pair:
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # Compute updated Q-value using Bellman update
                updated_q = self.bellman_update(s, a, q_values)
                
                # Calculate absolute difference from current Q-value
                current_q = q_values[s, a]
                error = np.abs(current_q - updated_q)
                
                # Update maximum error
                max_error = max(max_error, error)
                    
        # Return maximum error
        return max_error