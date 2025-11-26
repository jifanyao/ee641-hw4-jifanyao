"""
Interface tests for Problem 1 implementations.

These tests verify that your implementations conform to the required interfaces.
They do not test correctness of algorithms, only that functions exist with
correct signatures and return appropriate types.
"""

import numpy as np
import pytest
import sys
from typing import Dict, Tuple

# Import student implementations
try:
    from environment import GridWorldEnv
    from value_iteration import ValueIteration
    from q_iteration import QIteration
except ImportError as e:
    print(f"Import error: {e}")
    print("Ensure all required files are in the current directory")
    sys.exit(1)


class TestEnvironmentInterface:
    """Test GridWorldEnv interface compliance."""

    def test_environment_initialization(self):
        """Test environment can be initialized with and without seed."""
        env1 = GridWorldEnv()
        assert env1.grid_size == 5
        assert env1.action_space == 4

        env2 = GridWorldEnv(seed=42)
        assert env2.grid_size == 5

    def test_reset_interface(self):
        """Test reset returns correct type."""
        env = GridWorldEnv(seed=123)
        state = env.reset()

        assert isinstance(state, (int, np.integer))
        assert 0 <= state < 25

    def test_step_interface(self):
        """Test step function interface."""
        env = GridWorldEnv(seed=123)
        env.reset()

        # Test valid action
        next_state, reward, done, info = env.step(0)

        assert isinstance(next_state, (int, np.integer))
        assert isinstance(reward, (float, np.floating))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_transition_probability_interface(self):
        """Test get_transition_prob returns correct type."""
        env = GridWorldEnv(seed=123)

        # Test for state 0, action 0
        transitions = env.get_transition_prob(0, 0)

        assert isinstance(transitions, dict)
        assert all(isinstance(k, (int, np.integer)) for k in transitions.keys())
        assert all(isinstance(v, (float, np.floating)) for v in transitions.values())

        # Probabilities should sum to 1
        total_prob = sum(transitions.values())
        assert abs(total_prob - 1.0) < 1e-6

    def test_reward_function_interface(self):
        """Test get_reward returns correct type."""
        env = GridWorldEnv(seed=123)

        reward = env.get_reward(0, 0, 1)
        assert isinstance(reward, (float, np.floating))

    def test_terminal_check_interface(self):
        """Test is_terminal returns correct type."""
        env = GridWorldEnv(seed=123)

        # Test non-terminal state
        is_term = env.is_terminal(0)
        assert isinstance(is_term, bool)

        # Test terminal state (goal)
        goal_state = 24  # (4, 4)
        is_term = env.is_terminal(goal_state)
        assert isinstance(is_term, bool)
        assert is_term == True

    def test_position_conversion_interfaces(self):
        """Test position/state conversion functions."""
        env = GridWorldEnv()

        # Test pos_to_state
        state = env._pos_to_state((0, 0))
        assert isinstance(state, (int, np.integer))
        assert state == 0

        state = env._pos_to_state((4, 4))
        assert state == 24

        # Test state_to_pos
        pos = env._state_to_pos(0)
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert pos == (0, 0)

        pos = env._state_to_pos(24)
        assert pos == (4, 4)


class TestValueIterationInterface:
    """Test ValueIteration interface compliance."""

    def test_value_iteration_initialization(self):
        """Test ValueIteration can be initialized."""
        env = GridWorldEnv(seed=123)
        vi = ValueIteration(env)

        assert vi.gamma == 0.95
        assert vi.epsilon == 1e-4
        assert vi.n_states == 25
        assert vi.n_actions == 4

    def test_solve_interface(self):
        """Test solve returns correct types."""
        env = GridWorldEnv(seed=123)
        vi = ValueIteration(env, gamma=0.9, epsilon=0.01)

        values, iterations = vi.solve(max_iterations=10)

        assert isinstance(values, np.ndarray)
        assert values.shape == (25,)
        assert isinstance(iterations, (int, np.integer))
        assert iterations > 0

    def test_compute_q_values_interface(self):
        """Test compute_q_values returns correct type."""
        env = GridWorldEnv(seed=123)
        vi = ValueIteration(env)

        values = np.zeros(25)
        q_values = vi.compute_q_values(0, values)

        assert isinstance(q_values, np.ndarray)
        assert q_values.shape == (4,)

    def test_extract_policy_interface(self):
        """Test extract_policy returns correct type."""
        env = GridWorldEnv(seed=123)
        vi = ValueIteration(env)

        values = np.random.rand(25)
        policy = vi.extract_policy(values)

        assert isinstance(policy, np.ndarray)
        assert policy.shape == (25,)
        assert policy.dtype in [np.int32, np.int64]
        assert all(0 <= a < 4 for a in policy)

    def test_bellman_backup_interface(self):
        """Test bellman_backup returns correct type."""
        env = GridWorldEnv(seed=123)
        vi = ValueIteration(env)

        values = np.zeros(25)
        new_value = vi.bellman_backup(0, values)

        assert isinstance(new_value, (float, np.floating))

    def test_bellman_error_interface(self):
        """Test compute_bellman_error returns correct type."""
        env = GridWorldEnv(seed=123)
        vi = ValueIteration(env)

        values = np.random.rand(25)
        error = vi.compute_bellman_error(values)

        assert isinstance(error, (float, np.floating))
        assert error >= 0


class TestQIterationInterface:
    """Test QIteration interface compliance."""

    def test_q_iteration_initialization(self):
        """Test QIteration can be initialized."""
        env = GridWorldEnv(seed=123)
        qi = QIteration(env)

        assert qi.gamma == 0.95
        assert qi.epsilon == 1e-4
        assert qi.n_states == 25
        assert qi.n_actions == 4

    def test_solve_interface(self):
        """Test solve returns correct types."""
        env = GridWorldEnv(seed=123)
        qi = QIteration(env, gamma=0.9, epsilon=0.01)

        q_values, iterations = qi.solve(max_iterations=10)

        assert isinstance(q_values, np.ndarray)
        assert q_values.shape == (25, 4)
        assert isinstance(iterations, (int, np.integer))
        assert iterations > 0

    def test_bellman_update_interface(self):
        """Test bellman_update returns correct type."""
        env = GridWorldEnv(seed=123)
        qi = QIteration(env)

        q_values = np.zeros((25, 4))
        new_q = qi.bellman_update(0, 0, q_values)

        assert isinstance(new_q, (float, np.floating))

    def test_extract_policy_interface(self):
        """Test extract_policy returns correct type."""
        env = GridWorldEnv(seed=123)
        qi = QIteration(env)

        q_values = np.random.rand(25, 4)
        policy = qi.extract_policy(q_values)

        assert isinstance(policy, np.ndarray)
        assert policy.shape == (25,)
        assert policy.dtype in [np.int32, np.int64]
        assert all(0 <= a < 4 for a in policy)

    def test_extract_values_interface(self):
        """Test extract_values returns correct type."""
        env = GridWorldEnv(seed=123)
        qi = QIteration(env)

        q_values = np.random.rand(25, 4)
        values = qi.extract_values(q_values)

        assert isinstance(values, np.ndarray)
        assert values.shape == (25,)

    def test_bellman_error_interface(self):
        """Test compute_bellman_error returns correct type."""
        env = GridWorldEnv(seed=123)
        qi = QIteration(env)

        q_values = np.random.rand(25, 4)
        error = qi.compute_bellman_error(q_values)

        assert isinstance(error, (float, np.floating))
        assert error >= 0


def run_interface_tests():
    """Run all interface tests and report results."""
    print("Testing Problem 1 Interface Compliance")
    print("=" * 50)

    # Count tests
    test_classes = [TestEnvironmentInterface, TestValueIterationInterface, TestQIterationInterface]
    total_tests = sum(len([m for m in dir(tc) if m.startswith('test_')]) for tc in test_classes)

    passed = 0
    failed = 0

    for test_class in test_classes:
        tc = test_class()
        class_name = test_class.__name__
        print(f"\n{class_name}:")

        for method_name in dir(tc):
            if method_name.startswith('test_'):
                try:
                    method = getattr(tc, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: Unexpected error - {e}")
                    failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total_tests} tests passed")

    if failed == 0:
        print("All interface tests passed!")
        return 0
    else:
        print(f"{failed} tests failed - review your implementations")
        return 1


if __name__ == "__main__":
    sys.exit(run_interface_tests())