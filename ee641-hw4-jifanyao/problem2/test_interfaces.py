"""
Interface tests for Problem 2 implementations.

These tests verify that your implementations conform to the required interfaces.
They do not test correctness of algorithms, only that functions exist with
correct signatures and return appropriate types.
"""

import numpy as np
import torch
import sys
from typing import Tuple

# Import student implementations
try:
    from multi_agent_env import MultiAgentEnv
    from models import AgentDQN, DuelingDQN
    from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
except ImportError as e:
    print(f"Import error: {e}")
    print("Ensure all required files are in the current directory")
    sys.exit(1)


class TestEnvironmentInterface:
    """Test MultiAgentEnv interface compliance."""

    def test_environment_initialization(self):
        """Test environment can be initialized with default and custom parameters."""
        env1 = MultiAgentEnv()
        assert env1.grid_size == (10, 10)
        assert env1.obs_window == 3
        assert env1.max_steps == 50

        env2 = MultiAgentEnv(grid_size=(8, 8), obs_window=3, max_steps=100, seed=42)
        assert env2.grid_size == (8, 8)
        assert env2.max_steps == 100

    def test_reset_interface(self):
        """Test reset returns correct types and shapes."""
        env = MultiAgentEnv(seed=123)
        obs_A, obs_B = env.reset()

        assert isinstance(obs_A, np.ndarray)
        assert isinstance(obs_B, np.ndarray)
        assert obs_A.shape == (11,)
        assert obs_B.shape == (11,)
        assert obs_A.dtype == np.float32
        assert obs_B.dtype == np.float32

        # Check communication signals and distance are initialized properly
        assert obs_A[9] == 0.0  # Communication
        assert obs_B[9] == 0.0  # Communication
        assert 0.0 <= obs_A[10] <= 1.0  # Distance
        assert 0.0 <= obs_B[10] <= 1.0  # Distance

    def test_step_interface(self):
        """Test step function interface."""
        env = MultiAgentEnv(seed=123)
        env.reset()

        # Test with valid actions and communication
        action_A = 0  # UP
        action_B = 2  # LEFT
        comm_A = 0.5
        comm_B = 0.7

        result = env.step(action_A, action_B, comm_A, comm_B)
        observations, reward, done = result

        assert isinstance(observations, tuple)
        assert len(observations) == 2
        obs_A, obs_B = observations

        assert isinstance(obs_A, np.ndarray)
        assert isinstance(obs_B, np.ndarray)
        assert obs_A.shape == (11,)
        assert obs_B.shape == (11,)

        assert isinstance(reward, (float, np.floating))
        assert isinstance(done, bool)

        # Check communication signals and distance are bounded
        assert 0.0 <= obs_A[9] <= 1.0  # Communication
        assert 0.0 <= obs_B[9] <= 1.0  # Communication
        assert 0.0 <= obs_A[10] <= 1.0  # Distance
        assert 0.0 <= obs_B[10] <= 1.0  # Distance

    def test_observation_extraction(self):
        """Test observation extraction produces correct format."""
        env = MultiAgentEnv(seed=123)
        env.reset()

        obs_A = env._get_observation(0)
        obs_B = env._get_observation(1)

        assert isinstance(obs_A, np.ndarray)
        assert isinstance(obs_B, np.ndarray)
        assert obs_A.shape == (11,)
        assert obs_B.shape == (11,)

        # First 9 elements should be grid values
        for i in range(9):
            assert obs_A[i] >= -1.0  # -1 for off-grid
            assert obs_A[i] <= 2.0   # 2 for target

    def test_position_validation(self):
        """Test position validation function."""
        env = MultiAgentEnv(seed=123)
        env.reset()

        # Test valid position
        is_valid = env._is_valid_position((5, 5))
        assert isinstance(is_valid, bool)

        # Test invalid positions
        is_valid = env._is_valid_position((-1, 0))
        assert is_valid == False

        is_valid = env._is_valid_position((10, 10))
        assert is_valid == False

    def test_action_application(self):
        """Test action application returns valid positions."""
        env = MultiAgentEnv(seed=123)
        env.reset()

        # Test all actions
        for action in range(5):
            new_pos = env._apply_action((5, 5), action)
            assert isinstance(new_pos, tuple)
            assert len(new_pos) == 2
            assert isinstance(new_pos[0], (int, np.integer))
            assert isinstance(new_pos[1], (int, np.integer))


class TestModelInterface:
    """Test neural network model interface compliance."""

    def test_agent_dqn_initialization(self):
        """Test AgentDQN can be initialized."""
        model = AgentDQN()
        assert hasattr(model, 'forward')

        model = AgentDQN(input_dim=11, hidden_dim=128, num_actions=5)
        assert hasattr(model, 'forward')

    def test_agent_dqn_forward(self):
        """Test AgentDQN forward pass interface."""
        model = AgentDQN(input_dim=11, hidden_dim=64, num_actions=5)

        # Single observation
        obs = torch.randn(1, 11)
        q_values, comm_signal = model(obs)

        assert isinstance(q_values, torch.Tensor)
        assert isinstance(comm_signal, torch.Tensor)
        assert q_values.shape == (1, 5)
        assert comm_signal.shape == (1, 1)

        # Batch of observations
        batch = torch.randn(32, 11)
        q_values, comm_signal = model(batch)

        assert q_values.shape == (32, 5)
        assert comm_signal.shape == (32, 1)

        # Check communication signal is bounded [0, 1]
        assert torch.all(comm_signal >= 0.0)
        assert torch.all(comm_signal <= 1.0)

    def test_dueling_dqn_initialization(self):
        """Test DuelingDQN can be initialized."""
        model = DuelingDQN()
        assert hasattr(model, 'forward')

        model = DuelingDQN(input_dim=11, hidden_dim=128, num_actions=5)
        assert hasattr(model, 'forward')

    def test_dueling_dqn_forward(self):
        """Test DuelingDQN forward pass interface."""
        model = DuelingDQN(input_dim=11, hidden_dim=64, num_actions=5)

        # Single observation
        obs = torch.randn(1, 11)
        q_values, comm_signal = model(obs)

        assert isinstance(q_values, torch.Tensor)
        assert isinstance(comm_signal, torch.Tensor)
        assert q_values.shape == (1, 5)
        assert comm_signal.shape == (1, 1)

        # Check communication signal is bounded
        assert torch.all(comm_signal >= 0.0)
        assert torch.all(comm_signal <= 1.0)


class TestReplayBufferInterface:
    """Test replay buffer interface compliance."""

    def test_replay_buffer_initialization(self):
        """Test ReplayBuffer can be initialized."""
        buffer = ReplayBuffer()
        assert len(buffer) == 0

        buffer = ReplayBuffer(capacity=5000, seed=42)
        assert len(buffer) == 0

    def test_replay_buffer_push(self):
        """Test pushing transitions to buffer."""
        buffer = ReplayBuffer(capacity=100)

        # Create sample transition
        state_A = np.random.randn(10).astype(np.float32)
        state_B = np.random.randn(10).astype(np.float32)
        action_A = 2
        action_B = 3
        comm_A = 0.5
        comm_B = 0.7
        reward = 1.0
        next_state_A = np.random.randn(10).astype(np.float32)
        next_state_B = np.random.randn(10).astype(np.float32)
        done = False

        buffer.push(state_A, state_B, action_A, action_B, comm_A, comm_B,
                   reward, next_state_A, next_state_B, done)

        assert len(buffer) == 1

    def test_replay_buffer_sample(self):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(capacity=100, seed=123)

        # Add multiple transitions
        for _ in range(50):
            state_A = np.random.randn(11).astype(np.float32)
            state_B = np.random.randn(11).astype(np.float32)
            action_A = np.random.randint(0, 5)
            action_B = np.random.randint(0, 5)
            comm_A = np.random.random()
            comm_B = np.random.random()
            reward = np.random.randn()
            next_state_A = np.random.randn(11).astype(np.float32)
            next_state_B = np.random.randn(11).astype(np.float32)
            done = np.random.random() > 0.9

            buffer.push(state_A, state_B, action_A, action_B, comm_A, comm_B,
                       reward, next_state_A, next_state_B, done)

        # Sample batch
        batch = buffer.sample(32)

        assert isinstance(batch, tuple)
        assert len(batch) == 10

        state_A_batch, state_B_batch, action_A_batch, action_B_batch, \
            comm_A_batch, comm_B_batch, reward_batch, \
            next_state_A_batch, next_state_B_batch, done_batch = batch

        assert state_A_batch.shape == (32, 11)
        assert state_B_batch.shape == (32, 11)
        assert action_A_batch.shape == (32,)
        assert action_B_batch.shape == (32,)
        assert comm_A_batch.shape == (32,)
        assert comm_B_batch.shape == (32,)
        assert reward_batch.shape == (32,)
        assert next_state_A_batch.shape == (32, 11)
        assert next_state_B_batch.shape == (32, 11)
        assert done_batch.shape == (32,)

    def test_prioritized_buffer_initialization(self):
        """Test PrioritizedReplayBuffer can be initialized."""
        buffer = PrioritizedReplayBuffer()
        assert len(buffer) == 0

        buffer = PrioritizedReplayBuffer(capacity=5000, alpha=0.6,
                                        beta_start=0.4, beta_steps=100000)
        assert len(buffer) == 0

    def test_prioritized_buffer_interfaces(self):
        """Test prioritized buffer specific interfaces."""
        buffer = PrioritizedReplayBuffer(capacity=100, seed=123)

        # Add transitions
        for _ in range(50):
            state_A = np.random.randn(11).astype(np.float32)
            state_B = np.random.randn(11).astype(np.float32)
            action_A = np.random.randint(0, 5)
            action_B = np.random.randint(0, 5)
            comm_A = np.random.random()
            comm_B = np.random.random()
            reward = np.random.randn()
            next_state_A = np.random.randn(11).astype(np.float32)
            next_state_B = np.random.randn(11).astype(np.float32)
            done = np.random.random() > 0.9

            buffer.push(state_A, state_B, action_A, action_B, comm_A, comm_B,
                       reward, next_state_A, next_state_B, done)

        # Sample with priorities
        result = buffer.sample(32)
        assert isinstance(result, tuple)
        assert len(result) == 3

        batch, weights, indices = result

        assert isinstance(batch, tuple)
        assert len(batch) == 10
        assert isinstance(weights, np.ndarray)
        assert weights.shape == (32,)
        assert isinstance(indices, (list, np.ndarray))

        # Test priority update
        priorities = np.random.rand(32) + 0.01
        buffer.update_priorities(indices[:32], priorities)


def run_interface_tests():
    """Run all interface tests and report results."""
    print("Testing Problem 2 Interface Compliance")
    print("=" * 50)

    # Count tests
    test_classes = [TestEnvironmentInterface, TestModelInterface, TestReplayBufferInterface]
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
                    print(f"  PASS: {method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  FAIL: {method_name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  FAIL: {method_name}: Unexpected error - {e}")
                    failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total_tests} tests passed")

    if failed == 0:
        print("All interface tests passed")
        return 0
    else:
        print(f"{failed} tests failed - review your implementations")
        return 1


if __name__ == "__main__":
    sys.exit(run_interface_tests())