"""
Training script for multi-agent DQN with communication.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import json
import os
import random
import matplotlib.pyplot as plt 
from typing import Tuple, Optional, Dict
from multi_agent_env import MultiAgentEnv
from models import AgentDQN, DuelingDQN
from replay_buffer import ReplayBuffer


# Utility function from snippet for seed setting
def set_all_seeds(seed: int):
    """Set seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

# Utility function from snippet for observation masking
def apply_observation_mask(obs, mode: str):
    """
    Apply masking to observation based on ablation mode.
    """
    is_tensor = isinstance(obs, torch.Tensor)
    
    if is_tensor:
        masked_obs = obs.clone()
    else:
        masked_obs = obs.copy()

    # Elements 9 (comm_in) and 10 (normalized_dist) are the last two in the 11-dim vector.
    index_9_to_11 = slice(9, 11)
    index_10 = 10
    zero_val = 0.0

    if mode == 'independent':
        # (a) Independent: Set elements 9 (comm) and 10 (dist) to zero
        if masked_obs.ndim <= 1: 
            masked_obs[index_9_to_11] = zero_val
        else:
            masked_obs[:, index_9_to_11] = zero_val
    elif mode == 'comm':
        # (b) Communication Only: Set element 10 (dist) to zero
        if masked_obs.ndim <= 1: 
            masked_obs[index_10] = zero_val
        else:
            masked_obs[:, index_10] = zero_val
    elif mode == 'full':
        # (c) Full Information: No masking
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return masked_obs


def plot_training_curves(mode: str, root_dir: str = 'results') -> None:
    """Plot the training curves from the log file."""
    log_path = os.path.join(root_dir, mode, 'training_logs', 'training_log.json')
    if not os.path.exists(log_path):
        print(f"Warning: Log file not found at {log_path}. Skipping plot generation.")
        return

    with open(log_path, 'r') as f:
        logs = json.load(f)

    episodes = [log['episode'] for log in logs]
    avg_rewards = [log['avg_reward'] for log in logs]
    success_rates = [log['success_rate'] for log in logs]
    loss_A = [log['loss_A'] for log in logs]
    loss_B = [log['loss_B'] for log in logs]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 1. Rewards
    axes[0].plot(episodes, avg_rewards)
    axes[0].set_title(f'Avg Reward per Episode ({mode} Mode)')
    axes[0].set_ylabel('Avg Reward')
    axes[0].grid(True)

    # 2. Success Rate
    axes[1].plot(episodes, success_rates)
    axes[1].set_title(f'Success Rate per Episode ({mode} Mode)')
    axes[1].set_ylabel('Success Rate')
    axes[1].grid(True)

    # 3. Loss
    axes[2].plot(episodes, loss_A, label='Loss A')
    axes[2].plot(episodes, loss_B, label='Loss B', linestyle='--')
    axes[2].set_title(f'Loss per Update ({mode} Mode)')
    axes[2].set_ylabel('Total Loss')
    axes[2].set_xlabel('Episode')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    
    plot_dir = os.path.join(root_dir, mode, 'training_logs')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'training_curves.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Training curves plotted and saved to {plot_path}")


class MultiAgentTrainer:
    """
    Trainer for multi-agent DQN system.
    """

    def __init__(self, env: MultiAgentEnv, args):
        """
        Initialize trainer and networks.
        """
        self.env = env
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episode_log = []
        self.global_frame = 0

        ModelClass = DuelingDQN if args.use_dueling else AgentDQN

        # Initialize current and target networks for Agent A
        self.model_A = ModelClass(input_dim=11, hidden_dim=args.hidden_dim, num_actions=env.num_actions).to(self.device)
        self.target_A = ModelClass(input_dim=11, hidden_dim=args.hidden_dim, num_actions=env.num_actions).to(self.device)
        self.target_A.load_state_dict(self.model_A.state_dict())
        self.target_A.eval() 

        # Initialize current and target networks for Agent B
        self.model_B = ModelClass(input_dim=11, hidden_dim=args.hidden_dim, num_actions=env.num_actions).to(self.device)
        self.target_B = ModelClass(input_dim=11, hidden_dim=args.hidden_dim, num_actions=env.num_actions).to(self.device)
        self.target_B.load_state_dict(self.model_B.state_dict())
        self.target_B.eval()
        
        # Initialize optimizers
        self.optimizer_A = optim.Adam(self.model_A.parameters(), lr=args.lr)
        self.optimizer_B = optim.Adam(self.model_B.parameters(), lr=args.lr)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=args.buffer_size, seed=args.seed)
        
        # Epsilon schedule parameters
        self.epsilon = args.epsilon_start
        self.epsilon_decay_rate = (args.epsilon_start - args.epsilon_end) / args.epsilon_frames if args.epsilon_frames > 0 else 0


    def _select_action(self, state: np.ndarray, network: nn.Module) -> Tuple[int, float, float]:
        """
        Select action using epsilon-greedy policy and get communication signal.
        
        Returns: action, comm_out, epsilon
        """
        # Apply observation mask based on mode
        state_masked = apply_observation_mask(state, self.args.mode)
        
        # Decay epsilon
        self.epsilon = max(self.args.epsilon_end, self.args.epsilon_start - self.global_frame * self.epsilon_decay_rate)
        
        # Exploration vs Exploitation
        if random.random() < self.epsilon:
            # Exploration: Choose random action (0 to 4)
            action = random.randrange(self.env.num_actions)
            # Use current network to get communication signal for the random state
            with torch.no_grad():
                state_tensor = torch.from_numpy(state_masked).float().unsqueeze(0).to(self.device)
                _, comm_out_tensor = network(state_tensor)
                comm_out = comm_out_tensor.item()
        else:
            # Exploitation: Choose greedy action
            with torch.no_grad():
                state_tensor = torch.from_numpy(state_masked).float().unsqueeze(0).to(self.device)
                q_values, comm_out_tensor = network(state_tensor)
                action = q_values.max(1)[1].item()
                comm_out = comm_out_tensor.item()
                
        # Comm signal is clamped in the environment, but for consistency here too.
        comm_out = np.clip(comm_out, 0.0, 1.0)
        
        return action, comm_out, self.epsilon


    def _update_networks(self) -> Tuple[float, float]:
        """
        Perform a single gradient update step on both agent networks.
        """
        if len(self.replay_buffer) < self.args.batch_size:
            return 0.0, 0.0

        # Sample batch
        transitions = self.replay_buffer.sample(self.args.batch_size)
        
        # Unpack and move to device
        s_A, s_B, a_A, a_B, c_A_in, c_B_in, r, s_A_p, s_B_p, d = [t.to(self.device) for t in transitions]

        # -----------------------------------------------------------
        # AGENT A UPDATE
        # -----------------------------------------------------------
        
        # 1. Compute Q(s, a) for the taken action a_A (Current Q-values)
        # We must mask the current state s_A as well, as this state includes comm_in
        s_A_masked = apply_observation_mask(s_A, self.args.mode)
        q_A_current, c_A_out = self.model_A(s_A_masked)
        q_A_current_action = q_A_current.gather(1, a_A) # Q(s_A, a_A)
        
        # 2. Compute Target Q-values: R + gamma * max_a' Q_target(s', a')
        s_A_p_masked = apply_observation_mask(s_A_p, self.args.mode)
        with torch.no_grad():
            # Calculate max Q' using the Target Network
            q_A_next_target, _ = self.target_A(s_A_p_masked)
            max_q_A_next = q_A_next_target.max(1)[0].unsqueeze(1)
            
            # Compute Q_target: R + gamma * max_Q' (0 if done)
            q_A_target = r + self.args.gamma * max_q_A_next * (1 - d)
        
        # 3. Calculate Loss for Agent A
        # DQN Loss (MSE of TD Error)
        loss_dqn_A = F.mse_loss(q_A_current_action, q_A_target)
        
        # Communication Loss (L2 Regularization on output signal magnitude)
        loss_comm_A = c_A_out.pow(2).mean() # E[c_out^2]
        
        # Total Loss
        loss_A = loss_dqn_A + self.args.comm_lambda * loss_comm_A
        
        # 4. Optimize Agent A
        self.optimizer_A.zero_grad()
        loss_A.backward()
        nn.utils.clip_grad_norm_(self.model_A.parameters(), self.args.grad_clip) # Gradient Clipping
        self.optimizer_A.step()


        # -----------------------------------------------------------
        # AGENT B UPDATE (Symmetric to Agent A)
        # -----------------------------------------------------------
        
        # 1. Current Q-values
        s_B_masked = apply_observation_mask(s_B, self.args.mode)
        q_B_current, c_B_out = self.model_B(s_B_masked)
        q_B_current_action = q_B_current.gather(1, a_B) # Q(s_B, a_B)
        
        # 2. Target Q-values
        s_B_p_masked = apply_observation_mask(s_B_p, self.args.mode)
        with torch.no_grad():
            q_B_next_target, _ = self.target_B(s_B_p_masked)
            max_q_B_next = q_B_next_target.max(1)[0].unsqueeze(1)
            q_B_target = r + self.args.gamma * max_q_B_next * (1 - d)
        
        # 3. Calculate Loss for Agent B
        loss_dqn_B = F.mse_loss(q_B_current_action, q_B_target)
        loss_comm_B = c_B_out.pow(2).mean() # E[c_out^2]
        loss_B = loss_dqn_B + self.args.comm_lambda * loss_comm_B
        
        # 4. Optimize Agent B
        self.optimizer_B.zero_grad()
        loss_B.backward()
        nn.utils.clip_grad_norm_(self.model_B.parameters(), self.args.grad_clip) # Gradient Clipping
        self.optimizer_B.step()
        
        return loss_A.item(), loss_B.item()


    def train(self):
        """
        Main training loop.
        """
        print(f"Starting training in {self.args.mode} mode on {self.device}. Using DuelingDQN: {self.args.use_dueling}")

        # Setup results directory
        model_dir = os.path.join('results', self.args.mode, 'agent_models')
        log_dir = os.path.join('results', self.args.mode, 'training_logs')
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'training_log.json')
        
        # Reset episode-wise statistics
        episode_rewards = []
        episode_successes = []
        update_losses_A = []
        update_losses_B = []

        # Start episode loop
        for episode in range(1, self.args.num_episodes + 1):
            s_A, s_B = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                self.global_frame += 1

                # 1. Action selection (Epsilon-greedy)
                a_A, c_A_out, _ = self._select_action(s_A, self.model_A)
                a_B, c_B_out, current_eps = self._select_action(s_B, self.model_B)

                # 2. Environment step
                (s_A_p, s_B_p), r, done = self.env.step(a_A, a_B, c_A_out, c_B_out)
                total_reward += r

                # 3. Store experience (c_A_out/c_B_out are the *output* comm signals)
                self.replay_buffer.push(s_A, s_B, a_A, a_B, c_A_out, c_B_out, r, s_A_p, s_B_p, done)

                # 4. Update Q-networks
                if self.global_frame % self.args.train_freq == 0 and len(self.replay_buffer) > self.args.batch_size:
                    loss_A, loss_B = self._update_networks()
                    update_losses_A.append(loss_A)
                    update_losses_B.append(loss_B)
                
                # 5. Update target networks
                if self.global_frame % self.args.target_update == 0:
                    self.target_A.load_state_dict(self.model_A.state_dict())
                    self.target_B.load_state_dict(self.model_B.state_dict())

                s_A, s_B = s_A_p, s_B_p

            # Episode finished
            episode_rewards.append(total_reward)
            # Success is defined as simultaneous arrival (reward == 10.0 in the step function)
            episode_successes.append(1 if self.env.agent_positions[0] == self.env.target_pos and self.env.agent_positions[1] == self.env.target_pos else 0)

            # Logging
            if episode % self.args.log_freq == 0:
                avg_reward = np.mean(episode_rewards[-self.args.log_freq:])
                success_rate = np.mean(episode_successes[-self.args.log_freq:])
                avg_loss_A = np.mean(update_losses_A[-self.args.log_freq * self.env.max_steps:]) if update_losses_A else 0.0
                avg_loss_B = np.mean(update_losses_B[-self.args.log_freq * self.env.max_steps:]) if update_losses_B else 0.0
                
                print(f"Episode {episode}/{self.args.num_episodes} | Frames {self.global_frame} | Success Rate: {success_rate:.4f} | Avg Reward: {avg_reward:.2f} | Epsilon: {current_eps:.4f} | Avg Loss A: {avg_loss_A:.4f} | Avg Loss B: {avg_loss_B:.4f}")
                
                self.episode_log.append({
                    'episode': episode,
                    'avg_reward': avg_reward,
                    'success_rate': success_rate,
                    'loss_A': avg_loss_A,
                    'loss_B': avg_loss_B,
                    'epsilon': current_eps
                })
                
                # Save log file
                with open(log_path, 'w') as f:
                    json.dump(self.episode_log, f, indent=4)

            # Model Saving
            if episode % self.args.save_freq == 0 or episode == self.args.num_episodes:
                path_A = os.path.join(model_dir, f'agent_A_ep{episode}.pth')
                path_B = os.path.join(model_dir, f'agent_B_ep{episode}.pth')
                torch.save(self.model_A.state_dict(), path_A)
                torch.save(self.model_B.state_dict(), path_B)
                print(f"Models saved at episode {episode}")

        # Final cleanup and plotting
        print("Training complete.")
        plot_training_curves(self.args.mode, 'results')


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent DQN Training Script")

    # Environment parameters
    parser.add_argument('--max-steps', type=int, default=50,
                        help='Maximum steps per episode')

    # DQN/Training parameters
    parser.add_argument('--num-episodes', type=int, default=5000,
                        help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Dimension of hidden layers')
    parser.add_argument('--buffer-size', type=int, default=50000,
                        help='Replay buffer capacity')
    parser.add_argument('--train-freq', type=int, default=1,
                        help='Number of frames between optimization steps')
    parser.add_argument('--target-update', type=int, default=100,
                        help='Target network update frequency')

    # Exploration parameters
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Initial epsilon for exploration')
    parser.add_argument('--epsilon-end', type=float, default=0.05,
                        help='Final epsilon for exploration')
    parser.add_argument('--epsilon-frames', type=int, default=40000,
                        help='Frames over which epsilon decays')

    # Communication/Ablation parameters
    parser.add_argument('--comm-lambda', type=float, default=1e-4,
                        help='Regularization strength for communication loss (lambda)')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['independent', 'comm', 'full'],
                       help='Information mode: independent (mask comm+dist), '
                            'comm (mask dist only), full (no masking)')
    parser.add_argument('--use-dueling', action='store_true',
                       help='Use DuelingDQN architecture')
    parser.add_argument('--grad-clip', type=float, default=10.0,
                        help='Gradient clipping value')


    # Other parameters
    parser.add_argument('--seed', type=int, default=641,
                       help='Random seed')
    parser.add_argument('--save-freq', type=int, default=500,
                       help='Model save frequency')
    parser.add_argument('--log-freq', type=int, default=10,
                       help='Training log frequency (episodes)')

    args = parser.parse_args()

    # Set random seeds
    set_all_seeds(args.seed)

    # Create environment
    env = MultiAgentEnv(max_steps=args.max_steps, seed=args.seed)

    # Create trainer
    trainer = MultiAgentTrainer(env, args)

    # Run training
    trainer.train()

if __name__ == '__main__':
    # Add a try/except block to handle argparse when running in environments like notebooks
    try:
        main()
    except SystemExit:
        # This occurs if running in an environment that simulates command line execution
        pass