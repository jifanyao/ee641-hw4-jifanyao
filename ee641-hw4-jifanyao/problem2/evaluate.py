"""
Evaluation script for trained multi-agent models.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Any
import json
import os
import argparse
import math

# Import student implementations
from multi_agent_env import MultiAgentEnv
from models import AgentDQN, DuelingDQN
# Reuse the masking utility from train.py
try:
    from train import apply_observation_mask
except ImportError:
    # Define a dummy one if train.py isn't imported correctly in some environments
    def apply_observation_mask(obs, mode: str):
        return obs


class MultiAgentEvaluator:
    """
    Evaluator for analyzing trained multi-agent policies.
    """

    def __init__(self, env: MultiAgentEnv, model_A: nn.Module, model_B: nn.Module, mode: str, device: str = 'cpu'):
        """
        Initialize evaluator.

        Args:
            env: Multi-agent environment
            model_A: Trained model for Agent A
            model_B: Trained model for Agent B
            mode: The ablation mode ('independent', 'comm', 'full') used for training
            device: 'cpu' or 'cuda'
        """
        self.env = env
        self.model_A = model_A
        self.model_B = model_B
        self.mode = mode
        self.device = torch.device(device)

        # Move models to device and set to evaluation mode
        self.model_A.to(self.device)
        self.model_B.to(self.device)
        self.model_A.eval()
        self.model_B.eval()

    def _select_greedy_action(self, state: np.ndarray, network: nn.Module) -> Tuple[int, float]:
        """
        Select action using greedy policy (no epsilon) and get communication signal.
        """
        state_masked = apply_observation_mask(state, self.mode)
        
        with torch.no_grad():
            state_tensor = torch.from_numpy(state_masked).float().unsqueeze(0).to(self.device)
            q_values, comm_out_tensor = network(state_tensor)
            action = q_values.max(1)[1].item()
            comm_out = comm_out_tensor.item()
            
        return action, comm_out

    def run_episode(self, start_A: Tuple[int, int], start_B: Tuple[int, int], target: Tuple[int, int], render: bool = False) -> Dict[str, Any]:
        """
        Run a single episode with the learned policy.
        """
        s_A, s_B = self.env.reset(start_A=start_A, start_B=start_B, target=target)
        done = False
        total_reward = 0.0
        trajectory_A = [start_A]
        trajectory_B = [start_B]
        comm_signals_A = [] # A's output
        comm_signals_B = [] # B's output
        
        while not done:
            # Select greedy action and get communication signal
            a_A, c_A_out = self._select_greedy_action(s_A, self.model_A)
            a_B, c_B_out = self._select_greedy_action(s_B, self.model_B)

            # Environment step
            (s_A_p, s_B_p), r, done = self.env.step(a_A, a_B, c_A_out, c_B_out)
            
            s_A, s_B = s_A_p, s_B_p
            total_reward += r
            trajectory_A.append(self.env.agent_positions[0])
            trajectory_B.append(self.env.agent_positions[1])
            comm_signals_A.append(c_A_out)
            comm_signals_B.append(c_B_out)

            if render:
                self.env.render()
                if done:
                    break

        # Check for success (simultaneous arrival)
        success = (self.env.agent_positions[0] == target and self.env.agent_positions[1] == target)
        
        return {
            'success': success,
            'total_reward': total_reward,
            'steps': self.env.step_count,
            'trajectory_A': trajectory_A,
            'trajectory_B': trajectory_B,
            'comm_A': comm_signals_A,
            'comm_B': comm_signals_B
        }

    def evaluate(self, num_episodes: int) -> Dict[str, Any]:
        """
        Run multiple evaluation episodes and collect statistics.
        """
        print(f"Running {num_episodes} evaluation episodes in {self.mode} mode...")
        
        results = []
        target = self.env.target_pos # Use default target (9, 9)
        start_A = self.env.agent_positions[0] # Use default start (2, 1)
        start_B = self.env.agent_positions[1] # Use default start (6, 4)
        
        # Ensure env is reset to default config for initial evaluation
        self.env.reset(start_A, start_B, target)

        for _ in range(num_episodes):
            results.append(self.run_episode(start_A, start_B, target))

        success_rate = np.mean([res['success'] for res in results])
        avg_reward = np.mean([res['total_reward'] for res in results])
        avg_steps = np.mean([res['steps'] for res in results])
        
        # Collect trajectories for plotting (e.g., the last successful trajectory)
        successful_trajectories = [(res['trajectory_A'], res['trajectory_B']) for res in results if res['success']]
        plot_trajectory = successful_trajectories[-1] if successful_trajectories else (results[-1]['trajectory_A'], results[-1]['trajectory_B'])


        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'trajectories': {
                'A': plot_trajectory[0],
                'B': plot_trajectory[1]
            },
            'all_comm_A': np.concatenate([res['comm_A'] for res in results]).tolist(),
            'all_comm_B': np.concatenate([res['comm_B'] for res in results]).tolist()
        }
        
    def analyze_communication(self, all_comm_A: List[float], all_comm_B: List[float]) -> Dict[str, Dict[str, float]]:
        """
        Analyze the emergent communication protocol.
        """
        comm_A = np.array(all_comm_A)
        comm_B = np.array(all_comm_B)
        
        analysis = {}
        
        if self.mode == 'independent':
            # Communication signal is masked/ignored, so analysis is not relevant.
            return {'message': 'Not applicable for independent mode.'}

        # Agent A analysis
        analysis['agent_A'] = {
            'mean_comm_out': np.mean(comm_A) if comm_A.size > 0 else 0.0,
            'std_comm_out': np.std(comm_A) if comm_A.size > 0 else 0.0,
            'max_comm_out': np.max(comm_A) if comm_A.size > 0 else 0.0
        }
        
        # Agent B analysis
        analysis['agent_B'] = {
            'mean_comm_out': np.mean(comm_B) if comm_B.size > 0 else 0.0,
            'std_comm_out': np.std(comm_B) if comm_B.size > 0 else 0.0,
            'max_comm_out': np.max(comm_B) if comm_B.size > 0 else 0.0
        }
        
        return analysis

    def test_generalization(self, configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Test the policy on novel start/target configurations.
        """
        generalization_results = {}
        num_test_episodes = 50 
        
        for i, config in enumerate(configs):
            key = f"config_{i+1}"
            
            target = tuple(config.get('target', (9, 9)))
            start_A = tuple(config.get('start_A', (2, 1)))
            start_B = tuple(config.get('start_B', (6, 4)))
            
            successes = 0
            # Run multiple episodes per config
            for _ in range(num_test_episodes):
                res = self.run_episode(start_A, start_B, target)
                if res['success']:
                    successes += 1
            
            generalization_results[key] = {
                'target': target,
                'start_A': start_A,
                'start_B': start_B,
                'success_rate': successes / num_test_episodes
            }
            
        return generalization_results

def plot_trajectories(grid_size: Tuple[int, int], trajectory_A: List[Tuple[int, int]], trajectory_B: List[Tuple[int, int]], 
                      target_pos: Tuple[int, int], obstacles: List[Tuple[int, int]], plot_path: str):
    """Plot the trajectories of agents A and B on the grid."""
    H, W = grid_size
    fig, ax = plt.subplots(figsize=(W, H))
    
    # 1. Plot grid lines and format axes
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5) # Invert y-axis to match (row, col) indexing
    ax.set_aspect('equal', adjustable='box')
    
    # 2. Plot features
    # Obstacles
    obs_rows = [r for r, c in obstacles]
    obs_cols = [c for r, c in obstacles]
    ax.scatter(obs_cols, obs_rows, marker='X', color='gray', s=200, label='Obstacle')
    
    # Target
    ax.scatter(target_pos[1], target_pos[0], marker='*', color='gold', s=400, edgecolors='black', label='Target (T)')
    
    # Start positions
    start_A = trajectory_A[0]
    start_B = trajectory_B[0]
    ax.scatter(start_A[1], start_A[0], marker='o', color='red', s=100, label='Start A')
    ax.scatter(start_B[1], start_B[0], marker='o', color='blue', s=100, label='Start B')

    # 3. Plot trajectories
    traj_A_cols = [c for r, c in trajectory_A]
    traj_A_rows = [r for r, c in trajectory_A]
    traj_B_cols = [c for r, c in trajectory_B]
    traj_B_rows = [r for r, c in trajectory_B]
    
    ax.plot(traj_A_cols, traj_A_rows, color='red', linestyle='-', linewidth=2, alpha=0.6)
    ax.plot(traj_B_cols, traj_B_rows, color='blue', linestyle='--', linewidth=2, alpha=0.6)
    
    # 4. Final positions
    ax.scatter(traj_A_cols[-1], traj_A_rows[-1], marker='s', color='red', s=150, label='End A', zorder=10)
    ax.scatter(traj_B_cols[-1], traj_B_rows[-1], marker='d', color='blue', s=150, label='End B', zorder=10)
    
    ax.set_title(f'Agent Trajectories ({len(trajectory_A)-1} Steps)')
    ax.legend(loc='lower right')
    
    plt.savefig(plot_path)
    plt.close()
    print(f"Trajectories plotted and saved to {plot_path}")

def create_evaluation_report(results: Dict[str, Any], report_path: str):
    """Save the final evaluation results to a JSON file."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation report saved to {report_path}")

# Default generalization configurations based on common test scenarios
DEFAULT_GENERALIZATION_CONFIGS = json.dumps([
    # New start positions, same target
    {"target": [9, 9], "start_A": [5, 5], "start_B": [1, 8]},
    # Same start, new target
    {"target": [1, 1], "start_A": [2, 1], "start_B": [6, 4]},
    # New start and new target
    {"target": [4, 9], "start_A": [0, 0], "start_B": [9, 0]}
])


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent DQN Evaluation Script")
    parser.add_argument('--mode', type=str, default='full',
                       choices=['independent', 'comm', 'full'],
                       help='Ablation mode used for training the model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to the model checkpoint (e.g., results/full/agent_models/agent_A_ep5000.pth)')
    parser.add_argument('--num-episodes', type=int, default=100,
                       help='Number of episodes for standard evaluation')
    parser.add_argument('--use-dueling', action='store_true',
                       help='Flag to specify if DuelingDQN was used during training')
    parser.add_argument('--render', action='store_true',
                       help='Render one episode visualization')
    parser.add_argument('--generalization-configs', type=str, 
                       default=DEFAULT_GENERALIZATION_CONFIGS,
                       help='JSON string defining generalization test configurations')
    
    args = parser.parse_args()

    # Determine model directory and episode
    model_dir = os.path.dirname(args.checkpoint)
    episode = os.path.basename(args.checkpoint).split('_ep')[1].replace('.pth', '')
    
    # Define paths for outputs
    output_dir = os.path.join('results', args.mode, 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Setup Environment and Models
    env = MultiAgentEnv(seed=641) # Use the same environment configuration
    ModelClass = DuelingDQN if args.use_dueling else AgentDQN

    # Load Agent A model
    model_A = ModelClass(input_dim=11, num_actions=env.num_actions)
    try:
        model_A.load_state_dict(torch.load(args.checkpoint))
    except RuntimeError as e:
        # Try loading the corresponding B model if A was provided
        if 'agent_A' in args.checkpoint:
            raise e
        elif 'agent_B' in args.checkpoint:
            print(f"Warning: Loaded checkpoint is for Agent B. Attempting to load Agent A from {args.checkpoint.replace('agent_B', 'agent_A')}")
            args.checkpoint = args.checkpoint.replace('agent_B', 'agent_A')
            model_A.load_state_dict(torch.load(args.checkpoint))
        else:
            raise e


    # Load Agent B model (assuming symmetry in naming convention)
    checkpoint_B = args.checkpoint.replace('agent_A', 'agent_B')
    model_B = ModelClass(input_dim=11, num_actions=env.num_actions)
    model_B.load_state_dict(torch.load(checkpoint_B))
    
    evaluator = MultiAgentEvaluator(env, model_A, model_B, args.mode)

    # 2. Standard Evaluation
    standard_results = evaluator.evaluate(args.num_episodes)
    
    # 3. Communication Analysis (Only if not in independent mode)
    if args.mode != 'independent':
        comm_analysis = evaluator.analyze_communication(standard_results['all_comm_A'], standard_results['all_comm_B'])
    else:
        comm_analysis = {'message': 'Not applicable for independent mode.'}
    
    # 4. Trajectory Plotting
    plot_trajectories(env.grid_size, standard_results['trajectories']['A'], standard_results['trajectories']['B'],
                      env.target_pos, env.obstacle_positions, os.path.join(output_dir, 'trajectories.png'))

    # 5. Generalization Test
    try:
        generalization_configs = json.loads(args.generalization_configs)
        generalization_results = evaluator.test_generalization(generalization_configs)
        print("\nGeneralization Test Results:")
        for k, v in generalization_results.items():
            print(f"  {k}: Target={v['target']}, Start A={v['start_A']}, Start B={v['start_B']}, Success Rate={v['success_rate']:.4f}")
    except json.JSONDecodeError:
        print("Warning: Failed to parse generalization configs JSON. Using default config in report.")
        generalization_results = {'message': 'Configuration parsing failed.'}
    except Exception as e:
        print(f"Warning: Generalization test failed: {e}")
        generalization_results = {'message': f'Test failed: {str(e)}'}

    # 6. Final Report Structure
    full_results = {
        'metadata': {
            'mode': args.mode,
            'num_episodes': args.num_episodes,
            'checkpoint_A': args.checkpoint,
            'checkpoint_B': checkpoint_B,
            'dueling_dqn': args.use_dueling
        },
        'standard_evaluation': {
            'success_rate': standard_results['success_rate'],
            'avg_reward': standard_results['avg_reward'],
            'avg_steps': standard_results['avg_steps']
        },
        'communication_analysis': comm_analysis,
        'generalization': generalization_results
    }
    
    # Optional: Render a single episode
    if args.render:
        print("\nRendering one episode (default configuration):")
        default_start_A = (2, 1)
        default_start_B = (6, 4)
        default_target = (9, 9)
        evaluator.run_episode(default_start_A, default_start_B, default_target, render=True)
    
    # Generate report
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    create_evaluation_report(full_results, report_path)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass