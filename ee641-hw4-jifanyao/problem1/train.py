"""
Training script for Value Iteration and Q-Iteration.
"""

import numpy as np
import argparse
import json
import os
import time
from environment import GridWorldEnv
from value_iteration import ValueIteration
from q_iteration import QIteration


def main():
    """
    Run both algorithms and save results.
    """
    parser = argparse.ArgumentParser(description='Train RL algorithms on GridWorld')
    parser.add_argument('--seed', type=int, default=641, help='Random seed')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Convergence threshold')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum iterations')
    args = parser.parse_args()

    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Initialize environment with seed
    env = GridWorldEnv(seed=args.seed)
    n_states = env.grid_size ** 2

    print(f"--- Running Value Iteration (gamma={args.gamma}, epsilon={args.epsilon}) ---")
    
    # Run Value Iteration
    vi_solver = ValueIteration(env, gamma=args.gamma, epsilon=args.epsilon)
    start_time_vi = time.time()
    # Solve for optimal values
    vi_values, vi_iterations = vi_solver.solve(max_iterations=args.max_iter)
    end_time_vi = time.time()
    # Extract policy
    vi_policy = vi_solver.extract_policy(vi_values)
    vi_time = end_time_vi - start_time_vi
    
    # Save results
    vi_results = {
        'algorithm': 'Value Iteration',
        'iterations': vi_iterations,
        'convergence_time': vi_time,
        'gamma': args.gamma,
        'epsilon': args.epsilon,
        'values': vi_values.tolist(),
        'policy': vi_policy.tolist(),
        'error_history': vi_solver.error_history
    }
    with open('results/vi_results.json', 'w') as f:
        json.dump(vi_results, f, indent=4)
    print(f"Value Iteration converged in {vi_iterations} iterations ({vi_time:.4f}s). Results saved.")


    print(f"\n--- Running Q-Iteration (gamma={args.gamma}, epsilon={args.epsilon}) ---")
    
    # Run Q-Iteration
    qi_solver = QIteration(env, gamma=args.gamma, epsilon=args.epsilon)
    start_time_qi = time.time()
    # Solve for optimal Q-values
    qi_q_values, qi_iterations = qi_solver.solve(max_iterations=args.max_iter)
    end_time_qi = time.time()
    # Extract policy and values
    qi_policy = qi_solver.extract_policy(qi_q_values)
    qi_values = qi_solver.extract_values(qi_q_values)
    qi_time = end_time_qi - start_time_qi
    
    # Save results
    qi_results = {
        'algorithm': 'Q-Iteration',
        'iterations': qi_iterations,
        'convergence_time': qi_time,
        'gamma': args.gamma,
        'epsilon': args.epsilon,
        'q_values': qi_q_values.tolist(),
        'values': qi_values.tolist(),
        'policy': qi_policy.tolist(),
        'error_history': qi_solver.error_history
    }
    with open('results/qi_results.json', 'w') as f:
        json.dump(qi_results, f, indent=4)
    print(f"Q-Iteration converged in {qi_iterations} iterations ({qi_time:.4f}s). Results saved.")


    # Compare algorithms
    print("\n--- Algorithm Comparison ---")
    
    # Print convergence statistics
    print(f"Value Iteration: {vi_iterations} iterations, {vi_time:.4f}s")
    print(f"Q-Iteration:     {qi_iterations} iterations, {qi_time:.4f}s")
    
    # Check if policies match
    policies_match = np.array_equal(vi_policy, qi_policy)
    print(f"Policies match: {policies_match}")

    # Check if values match (using the extracted values from QI)
    max_value_diff = np.max(np.abs(vi_values - qi_values))
    print(f"Max Value Difference: {max_value_diff:.6e}")
    # Values should be extremely close (within a small tolerance)
    values_close = max_value_diff < args.epsilon * 10 
    print(f"Values are close (within 10*epsilon): {values_close}")
    
    # Save comparison results
    comparison_results = {
        'vi_iterations': vi_iterations,
        'qi_iterations': qi_iterations,
        'vi_convergence_time': vi_time,
        'qi_convergence_time': qi_time,
        'policies_match': policies_match,
        'max_value_diff': max_value_diff
    }
    with open('results/comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=4)
    print("Comparison results saved to results/comparison_results.json")


if __name__ == '__main__':
    main()