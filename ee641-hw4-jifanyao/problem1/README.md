# Problem 1: Value and Q-Iteration - Stochastic Gridworld

Implementation of dynamic programming algorithms for optimal control in a stochastic gridworld MDP.

## Environment

The agent navigates a 5×5 grid with stochastic transitions modeling environmental uncertainty:

- **Transition Dynamics**: P(intended) = 0.8, P(drift left) = 0.1, P(drift right) = 0.1
- **Grid Elements**: Start (0,0), Goal (4,4), Obstacles (2,2), (1,3), Penalties (3,1), (0,3)
- **Rewards**: Goal +10, Penalties -5, Movement cost -0.1
- **Termination**: Goal reached or 50 steps

## File Structure

```
problem1/
├── environment.py         # GridWorld MDP implementation
├── value_iteration.py     # Value Iteration algorithm
├── q_iteration.py         # Q-Iteration algorithm
├── train.py              # Training script for both algorithms
├── visualize.py          # Visualization utilities
└── results/              # Output directory (created during training)
```

## Training

```bash
python train.py --gamma 0.95 --epsilon 1e-4 --seed 641
```

**Options:**
- `--gamma`: Discount factor (default: 0.95)
- `--epsilon`: Convergence threshold (default: 1e-4)
- `--max-iter`: Maximum iterations (default: 1000)
- `--seed`: Random seed (default: 641)

Saves `value_function.npz`, `q_function.npz`, and `optimal_policy.npz` to results directory.

## Visualization

```bash
python visualize.py
```

Generates:
- `visualizations/value_heatmap.png`: Value function visualization
- `visualizations/policy_arrows.png`: Optimal policy visualization
- `visualizations/q_functions.png`: Q-values for each action
- `visualizations/convergence.png`: Algorithm convergence comparison

## Implementation Requirements

### Core Components

**`environment.py`:**
- `GridWorldEnv`: MDP with stochastic transitions
- `get_transition_prob()`: P(s'|s,a) computation
- `get_reward()`: R(s,a,s') function

**`value_iteration.py`:**
- `solve()`: Iterative value function computation
- `compute_q_values()`: Bellman backup for state
- `extract_policy()`: Policy extraction from values

**`q_iteration.py`:**
- `solve()`: Iterative Q-function computation
- `bellman_update()`: Q(s,a) update rule
- `extract_policy()`: Policy extraction from Q-values

**`train.py`:**
- Algorithm execution and comparison
- Convergence tracking
- Results serialization

**`visualize.py`:**
- Value function heatmap
- Policy arrow visualization
- Convergence analysis

## Algorithm Parameters

Default configuration:
- gamma = 0.95
- epsilon = 1e-4
- grid_size = 5×5
- action_space = 4 (UP, RIGHT, DOWN, LEFT)
- state_space = 25

Expected convergence: 50-100 iterations (Value Iteration), 30-70 iterations (Q-Iteration)