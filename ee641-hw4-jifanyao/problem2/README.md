# Problem 2: Multi-Agent DQN - Coordination with Communication

Implementation of multi-agent deep Q-learning with emergent communication protocols for cooperative navigation.

## Task

Two agents with partial observability must coordinate to simultaneously reach a target location in a 10×10 gridworld. Agents observe 3×3 local patches and exchange learned communication signals.

- **Observations**: 3×3 grid patch + communication scalar (10-dimensional)
- **Actions**: Movement (5 discrete) + communication output (continuous)
- **Reward**: +10 only when both agents reach target simultaneously
- **Challenge**: Discover coordination through sparse rewards and limited perception

## File Structure

```
problem2/
├── multi_agent_env.py     # Multi-agent gridworld environment
├── models.py              # DQN architectures with communication
├── replay_buffer.py       # Experience replay implementation
├── train.py               # Multi-agent training loop
├── evaluate.py            # Policy evaluation and analysis
└── results/               # Output directory (created during training)
```

## Training

```bash
python train.py --num-episodes 5000 --batch-size 32 --lr 1e-3
```

**Options:**
- `--num-episodes`: Training episodes (default: 5000)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--gamma`: Discount factor (default: 0.99)
- `--epsilon-start`: Initial exploration (default: 1.0)
- `--epsilon-end`: Final exploration (default: 0.05)
- `--epsilon-decay`: Decay rate (default: 0.995)
- `--hidden-dim`: Hidden layer size (default: 64)
- `--target-update`: Target network update frequency (default: 100)
- `--seed`: Random seed (default: 641)

Saves `agent_models/` and `training_logs/` to results directory.

## Evaluation

```bash
python evaluate.py --checkpoint results/agent_models/best_checkpoint.pth
```

**Options:**
- `--checkpoint`: Path to saved models (required)
- `--num-episodes`: Evaluation episodes (default: 100)
- `--render`: Visualize episodes (default: False)

Generates:
- `evaluation_results/performance.json`: Success rate and statistics
- `evaluation_results/trajectories.png`: Agent path visualization
- `evaluation_results/communication_analysis.json`: Protocol analysis

## Implementation Requirements

### Core Components

**`multi_agent_env.py`:**
- `MultiAgentEnv`: Gridworld with partial observations
- `_get_observation()`: 3×3 patch extraction with communication
- `step()`: Joint action execution and reward computation

**`models.py`:**
- `AgentDQN`: Dual-output network (Q-values + communication)
- `forward()`: Process observation, output actions and signal

**`replay_buffer.py`:**
- `ReplayBuffer`: Store and sample joint experiences
- `push()`: Add transitions with communication
- `sample()`: Batch sampling for training

**`train.py`:**
- `MultiAgentTrainer`: Coordinate training of both agents
- `select_action()`: Epsilon-greedy with communication
- `update_networks()`: Joint Q-learning updates

**`evaluate.py`:**
- `MultiAgentEvaluator`: Analyze learned policies
- `analyze_communication()`: Interpret emergent protocols
- `test_generalization()`: Evaluate on new configurations

## Network Architecture

Default configuration:
- input_dim = 10 (9 grid cells + 1 communication)
- hidden_dim = 64
- num_actions = 5
- communication_dim = 1 (scalar signal)

Total parameters per agent: ~5K