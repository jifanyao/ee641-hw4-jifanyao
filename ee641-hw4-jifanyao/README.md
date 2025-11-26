# EE 641 - Homework 4: Reinforcement Learning Foundations

Starter code for implementing dynamic programming and deep reinforcement learning algorithms.

## Structure

- **`problem1/`** - Value Iteration and Q-Iteration on stochastic gridworld
- **`problem2/`** - Multi-agent coordination with learned communication

See individual problem READMEs for detailed instructions:
- [Problem 1 README](problem1/README.md)
- [Problem 2 README](problem2/README.md)

## Requirements

```bash
pip install torch>=2.0.0 numpy>=1.24.0 matplotlib>=3.7.0 tqdm>=4.65.0
```

## Quick Start

### Problem 1: Value and Q-Iteration
```bash
cd problem1
python train.py --seed 641
python visualize.py
```

### Problem 2: Multi-Agent DQN
```bash
cd problem2
python train.py --seed 641 --num_episodes 5000
python evaluate.py
```

## Full Assignment

See the course website for complete assignment instructions, deliverables, and submission requirements.