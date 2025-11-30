# Q-Learning Sandbox

This repository contains an interactive gridworld environment to visualize and experiment with tabular Q-learning, implemented with Streamlit, NumPy, and Plotly.

The app allows you to:

- Design a custom gridworld (walls, pits, reward tiles, start, goal).
- Train a model-free Q-learning agent on that environment.
- Visualize:
  - Individual training episodes (with exploration and learning),
  - A greedy policy rollout given the current Q-table,
  - A timelapse of episode paths as training progresses.
- Experiment with core RL hyperparameters:
  - Learning rate (α),
  - Discount factor (γ),
  - Exploration rate (ε),
  - Episode length and training horizon.

## Problem Formulation

The environment is a 6 × 6 grid (easily adjustable). Each cell is one of:

- Empty  
- Wall (impassable)  
- Start (unique initial state)  
- Goal (terminal state with positive reward)  
- Pit (terminal state with negative reward)  
- Reward (non-terminal bonus tile)

At each step, the agent chooses one of four actions:

- Up, Right, Down, Left.

The reward structure:

- Step reward: `STEP_REWARD = -0.04`  
- Goal: `GOAL_REWARD = +1.0` (episode terminates)  
- Pit: `PIT_REWARD = -1.0` (episode terminates)  
- Reward tile: `REWARD_BONUS = +0.5` on first visit per episode, non-terminal

To correctly handle one-time reward tiles, the Markov state is defined as:

> `state = (row, col, mask)`

where:

- `row, col` are the agent’s coordinates,
- `mask` is an integer bitmask indicating which reward tiles have already been collected in the current episode (1 = collected, 0 = not yet).

Reward tiles are discovered and indexed:

- `reward_positions = [(r1, c1), (r2, c2), …]`  
- `reward_index[(ri, ci)] = i`  

When stepping onto a reward tile at index `i`:

- If bit `i` in `mask` is 0, the agent receives the bonus and the bit is set to 1.
- If bit `i` is already 1, no additional bonus is given.

This makes the environment a proper MDP.
---

## Q-Learning Algorithm

The implementation uses standard tabular Q-learning with ε-greedy exploration.

State–action values are stored as a Python dictionary:

```python
Q[(row, col, mask, action)] -> float
