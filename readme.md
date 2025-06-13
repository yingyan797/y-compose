# Y-Compose: Compositional Reinforcement Learning

This project explores reinforcement learning with boolean and temporal compositionality to improve safety, efficiency, and capability in robotic control/scheduling tasks.

## Overview

Y-Compose is a research framework that combines:

- Policy and value compositionality for efficient skill reuse
- Safety constraints and verification
- Performance optimization through hierarchical learning

## Key Features

- Modular architecture for composing primitive skills into complex behaviors
- Safety-aware planning with formal verification
- Efficient exploration through hierarchical decomposition
- Reusable skill library with transfer learning capabilities
- Interative, graphic web UI with custom task creation

## Installation and usage
- Download repository and install required Python packages.
- Use app.py to launch the web application for creating 2D terrain or analyze LTLf formula
- Use learning.py to run experiments of LTLf tasks and obtain animated trjectories
- Use atomic_task.py to draw policy plot for an atomic task.

## Experiment data
- Policy plots, DFA graphs, and animated trajectories can be seen in the home directory or project/static/training, project/static/policy